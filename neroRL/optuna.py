"""Hyperparameter search done with optuna

This search can be imagined as multiple 1D grid searches with adding TPE searched trials optionally.
This is the concept:

1. Define a very narrow categorical search space
    e.g. 10 hyperparameters with 3 choices
2. Use percentile pruner to keep the top 25.0 percentile
3. Enqueue baseline trial
4. Extrapolate baseline trial by every single choice among the search space
    Note that only one parameter at a time is changed
    Based on the example, 30 trials - 10 should be enqueued by this step (10 is subtraced because of duplicates)
5. Enqueue extrapolated trials
6. (optional) optuna can run furhter trials by sampling from the search space using TPE

Note that trials are not repeated as this very expensive.
Trials that are already weak and are thus pruned do not need repetitions.
The top trials need to be repeated manually to furhter narrow the best solution.

Trials can be synchronized via MySQL.
"""
import random
import numpy as np
import optuna
import os
import sys
import time
import torch

from docopt import docopt
from ruamel.yaml.comments import CommentedMap

from neroRL.evaluator import Evaluator
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.utils.yaml_parser import OptunaYamlParser, YamlParser
from neroRL.utils.monitor import TrainingMonitor
from neroRL.utils.utils import aggregate_episode_results, set_library_seeds

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        noptuna [options]
        noptuna --help

    Options:
        --config=<path>             Path to the config file [default: ./configs/default.yaml].
        --tune=<path>               Path to the config file that features the hyperparameter search space for tuning [default: ./configs/tune/optuna.yaml]
        --num-trials=<n>            Number of trials [default: 1]
        --db=<path>     	        MySQL URL [default: mysql://root@localhost/optuna]
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: no-id].
        --out=<path>                Where to output the generated config files [default: ./tpe_search/]
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    tune_config_path = options["--tune"]
    num_trials = int(options["--num-trials"])
    storage = options["--db"]
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]
    out_path = options["--out"]

    # If the db flag is not set, create the study in memory only
    use_storage = False
    for i, arg in enumerate(sys.argv):
        if "--db" in arg:
            use_storage = True
            break
    if not use_storage:
        storage = None

    # Load the original config file
    train_config = YamlParser(config_path).get_config()
    train_config["run_id"] = run_id # for some reason run_id and worker_id cannot be used in objective(trial)
    train_config["worker_id"] = worker_id

    # Load the tuning configuration that features the hyperparameter search space
    tune_config = OptunaYamlParser(tune_config_path).get_config()

    # Setup trial members
    num_updates = tune_config["num_updates"]
    trainer_period = tune_config["trainer_period"]
    seed = tune_config["seed"]
    # Sampled seed if a value smaller than 0 was submitted
    if seed < 0:
        seed = random.randint(0, 2 ** 31 - 1)

    # Define objective function
    def objective(trial):
        start_time = time.time()
        # Sample hyperparameters
        suggestions = {}
        for key, value in tune_config.items():
            if key == "categorical":
                for k, v in tune_config["categorical"].items():
                    suggestions[k] = trial.suggest_categorical(k, v)
            # We usually use only a very narrow categorical search space
            # elif key == "uniform":
            #     for k, v in tune_config["uniform"].items():
            #         suggestions[k] = trial.suggest_float(k, *v, log=False)
            # elif key == "loguniform":
            #     for k, v in tune_config["loguniform"].items():
            #         suggestions[k] = trial.suggest_float(k, *v, log=True)

        # Create the training config for this trial using the sampled hyperparameters
        trial_config = build_trial_config(suggestions, train_config)

        # Init monitor
        run_id = trial_config["run_id"] + "_" + str(trial.number)
        monitor = TrainingMonitor(out_path, run_id, trial_config["worker_id"])   
        # Start logging the training setup
        monitor.log("Training Seed: " + str(seed))
        monitor.log("Trial " + str(trial.number)  + " Suggestions:")
        for k, v in suggestions.items():
            monitor.log("\t" + str(k) + ": " + str(v))
        monitor.log("Trial Config:")
        for key in trial_config:
            if type(trial_config[key]) is dict:
                monitor.log("\t" + str(key) + ":")
                if type(trial_config[key]) is dict:
                    for k, v in trial_config[key].items():
                        monitor.log("\t" * 2 + str(k) + ": " + str(v))

        # Determine cuda availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        # Instantiate one trainer
        set_library_seeds(seed)
        monitor.log("Initializing trainer")
        monitor.log("\t" + "Run id: " + run_id + " Seed: " + str(seed))
        trainer = PPOTrainer(trial_config, device, train_config["worker_id"], run_id, out_path, seed)
        monitor.write_hyperparameters(train_config)

        # Log environment specifications
        monitor.log("Environment specs:")
        monitor.log("\t" + "Visual Observation Space: " + str(trainer.vis_obs_space))
        monitor.log("\t" + "Vector Observation Space: " + str(trainer.vec_obs_space))
        monitor.log("\t" + "Action Space Shape: " + str(trainer.action_space_shape))
        monitor.log("\t" + "Max Episode Steps: " + str(trainer.max_episode_steps))

        # Init evaluator
        monitor.log("Initializing Evaluator")
        evaluator = Evaluator(trial_config, trial_config["model"], worker_id, trainer.vis_obs_space,
                                trainer.vec_obs_space, trainer.max_episode_steps)
        
        # Execute all training runs "concurrently" (i.e. one trainer runs for the specified period and then training moves on with the next trainer)
        monitor.log("Executing trial using " + str(device) + " . . .")
        latest_checkpoint = ""

        try:
            for cycle in range(0, num_updates, trainer_period):
                # Lists to monitor results
                eval_episode_infos = []
                train_episode_infos = []
                training_stats = None
                update_durations = []
                for t in range(trainer_period):
                    episode_info, stats, formatted_string, update_duration = trainer.step(cycle + t)
                    train_episode_infos.extend(episode_info)
                    if training_stats is None:
                        training_stats = {}
                        for key, value in stats.items():
                            training_stats[key] = (value[0], [])
                    for key, value in stats.items():
                            training_stats[key][1].append(value[1])
                    update_durations.append(update_duration)
                # Aggregate training results
                for k, v in training_stats.items():
                    training_stats[k] = (v[0], np.asarray(v[1]).mean())
                train_results = aggregate_episode_results(train_episode_infos)
                # Log training results
                update = cycle + trainer_period
                monitor.log((("{:4} sec={:3} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                    (" value={:3f} adv={:.3f} loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}")).format(
                    update, sum(update_durations), train_results["reward_mean"], train_results["reward_std"],
                    train_results["length_mean"], train_results["length_std"], training_stats["value_mean"][1],
                    training_stats["advantage_mean"][1], training_stats["loss"][1], training_stats["policy_loss"][1],
                    training_stats["value_loss"][1], training_stats["entropy"][1]))

                # Evaluate
                eval_duration, eval_episode_infos = evaluator.evaluate(trainer.model, device)
                eval_results = aggregate_episode_results(eval_episode_infos)
                eval_episode_infos.extend(eval_episode_infos)

                # Write to tensorboard
                monitor.write_training_summary(update, training_stats, train_results)
                monitor.write_eval_summary(update, eval_results)

                # Save checkpoint (keep only the latest checkpoint)
                if os.path.isfile(latest_checkpoint):
                    os.remove(latest_checkpoint)
                latest_checkpoint = monitor.checkpoint_path + run_id + "-" + str(update) + ".pt"
                trainer.save_checkpoint(update, latest_checkpoint[:-3])

                # Log evaluation results
                result_string = "{:4} sec={:3} eval_reward={:.2f} std={:.2f} eval_length={:.1f} std={:.2f}".format(update, eval_duration,
                        eval_results["reward_mean"], eval_results["reward_std"], eval_results["length_mean"], eval_results["length_mean"])
                additional_string = ""
                if "success_mean" in eval_results.keys():
                    additional_string = " success={:.2f} std={:.2f}".format(eval_results["success_mean"], eval_results["success_std"])
                monitor.log(result_string + additional_string)

                # Report evaluation result to optuna to allow for pruning
                trial.report(eval_results["reward_mean"], update)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    monitor.log("Pruning trial " + str(trial.number) + " at update " + str(update) + " . . .")
                    hours, remainder = divmod(time.time() - start_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    monitor.log("Trial duration: {:.0f}h {:.0f}m {:.2f}s".format(hours, minutes, seconds))
                    monitor.log("Closing trainer . . .")
                    try:
                        trainer.close()
                        evaluator.close()
                        monitor.close()
                        del monitor, trainer, evaluator
                    except:
                        pass
                    raise optuna.exceptions.TrialPruned()

            # Finish up trial
            hours, remainder = divmod(time.time() - start_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            monitor.log("Trial " + str(trial.number) + " completed")
            monitor.log("Trial duration: {:.0f}h {:.0f}m {:.2f}s".format(hours, minutes, seconds))
            monitor.log("Closing trainer . . .")
            try:
                trainer.close()
                evaluator.close()
                monitor.close()
                del monitor, trainer, evaluator
            except:
                pass
        except:
            monitor.logger.exception("Trial " + str(trial.number) + " failed")
            monitor.log("Closing trainer . . .")
            try:
                trainer.close()
                evaluator.close()
                monitor.close()
                del monitor, trainer, evaluator
            except:
                pass
            # Return None to indicate that the trial failed
            return None

        # Return final result
        return eval_results["reward_mean"]

    print("Create/continue study")
    pruner = optuna.pruners.PercentilePruner(25.0, n_startup_trials=tune_config["n_startup_trials"], n_warmup_steps=tune_config["n_warmup_steps"])
    study = optuna.create_study(study_name=run_id, sampler=optuna.samplers.TPESampler(), direction="maximize",
                                pruner=pruner, storage=storage, load_if_exists=True)
    
    # Extract baseline suggestions from the provided train config and the suggested categorical search space
    baseline_suggestions = {}
    for key in tune_config["categorical"]:
        # Find values of the search space in the baseline training config
        # Be cautious with the schedules of learning rate, beta and clip range
        if key in ("learning_rate", "clip_range", "beta"):
            value = find_value_in_nested_dict(train_config, key + "_schedule")
            assert value is not None
            value = value["initial"] # The schedule is a dictionary with the keys (initial, final, power, max_decay_steps)
            baseline_suggestions[key] = value # We only consider the initial value of the schedule
        else:
            value = find_value_in_nested_dict(train_config, key)
            assert value is not None
        # Add key and value to baseline suggestions
        baseline_suggestions[key] = value
    # Enqueue baseline trial
    study.enqueue_trial(baseline_suggestions, skip_if_exists=True)

    # Extrapolate baseline suggestions based on each single parameter choice of the search space
    # We consider this as a 1D grid search
    for key, value in tune_config["categorical"].items():
        for v in value:
            custom_trial_suggestions = baseline_suggestions.copy()
            custom_trial_suggestions[key] = v
            study.enqueue_trial(custom_trial_suggestions, skip_if_exists=True)

    # get a list of failed trials and enqueue them again
    # for trial in study.trials:
    #     if trial.state == optuna.trial.TrialState.FAIL:
    #         # enqueue failed trials again
    #         study.enqueue_trial(trial.params, skip_if_exists=False)

    print("Best params before study")
    try:
        print(study.best_params)
    except:
        print("no study results yet")
    study.optimize(objective, n_trials=num_trials, n_jobs=1)
    print("Study done, best params")
    print(study.best_params)

def build_trial_config(suggestions, train_config):
    trial_config = dict(train_config)
    for key in ["model", "sampler", "trainer"]:
        for k in trial_config[key]:
            if type(trial_config[key][k]) == CommentedMap or type(trial_config[key][k]) == dict:
                for k1 in trial_config[key][k]:
                    if k1 in suggestions:
                        trial_config[key][k][k1] = suggestions[k1]
                        # Override num_hidden_units if embed_dim or hidden_state_size is changed
                        if k1 in ("embed_dim", "hidden_state_size"):
                            trial_config["model"]["num_hidden_units"] = suggestions[k1]
                # Only modify the initial value of the schedules
                if "learning_rate" in k and "learning_rate" in suggestions:
                    trial_config[key]["learning_rate_schedule"]["initial"] = suggestions["learning_rate"]
                    # trial_config[key]["learning_rate_schedule"]["final"] = suggestions["learning_rate"]
                elif "beta" in k and "beta" in suggestions:
                    trial_config[key]["beta_schedule"]["initial"] = suggestions["beta"]
                    # trial_config[key]["beta_schedule"]["final"] = suggestions["beta"]
                elif "clip_range" in k and "clip_range" in suggestions:
                    trial_config[key]["clip_range_schedule"]["initial"] = suggestions["clip_range"]
                    trial_config[key]["clip_range_schedule"]["final"] = suggestions["clip_range"]
            else:
                if k in suggestions:
                    trial_config[key][k] = suggestions[k]
    return trial_config

def find_value_in_nested_dict(dictionary, key):
    """
    Recursively search for a key in a nested dictionary and return its value if found.

    Arguments:
        dictionary {dict}: The input dictionary to search in
        key {str}: The key to search for in the dictionary

    Returns:
        Union[Any, None]: The value associated with the key if found, or None if not found.

    """
    if key in dictionary.keys():
        value = dictionary[key]
        return value

    for k, v in dictionary.items():
        if isinstance(v, dict):
            value = find_value_in_nested_dict(v, key)
            if value is not None:
                return value
    return None

if __name__ == "__main__":
    main()
