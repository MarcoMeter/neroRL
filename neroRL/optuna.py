import random
import numpy as np
import optuna
import sys
import torch

from docopt import docopt
from ruamel.yaml.comments import CommentedMap

from neroRL.evaluator import Evaluator
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer
from neroRL.utils.yaml_parser import OptunaYamlParser, YamlParser
from neroRL.utils.monitor import TuningMonitor, Tag
from neroRL.utils.utils import aggregate_episode_results, set_library_seeds

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        ntune [options]
        ntune --help

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
    num_repetitions = tune_config["repetitions"]
    num_updates = tune_config["num_updates"]
    trainer_period = tune_config["trainer_period"]
    seed = tune_config["seed"]
    # Sampled seed if a value smaller than 0 was submitted
    if seed < 0:
        seed = random.randint(0, 2 ** 31 - 1)

    # Define objective function
    def objective(trial):
        # Sample hyperparameters
        suggestions = {}
        for key, value in tune_config.items():
            if key == "categorical":
                for k, v in tune_config["categorical"].items():
                    suggestions[k] = trial.suggest_categorical(k, v)
            elif key == "uniform":
                for k, v in tune_config["uniform"].items():
                    suggestions[k] = trial.suggest_float(k, *v, log=False)
            elif key == "loguniform":
                for k, v in tune_config["loguniform"].items():
                    suggestions[k] = trial.suggest_float(k, *v, log=True)

        # Create the training config for this trial using the sampled hyperparameters
        trial_config = build_trial_config(suggestions, train_config)

        # Init monitor
        run_id = trial_config["run_id"] + "_" + str(trial.number)
        monitor = TuningMonitor(out_path, run_id, trial_config["worker_id"], num_repetitions)
        monitor.log("Trial Seed: " + str(seed))
        # Start logging the training setup
        monitor.log("Trial Config:")
        for key in trial_config:
            if type(trial_config[key]) is dict:
                monitor.log("\t" + str(key) + ":")
                if type(trial_config[key]) is dict:
                    for k, v in trial_config[key].items():
                        monitor.log("\t" * 2 + str(k) + ": " + str(v))
        monitor.log("Trial Suggestions:")
        for k, v in suggestions.items():
                monitor.log("\t" + str(k) + ": " + str(v))

        # Determine cuda availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")
        cpu_device = torch.device("cpu")

        # Instantiate as many trainers as there are repetitions
        set_library_seeds(seed)
        trainers = []
        monitor.log("Initializing " + str(num_repetitions) + " trainers")
        for t in range(num_repetitions):
            train_config["worker_id"] += 50
            monitor.log("\t" + "Run id: " + run_id + " Seed: " + str(seed))
            trainer = PPOTrainer(trial_config, device, train_config["worker_id"], run_id, out_path, seed)
            trainer.to(device) # Move models and tensors to CPU
            trainers.append(trainer)
            monitor.write_hyperparameters(train_config, t)

        # Log environment specifications
        monitor.log("Environment specs:")
        monitor.log("\t" + "Visual Observation Space: " + str(trainers[0].vis_obs_space))
        monitor.log("\t" + "Vector Observation Space: " + str(trainers[0].vec_obs_space))
        monitor.log("\t" + "Action Space Shape: " + str(trainers[0].action_space_shape))
        monitor.log("\t" + "Max Episode Steps: " + str(trainers[0].max_episode_steps))

        # Init evaluator
        monitor.log("Initializing Evaluator")
        evaluator = Evaluator(trial_config, trial_config["model"], worker_id, trainers[0].vis_obs_space,
                                trainers[0].vec_obs_space, trainers[0].max_episode_steps)
        
        # Execute all training runs "concurrently" (i.e. one trainer runs for the specified period and then training moves on with the next trainer)
        monitor.log("Executing trial using " + str(device))
        for cycle in range(0, num_updates, trainer_period):
            # Lists to monitor results
            eval_episode_infos = []
            monitor.log("Updates Done: {:4}".format(cycle))
            for c, trainer in enumerate(trainers):
                train_episode_infos = []
                training_stats = None
                update_durations = []
                trainer.to(device) # Move current trainer model and tensors to GPU
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
                monitor.log((("T{:1} sec={:3} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                    (" value={:3f} adv={:.3f} loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}")).format(
                    c, sum(update_durations), train_results["reward_mean"], train_results["reward_std"],
                    train_results["length_mean"], train_results["length_std"], training_stats["value_mean"][1],
                    training_stats["advantage_mean"][1], training_stats["loss"][1], training_stats["policy_loss"][1],
                    training_stats["value_loss"][1], training_stats["entropy"][1]))

                # Evaluate
                eval_duration, eval_episode_infos = evaluator.evaluate(trainer.model, device)
                eval_results = aggregate_episode_results(eval_episode_infos)
                eval_episode_infos.extend(eval_episode_infos)
                # Log evaluation results for each individual trainer
                # result_string = "EVAL T{:1} sec={:3} reward={:.2f} std={:.2f} length={:.1f} std={:.2f}".format(c, eval_duration,
                #                 eval_results["reward_mean"], eval_results["reward_std"], eval_results["length_mean"],
                #                 eval_results["length_mean"])
                # additional_string = ""
                # if "success_mean" in eval_results.keys():
                #     additional_string = " success={:.2f} std={:.2f}".format(eval_results["success_mean"], eval_results["success_std"])
                # monitor.log(result_string + additional_string)

                # Write to tensorboard
                monitor.write_training_summary(cycle + trainer_period, training_stats, train_results, c)
                monitor.write_eval_summary(cycle + trainer_period, eval_results, c)

                # After the training period, move current trainer model and tensors to CPU
                trainer.to(cpu_device)

            # Aggregate and log evaluation results
            eval_results = aggregate_episode_results(eval_episode_infos)
            result_string = "EVAL reward={:.2f} std={:.2f} length={:.1f} std={:.2f}".format(eval_results["reward_mean"],
                                        eval_results["reward_std"], eval_results["length_mean"], eval_results["length_mean"])
            additional_string = ""
            if "success_mean" in eval_results.keys():
                additional_string = " success={:.2f} std={:.2f}".format(eval_results["success_mean"], eval_results["success_std"])
            monitor.log(result_string + additional_string)

            # Replace oldest checkpoints

            # Report evaluation result to optuna to allow for pruning
            trial.report(cycle + trainer_period, eval_results["reward_mean"])

        # Finish up trial
        monitor.log("Closing trainer . . .")
        for trainer in trainers:
            trainer.close()
            evaluator.close()
            monitor.close()        

        # Return final result
        return eval_results["reward_mean"]

    print("Create/continue study")
    study = optuna.create_study(study_name=run_id, sampler=optuna.samplers.TPESampler(), direction="maximize",
                                storage=storage, load_if_exists=True)
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
                if "learning_rate" in k and "learning_rate" in suggestions:
                    trial_config[key]["learning_rate_schedule"]["initial"] = suggestions["learning_rate"]
                    trial_config[key]["learning_rate_schedule"]["final"] = suggestions["learning_rate"]
                elif "beta" in k and "beta" in suggestions:
                    trial_config[key]["beta_schedule"]["initial"] = suggestions["beta"]
                    trial_config[key]["beta_schedule"]["final"] = suggestions["beta"]
                elif "clip_range" in k and "clip_range" in suggestions:
                    trial_config[key]["clip_range_schedule"]["initial"] = suggestions["clip_range"]
                    trial_config[key]["clip_range_schedule"]["final"] = suggestions["clip_range"]
            else:
                if k in suggestions:
                    trial_config[key][k] = suggestions[k]
    return trial_config

if __name__ == "__main__":
    main()
