import numpy as np
import optuna
import sys
import torch

from docopt import docopt
from ruamel.yaml.comments import CommentedMap

from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer
from neroRL.utils.yaml_parser import OptunaYamlParser, YamlParser
from neroRL.utils.monitor import Tag

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
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
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

    # Load the tuning configuration that features the hyperparameter search space
    tune_config = OptunaYamlParser(tune_config_path).get_config()

    # Setup trial members
    start_seed = tune_config["start_seed"]
    num_seeds = tune_config["num_seeds"]
    upper_threshold = tune_config["upper_threshold"]
    num_updates = tune_config["num_updates"]
    use_eval = tune_config["use_eval"]
    start_eval = tune_config["start_eval"]
    eval_interval = tune_config["eval_interval"]
    lower_threshold = tune_config["lower_threshold"]
    lower_steps = tune_config["lower_steps"]
    summary_frequency = tune_config["summary_frequency"]

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

        print("Launching trial " + str(trial.number))
        print("Trial params:")
        print(suggestions)

        # Create the training config for this trial using the sampled hyperparameters
        trial_config = build_trial_config(suggestions, train_config)

        # Monitor final training duration (total steps) and results
        results = []
        total_steps = []

        # Run all training repetitions using fixed seeds
        for seed in range(start_seed, start_seed + num_seeds):
            # Setup trainer
            if trial_config["trainer"]["algorithm"] == "PPO":
                trainer = PPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            elif trial_config["trainer"]["algorithm"] == "DecoupledPPO":
                trainer = DecoupledPPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            else:
                assert(False), "Unsupported algorithm specified"

            trainer.monitor.log("Run: " + str(seed - start_seed + 1) + "/" + str(num_seeds))

            # Run single training
            for update in range(num_updates):
                # step training
                episode_result, training_stats, formatted_string, update_duration, decayed_hyperparameters = trainer.step(update)

                # log stats once in a while
                if update % summary_frequency == 0:
                    if episode_result:
                        trainer.monitor.log((("{:4} sec={:2} reward={:.2f} std={:.2f} length={:.1f} std={:.2f} ") +
                            (" value={:3f} std={:.3f} adv={:.3f} std={:.3f}")).format(
                            update, update_duration, episode_result["reward_mean"], episode_result["reward_std"],
                            episode_result["length_mean"], episode_result["length_std"], torch.mean(trainer.sampler.buffer.values), torch.std(trainer.sampler.buffer.values),
                            torch.mean(trainer.sampler.buffer.advantages), torch.std(trainer.sampler.buffer.advantages)) +
                            " " + formatted_string)
                    else:
                        trainer.monitor.log("{:4} sec={:2} value={:3f} std={:.3f} adv={:.3f} std={:.3f}".format(
                            update, update_duration, torch.mean(trainer.sampler.buffer.values), torch.std(trainer.sampler.buffer.values),
                            torch.mean(trainer.sampler.buffer.advantages), torch.std(trainer.sampler.buffer.advantages)) +
                            " " + formatted_string)
                    training_stats = {
                    **training_stats,
                    **decayed_hyperparameters,
                    "advantage_mean": (Tag.EPISODE, torch.mean(trainer.sampler.buffer.advantages)),
                    "value_mean": (Tag.EPISODE, torch.mean(trainer.sampler.buffer.values)),
                    "sequence_length": (Tag.OTHER, trainer.sampler.buffer.actual_sequence_length),}
                    # write training statistics to tensorboard
                    trainer.monitor.write_training_summary(update, training_stats, episode_result)

                # eval?
                if use_eval:
                    if update >= start_eval:
                        if update % eval_interval == 0 or update + 1 == num_updates:
                            eval_duration, evaluation_result = trainer.evaluate()
                            # prune upper threshold based on evaluation score
                            if episode_result["reward_mean"] >= upper_threshold:
                                results.append(episode_result["reward_mean"])
                                total_steps.append(update)
                                trainer.monitor.log("Trial succeeded by reaching the threshold earlier.")
                                break
                else:
                    # prune upper threshold based on training score
                    if episode_result["reward_mean"] >= upper_threshold:
                        results.append(episode_result["reward_mean"])
                        total_steps.append(update)
                        break

                # TODO prune entire trial based on lower threshold

            # Save model and clean up training run
            trainer.close()
            del trainer

        # Process results across all seeds
        results = np.asarray(results)
        total_steps = np.asarray(total_steps)
        weights = (1 - total_steps / num_updates) + 1 # scale up scores that surpassed the upper threshold earlier
        result = weights * results
        result = result.mean()
        print("RESULT AGGREGATION")
        print("Raw results")
        print(results)
        print("Raw steps")
        print(total_steps)
        print("Weights")
        print(weights)
        print("Weighted results")
        print(weights * results)
        print("Aggregated result")
        print(result)
        return result

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
