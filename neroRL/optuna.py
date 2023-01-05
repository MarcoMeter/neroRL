import numpy as np
import optuna
import sys

from docopt import docopt
from ruamel.yaml.comments import CommentedMap

from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.trainers.policy_gradient.ppo_decoupled import DecoupledPPOTrainer
from neroRL.utils.yaml_parser import OptunaYamlParser, YamlParser

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
    config = YamlParser(config_path).get_config()

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

    # Define objective function
    def objective(trial):
        # Retrieve hyperparameter suggestions
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

        # Establish training config
        trial_config = dict(config)
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

        # Result trial members
        results = []
        steps = []

        # Run training
        for seed in range(start_seed, start_seed + num_seeds):
            if trial_config["trainer"]["algorithm"] == "PPO":
                trainer = PPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            elif trial_config["trainer"]["algorithm"] == "DecoupledPPO":
                trainer = DecoupledPPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            else:
                assert(False), "Unsupported algorithm specified"

            for update in range(num_updates):
                # step training
                episode_result = trainer.step(update)

                # eval?
                if use_eval:
                    if update >= start_eval:
                        if update % eval_interval == 0:
                            episode_result = trainer.evaluate()
                            # prune upper threshold based on evaluation score
                            if episode_result["reward_mean"] >= upper_threshold:
                                results.appendd(episode_result["reward_mean"])
                                steps.append(update)
                                break
                else:
                    # prune upper threshold based on training score
                    if episode_result["reward_mean"] >= upper_threshold:
                        results.appendd(episode_result["reward_mean"])
                        steps.append(update)
                        break

                # TODO prune entire trial based on lower threshold

            # Clean up training run
            trainer.close()
            del trainer

        # Process results across all seeds
        results = np.asarray(results)
        steps = np.asarray(steps)
        weights = (1 - steps / num_updates) + 1 # scale up scores that surpassed the upper threshold earlier
        result = weights * results
        result = result.mean()
        return result

    print("Create/continue study")
    study = optuna.create_study(study_name=run_id, sampler=optuna.samplers.TPESampler(), direction="maximize",
                                storage=storage, load_if_exists=True)
    print("Best params before study")
    print(study.best_params)
    study.optimize(objective, n_trials=num_trials, n_jobs=1)
    print("Study done, best params")
    print(study.best_params)

if __name__ == "__main__":
    main()
