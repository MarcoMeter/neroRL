from unicodedata import name
import optuna
import os
import random
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
        --num-seeds=<n>             Number of seeds [default: 1]
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
        --out=<path>                Where to output the generated config files [default: ./tpe_search/]
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    tune_config_path = options["--tune"]
    num_trials = int(options["--num-trials"])
    num_seeds = int(options["--num-seeds"])
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]
    out_path = options["--out"]

    # Load the original config file
    config = YamlParser(config_path).get_config()

    # Load the tuning configuration that features the hyperparameter search space
    tune_config = OptunaYamlParser(tune_config_path).get_config()

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

        # Run training
        for _ in range(num_seeds):
            seed = random.randint(0, 2 ** 31 - 1)
            if trial_config["trainer"]["algorithm"] == "PPO":
                trainer = PPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            elif trial_config["trainer"]["algorithm"] == "DecoupledPPO":
                trainer = DecoupledPPOTrainer(trial_config, worker_id, run_id, out_path, seed)
            else:
                assert(False), "Unsupported algorithm specified"

            result = trainer.run_training()
            trainer.close()
            del trainer

        return result

    study = optuna.create_study(study_name=run_id, sampler=optuna.samplers.TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=num_trials, n_jobs=1)
    print(study.best_params)

if __name__ == "__main__":
    main()
    