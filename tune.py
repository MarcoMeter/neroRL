"""
This program creates all hyperparameters permutations based on a tuning config.
It either executes all training runs sequentially or just outputs each permutation as a config file.
"""

from docopt import docopt
from neroRL.tune.grid_search import GridSearch
from neroRL.utils.yaml_parser import GridSearchYamlParser, YamlParser

def main():
    # Docopt command line arguments
    _USAGE = """
    Usage:
        tune.py [options]
        tune.py --help

    Options:
        --config=<path>             Path to the config file [default: ./configs/default.yaml].
        --tune=<path>               Path to the config file that features the hyperparameter search space for tuning [default: ./configs/tune/example.yaml]
        --num-repetitions=<n>       How many times to repeat the training of one config [default: 1]
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
        --low-mem-fix               Whether to load one mini_batch at a time to the GPU's memory. [default: False].
        --generate-only             Whether to only generate the config files [default: False]
        --out=<path>                Where to output the generated config files [default: ./grid_search/]
    """
    options = docopt(_USAGE)
    config_path = options["--config"]
    tune_config_path = options["--tune"]
    num_repetitions = int(options["--num-repetitions"])
    worker_id = int(options["--worker-id"])
    run_id = options["--run-id"]
    low_mem_fix = options["--low-mem-fix"]
    generate_only = options["--generate-only"]
    out_path = options["--out"]

    # Load the original config file
    config = YamlParser(config_path).get_config()

    # Load the tuning configuration that features the hyperparameter search space
    tune_config = GridSearchYamlParser(tune_config_path).get_config()

    # Init GridSearch: it creates config files for each permutation of the hyperparameter search space
    grid_search = GridSearch(config, tune_config)

    # Retrieve permuted configs
    configs = grid_search.get_permuted_configs()

    # Generate configs or run training sessions sequentially
    if generate_only:
        grid_search.write_permuted_configs_to_file(out_path)
    else:
        grid_search.run_trainings_sequentially(num_repetitions, run_id, worker_id, low_mem_fix, out_path)

if __name__ == "__main__":
    main()
    