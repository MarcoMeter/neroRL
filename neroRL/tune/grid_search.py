import copy
import itertools
import os

from ruamel.yaml import YAML
from neroRL.trainers.PPO.trainer import PPOTrainer

class GridSearch:
    """To conduct a grid search for hyperparameter tuning, this class permutes the hyperparameter choices of the to be searched space.
    Further, the permutations are used to modify an exisiting config.
    The final configs can be dumped to files or used to sequentially run training sessions based on these.
    """
    def __init__(self, base_config, tune_config):
        """Retrieves the configuration data and creates all permutations of the hyperparameter search space.

        Arguments:
            base_config {dict} -- Original configuration
            tune_config {dict} -- Configuration that provides the to be permuted hyperparameter choices
        """
        # Original config that is used to source all other values
        self.base_config = base_config

        # Permute all parameters of the tuning config
        permutations = self._permute(tune_config)

        # Create a new config for each permutation
        self._final_configs = []
        # Store a tuple of the final config and the used permuted hyperparameters
        for permutation in permutations:
            self._final_configs.append((self._generate_config(permutation), permutation))

    def _permute(self, tune_config):
        """Permutes all parameters as specified by the tuning config.

        Arguments:
            tune_config {dict}: The to be permuted tuning config

        Returns:
            {list}: Returns a list that contains all possible permutations of the provided tuning config
        """
        # Permute each subset individually
        permutations = {}
        for key in tune_config:
            keys, values = zip(*tune_config[key].items())
            permutations[key] = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Permute subsets altogether
        keys, values = zip(*permutations.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return permutations

    def _generate_config(self, permutation):
        """Generates a new config by modifying the original config using a single permutation of the hyperparmeter choices.

        Arguments:
            permutation {dict}: Single permutation that is used to modify the original config

        Returns:
            {dict}: New config that uses a single hyperparameter permutation
        """
        # Duplicate the original config file
        new_config = copy.deepcopy(self.base_config) # A shallow copy does not work here

        # Apply general singular hyperparameters
        if "hyperparameters" in permutation:
            # It is assumed that the nested config has a depth of 2
            # Depth 0, e.g. environment, model, trainer, ...
            for key, value in new_config.items():
                # Depth 1, e.g. algorithm, gamma, lamda, ...
                if isinstance(value, dict):
                    for ke, val in value.items():
                        # Depth 2, e.g. sequence_length, hidden_state_size, ...
                        if isinstance(val, dict):
                            for k, v in val.items():
                                # Apply new value
                                if k in permutation["hyperparameters"]:
                                    new_config[key][ke][k] = permutation["hyperparameters"][k]
                        else:
                            # Apply new value
                            if ke in permutation["hyperparameters"]:
                                new_config[key][ke] = permutation["hyperparameters"][ke]
            else:
                pass

        # Apply decay schedules
        for key in list(permutation.keys()):
            if key != "hyperparameters":
                for k, v in new_config["trainer"][key].items():
                    if k in permutation[key]:
                        new_config["trainer"][key][k] = permutation[key][k]
                        
        return new_config

    def write_permuted_configs_to_file(self, root_path):
        """Write all permuted configurations to files.
        All config files are named afters its ID.
        These will be plased in the configs directory of the to be created root directoy.
        In addition, an info.txt is being created that shows the used permutation for each file.

        Arguments:
            root_path {str}: Name of the target root directory
        """
        # Create directories
        if not os.path.exists(root_path) or not os.path.exists(root_path + "configs/"):
            os.makedirs(root_path + "configs/")

        # Write config files
        yaml=YAML()
        yaml.default_flow_style = False
        for i, item in enumerate(self._final_configs):
            config, permutation = item
            # Add the permutation to the config to easily keep track of it
            config["permutation"] = permutation
            # Write config to file, but check whethere the file already exists
            f = open(root_path + "configs/" + str(i) + ".yaml", "x")
            yaml.dump(config, f)
            # Create/Append info.txt to store the config's ID along with its used permutation
            f = open(root_path + "info.txt", "a")
            f.write(str(i) + ": " + str(permutation) +"\n\n")

    def run_trainings_sequentially(self, num_repetitions = 1, run_id="default", worker_id = 2, low_mem_fix = False, out_path = "./"):
        """Conducts one training session per generated config file.
        All training sessions can be repeated n-times.

        Args:
            num_repetitions {int}: Number of times a training session is being repeated. Defaults to 1.
            run_id {str}: The used string to name various things like the directory of the checkpoints. Defaults to "default".
            worker_id {int}: Sets the communication port for Unity environments. Defaults to 2.
            low_mem_fix {bool}: Whether to load one mini_batch at a time. This is needed for GPUs with low memory (e.g. 2GB). Defaults to False.
            out_path {string}: Target location to save files such as checkpoints and summaries. Defaults to "./"
        """
        print("Initialize Grid Search Training")
        print("Num training runs: " + str(num_repetitions * len(self._final_configs)))
        count = 0
        for i in range(num_repetitions):
            for j, item in enumerate(self._final_configs):
                config, permutation = item
                # Add the permutation to the config to easily keep track of it
                config["permutation"] = permutation
                # Init trainer
                if config["trainer"]["algorithm"] == "PPO":
                    trainer = PPOTrainer(config, worker_id, run_id + "_" + str(i) + "_" + str(j), low_mem_fix, out_path)
                else:
                    assert(False), "Unsupported algorithm specified"

                # Start training
                trainer.run_training()

                # Clean up after training
                trainer.close()

                count += 1
                print("Completed training sessions: " + str(count) + "/" + str(num_repetitions * len(self._final_configs)))
