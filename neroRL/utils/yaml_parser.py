import sys
from ruamel.yaml import YAML

class YamlParser:
    """The YamlParser parses a yaml file containing parameters for the environment, model, evaulation, and trainer.
    The data is parsed during initialization.
    Retrieve the parameters using the get_config function.

    The data can be accessed like:
    parser.get_config()["environment"]["name"]
    """

    def __init__(self, path = "./configs/default.yaml"):
        """Loads and prepares the specified config file.
        
        Arguments:
            path {str} -- Yaml file path to the to be loaded config (default: {"./configs/default.yaml"})
        """
        # Load the config file
        stream = open(path, "r")
        yaml = YAML()
        yaml_args = yaml.load_all(stream)
        
        # Final contents of the config file will be added to a dictionary
        self._config = {}

        # Prepare data
        for data in yaml_args:
            self._config = dict(data)

        # Process config, like adding missing keys with default values
        self._process_config()

    def _process_config(self):
        """Ensures that the config is complete. If incomplete, default values will be applied to missing entries.
        """
        # Default parameters
        env_dict = {
            "type": "Minigrid",
            "name": "MiniGrid-Empty-Random-6x6-v0",
            "frame_skip": 1,
            "last_action_to_obs": False,
            "last_reward_to_obs": False,
            "obs_stacks": 1,
            "grayscale": False,
            "resize_vis_obs": [84, 84],
            "reset_params": {"start-seed": 0, "num-seeds": 100}
        }

        model_dict = {
            "load_model": False,
            "model_path": "",
            "checkpoint_interval": 50,
            "activation": "relu",
            "share_parameters": True,
            "pi_estimate_advantages": False,
            "encoder": "cnn",
            "hidden_layer": "default",
            "num_hidden_layers": 2,
            "num_hidden_units": 512
        }

        eval_dict = {
            "evaluate": False,
            "n_workers": 3,
            "seeds": [1001, 1002, 1003, 1004, 1005],
            "interval": 50
        }

        trainer_dict = {
            "algorithm": "PPO",
            "resume_at": 0,
            "gamma": 0.99,
            "lamda": 0.95,
            "updates": 1000,
            "epochs": 4,
            "n_workers": 16,
            "worker_steps": 256,
            "n_mini_batch": 4,
            "value_coefficient": 0.25,
            "learning_rate_schedule": {"initial": 3.0e-4},
            "beta_schedule": {"initial": 0.001},
            "clip_range_schedule": {"initial": 0.2}
        }

        # Check if keys of the parent dictionaries are available, if not apply defaults from above
        if not "environment" in self._config:
            self._config["environment"] = env_dict
        if not "model" in self._config:
            self._config["model"] = model_dict
        if not "evaluation" in self._config:
            self._config["evaluation"] = eval_dict
        if not "trainer" in self._config:
            self._config["trainer"] = trainer_dict

        # Override defaults with parsed dict and check for completeness
        for key, value in self._config.items():
            if key == "environment":
                for k, v in value.items():
                    env_dict[k] = v
                self._config[key] = env_dict
            elif key == "model":
                for k, v in value.items():
                    model_dict[k] = v
                self._config[key] = model_dict
            elif key == "evaluation":
                for k, v in value.items():
                    eval_dict[k] = v
                self._config[key] = eval_dict
            elif key == "trainer":
                for k, v in value.items():
                    trainer_dict[k] = v
                self._config[key] = trainer_dict
                # Check final hyperparameter
                if "final" not in value["learning_rate_schedule"]:
                    trainer_dict["learning_rate_schedule"]["final"] = trainer_dict["learning_rate_schedule"]["initial"]
                if "final" not in value["beta_schedule"]:
                    trainer_dict["beta_schedule"]["final"] = trainer_dict["beta_schedule"]["initial"]
                if "final" not in value["clip_range_schedule"]:
                    trainer_dict["clip_range_schedule"]["final"] = trainer_dict["clip_range_schedule"]["initial"]
                # Check power
                if "power" not in value["learning_rate_schedule"]:
                    trainer_dict["learning_rate_schedule"]["power"] = 1.0
                if "power" not in value["beta_schedule"]:
                    trainer_dict["beta_schedule"]["power"] = 1.0
                if "power" not in value["clip_range_schedule"]:
                    trainer_dict["clip_range_schedule"]["power"] = 1.0
                # Check max_decay_steps
                if "max_decay_steps" not in value["learning_rate_schedule"]:
                    trainer_dict["learning_rate_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                if "max_decay_steps" not in value["beta_schedule"]:
                    trainer_dict["beta_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                if "max_decay_steps" not in value["clip_range_schedule"]:
                    trainer_dict["clip_range_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                self._config[key] = trainer_dict
            
            # Check if the model dict contains a recurrence dict
            # If no recurrence dict is available, it is assumed that a recurrent policy is not used
            # In the other case check for completeness and apply defaults if necessary
            if "recurrence" in self._config["model"]:
                if "layer_type" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["layer_type"] = "gru"
                if "sequence_length" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["sequence_length"] = 32
                if "hidden_state_size" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["hidden_state_size"] = 128
                if "hidden_state_init" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["hidden_state_init"] = "zero"
                if "reset_hidden_state" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["reset_hidden_state"] = True

    def get_config(self):
        """ 
        Returns:
            {dict} -- Nested dictionary that contains configs for the environment, model, evaluation and trainer.
        """
        return self._config

class GridSearchYamlParser:
    """The GridSearchYamlParser parses a yaml file containing parameters for tuning hyperparameters based on grid search.
    The data is parsed during initialization.
    Retrieve the parameters using the get_config function.

    The data can be accessed like:
    parser.get_config()["search_space"]["worker_steps"]
    """

    def __init__(self, path = "./configs/tune/example.yaml"):
        """Loads and prepares the specified config file.
        
        Arguments:
            path {str} -- Yaml file path to the to be loaded config (default: {"./configs/tune/search.yaml"})
        """
        # Load the config file
        stream = open(path, "r")
        yaml = YAML()
        yaml_args = yaml.load_all(stream)
        
        # Final contents of the config file will be added to a dictionary
        self._config = {}

        # Prepare data
        for data in yaml_args:
            self._config = dict(data)

    def get_config(self):
        """ 
        Returns:
            {dict} -- Dictionary that contains the config for the grid search.
        """
        return self._config
