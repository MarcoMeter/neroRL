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
        try:
            stream = open(path, "r")
            yaml = YAML()
            yaml_args = yaml.load_all(stream)
        except Exception:
            print("Config file not found: {}".format(path))
            if path == "./configs/default.yaml": # Only generate default config if default config is requested and not found
                print("Default config will be generated.")
                yaml_args = []
            else:
                raise Exception("Config file not found.")
        
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
            "positional_encoding": False,
            "reset_params": {"start-seed": 0, "num-seeds": 100},
            "reward_normalization": 0
        }

        model_dict = {
            "load_model": False,
            "model_path": "",
            "checkpoint_interval": 50,
            "activation": "relu",
            "vis_encoder": "cnn",
            "vec_encoder": "linear",
            "num_vec_encoder_units": 128,
            "hidden_layer": "default",
            "num_hidden_layers": 2,
            "num_hidden_units": 512
        }

        eval_dict = {
            "evaluate": False,
            "n_workers": 3,
            "seeds": {"start-seed": 100000, "num-seeds": 10},
            "interval": 50
        }

        sampler_dict = {
            "n_workers": 16,
            "worker_steps": 256
        }

        ppo_dict = {
            "algorithm": "PPO",
            "resume_at": 0,
            "gamma": 0.99,
            "lamda": 0.95,
            "updates": 1000,
            "epochs": 4,
            "refresh_buffer_epoch": -1,
            "n_mini_batches": 4,
            "advantage_normalization": "minibatch",
            "value_coefficient": 0.25,
            "max_grad_norm": 0.5,
            "share_parameters": True,
            "learning_rate_schedule": {"initial": 3.0e-4},
            "beta_schedule": {"initial": 0.001},
            "clip_range_schedule": {"initial": 0.2},
            "obs_reconstruction_schedule": {"initial": 0.0},
            "ground_truth_estimator_schedule": {"initial": 0.0}
        }

        decoupled_ppo_dict = {
            "algorithm": "DecoupledPPO",
            "resume_at": 0,
            "gamma": 0.99,
            "lamda": 0.95,
            "updates": 1000,
            "policy_epochs": 4,
            "n_policy_mini_batches": 4,
            "value_epochs": 9,
            "n_value_mini_batches": 1,
            "advantage_normalization": "minibatch",
            "refresh_buffer_epoch": -1,
            "value_update_interval": 1,
            "max_policy_grad_norm": 0.5,
            "max_value_grad_norm": 0.5,
            "run_threaded": True,
            "policy_learning_rate_schedule": {"initial": 3.0e-4},
            "value_learning_rate_schedule": {"initial": 3.0e-4},
            "beta_schedule": {"initial": 0.001},
            "policy_clip_range_schedule": {"initial": 0.2},
            "value_clip_range_schedule": {"initial": 0.2}
        }

        # Determine which algorithm is used to process the corresponding default config parameters
        # PPO is used as default, if the algorithm cannot be determined.
        if "trainer" in self._config:
            if "algorithm" in self._config["trainer"]:
                if self._config["trainer"]["algorithm"] == "PPO":
                    trainer_dict = ppo_dict
                elif self._config["trainer"]["algorithm"] == "DecoupledPPO":
                    trainer_dict = decoupled_ppo_dict
                elif self._config["trainer"]["algorithm"] == "DecoderTrainer":
                    trainer_dict = ppo_dict
                else:
                    assert(False), "Unsupported algorithm specified"
            else:
                trainer_dict = ppo_dict
        else:
            trainer_dict = ppo_dict

        # Check if keys of the parent dictionaries are available, if not apply defaults from above
        if not "environment" in self._config:
            self._config["environment"] = env_dict
        if not "model" in self._config:
            self._config["model"] = model_dict
        if not "evaluation" in self._config:
            self._config["evaluation"] = eval_dict
        if not "sampler" in self._config:
            self._config["sampler"] = sampler_dict
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
            elif key == "sampler":
                for k, v in value.items():
                    sampler_dict[k] = v
                self._config[key] = sampler_dict
            elif key == "trainer":
                for k, v in value.items():
                    trainer_dict[k] = v
                self._config[key] = trainer_dict

                #  Check beta
                if value["algorithm"] != "DecoderTrainer":
                    if "final" not in value["beta_schedule"]:
                        trainer_dict["beta_schedule"]["final"] = trainer_dict["beta_schedule"]["initial"]
                    if "power" not in value["beta_schedule"]:
                        trainer_dict["beta_schedule"]["power"] = 1.0        
                    if "max_decay_steps" not in value["beta_schedule"]:
                        trainer_dict["beta_schedule"]["max_decay_steps"] = self._config[key]["updates"]

                # Check decaying parameter schedules that have to be differentiated for the available algorithms
                if value["algorithm"] == "PPO" or value["algorithm"] == "DecoderTrainer":
                    # Check learning rate
                    if "final" not in value["learning_rate_schedule"]:
                        trainer_dict["learning_rate_schedule"]["final"] = trainer_dict["learning_rate_schedule"]["initial"]
                    if "power" not in value["learning_rate_schedule"]:
                        trainer_dict["learning_rate_schedule"]["power"] = 1.0
                    if "max_decay_steps" not in value["learning_rate_schedule"]:
                        trainer_dict["learning_rate_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                    # Check clip range
                    if value["algorithm"] != "DecoderTrainer":
                        if "final" not in value["clip_range_schedule"]:
                            trainer_dict["clip_range_schedule"]["final"] = trainer_dict["clip_range_schedule"]["initial"]
                        if "power" not in value["clip_range_schedule"]:
                            trainer_dict["clip_range_schedule"]["power"] = 1.0
                        if "max_decay_steps" not in value["clip_range_schedule"]:
                            trainer_dict["clip_range_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                elif value["algorithm"] == "DecoupledPPO":
                    # Policy learning rate schedule
                    if "final" not in value["policy_learning_rate_schedule"]:
                        trainer_dict["policy_learning_rate_schedule"]["final"] = trainer_dict["policy_learning_rate_schedule"]["initial"]
                    if "power" not in value["policy_learning_rate_schedule"]:
                        trainer_dict["policy_learning_rate_schedule"]["power"] = 1.0
                    if "max_decay_steps" not in value["policy_learning_rate_schedule"]:
                        trainer_dict["policy_learning_rate_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                    # Value learning rate schedule
                    if "final" not in value["value_learning_rate_schedule"]:
                        trainer_dict["value_learning_rate_schedule"]["final"] = trainer_dict["value_learning_rate_schedule"]["initial"]
                    if "power" not in value["value_learning_rate_schedule"]:
                        trainer_dict["value_learning_rate_schedule"]["power"] = 1.0
                    if "max_decay_steps" not in value["value_learning_rate_schedule"]:
                        trainer_dict["value_learning_rate_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                    # Policy clip range schedule
                    if "final" not in value["policy_clip_range_schedule"]:
                        trainer_dict["policy_clip_range_schedule"]["final"] = trainer_dict["policy_clip_range_schedule"]["initial"]
                    if "power" not in value["policy_clip_range_schedule"]:
                        trainer_dict["policy_clip_range_schedule"]["power"] = 1.0
                    if "max_decay_steps" not in value["policy_clip_range_schedule"]:
                        trainer_dict["policy_clip_range_schedule"]["max_decay_steps"] = self._config[key]["updates"]
                    # Value clip range schedule
                    if "final" not in value["value_clip_range_schedule"]:
                        trainer_dict["value_clip_range_schedule"]["final"] = trainer_dict["value_clip_range_schedule"]["initial"]
                    if "power" not in value["value_clip_range_schedule"]:
                        trainer_dict["value_clip_range_schedule"]["power"] = 1.0
                    if "max_decay_steps" not in value["value_clip_range_schedule"]:
                        trainer_dict["value_clip_range_schedule"]["max_decay_steps"] = self._config[key]["updates"]

                # Apply trainer config
                self._config[key] = trainer_dict
            
            # Check if the model dict contains a recurrence dict
            # If no recurrence dict is available, it is assumed that a recurrent policy is not used
            # In the other case check for completeness and apply defaults if necessary
            if "recurrence" in self._config["model"]:
                if "layer_type" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["layer_type"] = "gru"
                if "num_layers" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["num_layers"] = 1
                if "sequence_length" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["sequence_length"] = 32
                if "hidden_state_size" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["hidden_state_size"] = 128
                if "hidden_state_init" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["hidden_state_init"] = "zero"
                if "reset_hidden_state" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["reset_hidden_state"] = True
                if "residual" not in self._config["model"]["recurrence"]:
                    self._config["model"]["recurrence"]["residual"] = False 

            # Check if the model dict contains a transformer dict
            # If no transformer dict is available, it is assumed that a transformer-based policy is not used
            # In the other case check for completeness and apply defaults if necessary
            if "transformer" in self._config["model"]:
                if "num_blocks" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["num_blocks"] = 1
                if "embed_dim" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["embed_dim"] = 512
                if "num_heads" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["num_heads"] = 8
                if "share_heads" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["share_heads"] = True
                if "memory_length" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["memory_length"] = 512
                if "positional_encoding" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["positional_encoding"] = "relative"
                if "layer_norm" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["layer_norm"] = "pre"
                if "init_weights" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["init_weights"] = "xavier"
                if "gtrxl" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["gtrxl"] = False
                if "gtrxl_bias" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["gtrxl_bias"] = 0.0
                if "gtrxl_swap" not in self._config["model"]["transformer"]:
                    self._config["model"]["transformer"]["gtrxl_swap"] = False

            # Check if the model dict contains an obs_decoder dict
            # If no obs_decoder dict is available, it is assumed that an observation decoder is not used
            # In the other case check for completeness and apply defaults if necessary
            if "obs_decoder" in self._config["model"]:
                if "attach_to" not in self._config["model"]["obs_decoder"]:
                    self._config["model"]["obs_decoder"]["attach_to"] = "cnn"
                if "detach_gradient" not in self._config["model"]["obs_decoder"]:
                    self._config["model"]["obs_decoder"]["detach_gradient"] = False

            # Check if the model dict contains an ground_truth_estimator dict
            # If no ground_truth_estimator dict is available, it is assumed that a ground truth estimator is not used
            # In the other case check for completeness and apply defaults if necessary
            if "ground_truth_estimator" in self._config["model"]:
                if "detach_gradient" not in self._config["model"]["ground_truth_estimator"]:
                    self._config["model"]["ground_truth_estimator"]["detach_gradient"] = False

            # Check DAAC if DecoupledPPO
            if "DAAC" in self._config["trainer"]:
                if "adv_coefficient" not in self._config["trainer"]["DAAC"]:
                    self._config["trainer"]["DAAC"] = 0.25

    def get_config(self):
        """ 
        Returns:
            {dict} -- Nested dictionary that contains configs for the environment, model, evaluation and trainer.
        """
        return self._config

class OptunaYamlParser:
    """The OptunaYamlParser parses a yaml file containing parameters for tuning hyperparameters.
    The data is parsed during initialization. Retrieve the parameters using the get_config function.

    The data can be accessed like:
    parser.get_config()["search_type"]["search_parameter"]
    """

    def __init__(self, path = "./configs/tune/optuna.yaml"):
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