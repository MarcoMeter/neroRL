# Configuration File

The configuration is the backbone of neroRL's toolset.
It specifies the properties of the environment like the dimensions of the visual observations or frame skip.
Also it specifies the path to a model for evaluation, enjoyment or for resuming training.
The architecture of the model can be configured as well.
At last it details the hyperparameters for training.
Each functionality of neroRL makes use of some portion of the config file.
If a provided config file is incomplete, default values (as shown in [yaml_parser.py](../neroRL/utils/yaml_parser.py#L36) are used.

## Environment Config

```
environment:
  # Environment Type (Unity, ObstacleTower, Minigrid, Procgen, CartPole)
  type: "Minigrid"
  # type: "Unity"
  # Environment Name (Unity environments have to specify the path to the executable)
  name: "MiniGrid-Empty-Random-6x6-v0"
  # name: "./UnityBuilds/ObstacleTowerReduced/ObstacleTower"
  # How many frames to repeat the same action
  frame_skip: 1
  # Whether to add the last action (one-hot encoded) to the vector observation space
  last_action_to_obs: False
  # Whether to add the last reward to the vector observation space
  last_reward_to_obs: False
  # Number of past observations, which shall be stacked to the current observation (1 means only the most recent observation)
  obs_stacks: 1
  # Whether to convert RGB visual observations to grayscale
  grayscale: False
  # Whether to rescale visual observations to the specified dimensions
  resize_vis_obs: [84, 84]
  # Whether to add or concatenate positional encodings to the agent's observation
  positional_encoding: False
  # Only add the spotlight_perturbation dictionary if an environment with visual observations shall be perturbed by randomly moving spotlights.
  spotlight_perturbation:
    start-seed: 0
    num-seeds: 100000
    # Whether to use the same seed as the environment
    use-environment-seed: False
    # Number of to be spawned spotlights initially
    initial_spawns: 4
    # Number of spotlights that are spawned throughout the episode at a decayed interval
    num_spawns: 30
    # Number of steps before spawning one spotlight
    initial_spawn_interval: 30
    # Lower threshold for the spotlights spawn interval
    spawn_interval_threshold: 10
    # Decay rate of the spawn interval
    spawn_interval_decay: 0.95
    # Minimum radius of the spotlights
    spot_min_radius: 7.5
    # Maximum radius of the spotlights
    spot_max_radius: 13.75
    # Minimum speed of the spotlights
    spot_min_speed: 0.0025
    # Maximum speed of the spotlights
    spot_max_speed: 0.0075
    # Opacity of the spotlights perturbation surface (0 refers to completely transparent)
    spotlight_opacity: 120
    # Whether to apply a color key to replace black with white
    black_to_white_filter: True
    # Whether to render the spotlight surface as dark shadow or as white fog
    shadow_theme: True
  # Reset parameters for the environment
  # At minimum, these parameters set the range of training seeds
  # Environments, like Obstacle Tower, provide more parameters to alter the environment
  reset_params:
    start-seed: 0
    num-seeds: 100
```

Basically every tool works with an environment.
Using the config, major parameters can be controlled instead of writting a new wrapper for the vanilla environment.
Some environments come with their own distinct reset parameters (see `reset_params`).

## Model Config

```
model:
  # Whether to load a model checkpoint to resume training
  load_model: False
  # File path to the model
  model_path: "path/to/model.pt"
  # Save the model after every n-th update
  checkpoint_interval: 50
  # Set the to be used activation function (elu, leaky_relu, relu, swish, gelu)
  activation: "relu"
  # Set the to be used visual encoder
  vis_encoder: "cnn"
  recurrence:
    # Supported recurrent layers: gru, lstm
    layer_type: "lstm"
    # Number of sequential gru or lstm layers
    num_layers: 1
    # Length of the trained sequences, if set to 0 or smaller the sequence length is dynamically fit to episode lengths
    sequence_length: 32
    # Size of the recurrent layer's hidden state
    hidden_state_size: 128
    # How to initialize the hidden state (zero, one, mean, sample, learned)
    hidden_state_init: "zero"
    # Whether to reset the hidden state before a new episode.
    # Environments that use short episodes are likely to profit from not resetting the hidden state.
    reset_hidden_state: True
    # Whether to add a residual connection between the input and the output of the recurrennt layers
    residual: False
  # Set the to be used vector encoder
  vec_encoder: "linear" # "linear", "none"
  num_vec_encoder_units: 128
  # Set the to be used hidden layer
  hidden_layer: "default"
  # Number of hidden layers
  num_hidden_layers: 1
  # Number of hidden units
  num_hidden_units: 512
```

Every tool may use the model_path.
During training models are saved to the `checkpoints` directory.
The [model's architecture](model.md) can be customized to support varying observation encoders or even recurrent or transformer policies.

## Evaluation Config

```
evaluation:
  # Whether to evaluate the model during training
  evaluate: False
  # Number of environments that are used
  n_workers: 3
  # Evaluation seeds (each worker performs on every seed: in this case, overall 30 episodes are used for evaluation (n_workers * seeds))
  seeds:
    start-seed: 100000
    num-seeds: 10
    # Use the explicit-seeds key to override the range of seeds in case of evaluating specific seeds
    # explicit-seeds: [1001, 1002, 1003, 1004, 1005]
  # Evaluate the model after every n-th update during training
  interval: 50
```

This configuration is shared among the training and the evaluation.
Evaluation can be configured to be run during training or as stand-alone process.

## Sampler Config

```
sampler:
  # Number of environments that are used for sampling data
  n_workers: 16
  # Number of steps an agent samples data in each environment (batch_size = n_workers * worker_steps)
  worker_steps: 256
```

The sampler config describes the behavior of sampling data by executing agent-environment interactions.
Multiplying both parameters determines the batch size.

## Trainer Config

### PPO

```
trainer:
  # Which algorithm to use. For now, PPO is supported.
  algorithm: "PPO"
  # On which step to resume the training. This affects the hyperparameter schedules only.
  resume_at: 0
  # Discount factor
  gamma: 0.99
  # Regularization parameter used when calculating the Generalized Advantage Estimation (GAE)
  lamda: 0.95
  # Number of PPO update cycles that shall be done (one whole cycle comprises n epochs of m mini_batch updates)
  updates: 1000
  # Number of times that the whole batch of data is used for optimization using PPO
  # Each epoch trains on a random permutation of the sampled training batch
  epochs: 4
  # Number of mini batches that are trained throughout one epoch
  # In case of using a recurrent net, this has to be a multiple of n_workers.
  n_mini_batches: 4
  # Wether to normalize the advantages on "minibatch" level, "batch" level or not at all ("no").
  advantage_normalization: "minibatch"
  # Coefficient of the loss of the value function (i.e. critic)
  value_coefficient: 0.25
  # Strength of clipping the loss gradients
  max_grad_norm: 0.5
  # Polynomial Decay Schedules
  # Learning Rate
  learning_rate_schedule:
    initial: 3.0e-4
    final: 3.0e-4
    power: 1.0
    max_decay_steps: 1000
  # Beta represents the entropy bonus coefficient
  beta_schedule:
    initial: 0.001
    final: 0.0005
    power: 1.0
    max_decay_steps: 800
  # Strength of clipping optimizations done by the PPO algorithm
  clip_range_schedule:
    initial: 0.2
    final: 0.2
    power: 1.0
    max_decay_steps: 1000
```

These are basically all the hyperparameters, which can be configured for training an agent using PPO.
Polynomial decay schedules can be applied to the clip_range, the entropy bonus coefficient and the learning rate.