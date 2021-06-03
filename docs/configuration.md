# Configuration File

The configuration is the backbone of neroRL's toolset.
It specifies the properties of the environment like the dimensions of the visual observations or frame skip.
Also it specifies the path to a model for evaluation, enjoyment or for resuming training.
At last it details the hyperparameters for training.
Each functionality of neroRL makes use of some portion of the config file.
If a provided config file is incomplete, default values (as shown in [yaml_parser.py](../neroRL/utils/yaml_parser.py#L53) are used.

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
  # Reset parameters for the environment
  # At minimum, these parameters set the range of training seeds
  # Environments, like Obstacle Tower, provide more parameters to alter the environment
  reset_params:
    start-seed: 0
    num-seeds: 100
```

Basically every tool works with an environment.
Using the config, major parameters can be controlled instead of writting a new wrapper for the vanilla environment.

## Model Config

```
model:
  # Whether to load a model
  load_model: False
  # File path to the model
  model_path: "path/to/model.pt"
  # Save the model after every n-th update
  checkpoint_interval: 50
  # Set the to be used activation function (relu, leaky_relu, swish)
  activation: "relu"
  recurrence:
    # Supported recurrent layers: gru, lstm
    layer_type: "gru"
    # Length of the trained sequences, if set to 0 or smaller the sequence length is dynamically fit to episode lengths
    sequence_length: 32
    # Size of the recurrent layer's hidden state
    hidden_state_size: 128
    # How to initialize the hidden state
    hidden_state_init: "zero"
```

Every tool may use the model_path.
During training models are saved to the `checkpoints` directory.
The [model's architecture](model.md) can be influenced by adding a GRU layer right behind the observation encoder.

## Evaluation Config

```
evaluation:
  # Whether to evaluate the model during training
  evaluate: False
  # Number of environments that are used
  n_workers: 3
  # Evaluation seeds (each worker performs on every seed: in this case, overall 15 episodes are used for evaluation (n_workers * seeds))
  seeds: [1001, 1002, 1003, 1004, 1005]
  # Evaluate the model after every n-th update during training
  interval: 50
```

This configuration is shared among the training and the evaluation.
Evaluation can be configured to be run during training or as stand-alone process.

## Trainer Config

```
trainer:
  # Which algorithm to use. For now, PPO is supported.
  algorithm: "PPO"
  # Discount factor
  gamma: 0.99
  # Regularization parameter used when calculating the Generalized Advantage Estimation (GAE)
  lamda: 0.95
  # Number of PPO update cycles that shall be done (one whole cycle comprises n epochs of m mini_batch updates)
  updates: 1000
  # Number of environments that are used for sampling data
  n_workers: 16
  # Number of steps an agent samples data in each environment (batch_size = n_workers * worker_steps)
  worker_steps: 256
  # Number of times that the whole batch of data is used for optimization using PPO
  # Each epoch trains on a random permutation of the sampled training batch
  epochs: 4
  # Number of mini batches that are trained throughout one epoch
  # In case of using a recurrent net, this has to be a multiple of n_workers.
  n_mini_batch: 4
  # On which step to resume the training. This affects the hyperparameter schedules only.
  resume_at: 0
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