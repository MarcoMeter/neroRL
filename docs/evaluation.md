# Evaluate a model

To evaluate a model utilize the `neval` command. `python neroRL/eval.py` is the alternative if the source code is used.

```
"""
    Usage:
        neval [options]
        neval --help

    Options:
        --config=<path>            Path to the config file [default: ].
        --checkpoint=<path>        Path to the checkpoint file [default: ].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --video=<path>             Specify a path for saving videos, if video recording is desired. The files' extension will be set automatically. [default: ./video].
        --start-seed=<n>           Specifies the start of the seed range [default: 200000].
        --num-seeds=<n>            Specifies the number of seeds to evaluate [default: 50].
        --repetitions=<n>          Specifies the number of repetitions for each seed [default: 3].
    """
```

## --config
In general, [training](training.md), evaluating and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/mortar_mayhem.yaml` argument to specify your configuration file.
If a configuration file is not provided, it will be sourced from the provided checkpoint.
When a checkpoint and a config are supplied, the checkpoint will determine the config of the model, while the provided config takes care of setting up the environment. This way, the trained model can be assessed on varied environment conditions.

## --checkpoint
The path to a checkpoint can be added by using the flag `--checkpoint=my_checkpoint.pt`. If no config is provided, the config is retrieved from the checkpoint.
Whenever a checkpoint is provided its model config is used to instantiate the model.

## --untrained
Instead of loading a model as stated by the config, a model can be created out of scratch using the `--untrained` option.
This is usually helpfull for debugging and testing.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --video
To render a video, you can use the `--video=target_path` flag.

## --start-seed

The agent can be trained or evaluated on a distinct range of seed. The start seed is set by this flag. The seeds are consumed by the environment to procedurally sample and generate a new level instance.

## --num-seeds

Starting from the `--start-seed`, the number of seeds are added to specify the range of seed that will be evaluated.

## --repetitions

This is the number of times that one seed is being repeated.

## Example

```
neval --config=./configs/procgen.yaml --checkpoint=./checkpoints/coinrun/model.pt --start-seed 1000 --num-seeds 100
```

This command
- loads the specified checkpoint and
- evaluates an agent playing CoinRun
- under the circumstances as stated in the config.
- 100 seeds are being evaluated ranging from 1000 to 1099.
- Playing these episodes are repeated 3 times.
