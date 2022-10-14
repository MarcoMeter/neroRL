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
"""
```

## --config
In general, [training](training.md), evaluating and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/mortar.yaml` argument to specify your configuration file.

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

## --framerate
If a video shall be rendered, this flag can be used to modify the video's frame rate.

## Example

```
neval --config=./configs/procgen.yaml --checkpoint=./checkpoints/coinrun/model.pt
```

This command
- loads the specified checkpoint and
- evaluates an agent playing CoinRun
- under the circumstances as stated in the config.
