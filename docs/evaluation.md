# Evaluate a model

To evaluate a model utilize the `python eval.py` command.

```
"""
    Usage:
        evaluate.py [options]
        evaluate.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of a video shall be rendered. [default: 6]
"""
```

## --config

In general, [training](training.md), evaluating and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/otc.yaml` argument to specify your configuration file.

## --untrained
Instead of loading a model as stated by the config, a model can be created out of scratch using the `--untrained` option.
This is usually helpfull for debugging/testing.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --video
To render a video, you can use the `--video=target_path` flag.

## --framerate
If a video shall be rendered, this flag can be used to modify the video's frame rate.

## Example
```
python eval.py --config=./configs/procgen.yaml
```
This command
- evaluates an agent playing CoinRun
- under the circumstances as stated in the config.
