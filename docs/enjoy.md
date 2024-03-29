# Enjoy (i.e. watch) a model

To watch an agent exploit its trained model, execute the `python enjoy.py` command.
Some already trained models can be found inside the `models` directory!

```
"""
    Usage:
        enjoy.py [options]
        enjoy.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of a video shall be rendered. [default: 6]
"""
```

## --config
In general, [training](training.md), [evaluating](evaluation.md) and enjoying a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/otc.yaml` argument to specify your configuration file.

## --untrained
Instead of loading a model as stated by the config, a model can be created out of scratch using the `--untrained` option.
This is usually helpfull for debugging/testing.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --seed
`--seed=1001` specifies which environment seed to use for the only episode, which is being played by the agent.

## --video
To render a video, you can use the `--video=target_path` flag.

## --framerate
If a video shall be rendered, this flag can be used to modify the video's frame rate.

## Example

```
python enjoy.py --config=./configs/minigrid.yaml --seed=1001 --video=my_video --framerate=10
```

This command
- loads the model as stated in the config,
- sets the seed 1001 for the specified environment,
- renders and runs in realtime the agent playing the environment,
- and outputs my_video.mp4 to the current path using a frame rate of 10.
