# Enjoy (i.e. watch) a model

To watch an agent exploit its trained model, execute the `nenjoy` command or directly run `python neroRL/enjoy.py` given the source code.
Some already trained models can be found inside the `models` directory!

```
"""
    Usage:
        nenjoy [options]
        nenjoy --help

    Options:
        --config=<path>            Path to the config file [default: ].
        --checkpoint=<path>        Path to the checkpoint file [default: ].
        --untrained                Whether an untrained model should be used [default: False].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --seed=<n>                 The to be played seed of an episode [default: 0].
        --num-episodes=<n>         The number of to be played episodes [default: 1].
        --video=<path>             Specify a path for saving a video, if video recording is desired. The file's extension will be set automatically. [default: ./video].
        --framerate=<n>            Specifies the frame rate of the to be rendered video. [default: 6]
        --generate_website         Specifies wether a website shall be generated. [default: False]
    """
```

## --config
In general, [training](training.md), [evaluating](evaluation.md) and enjoying a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/mortar.yaml` argument to specify your configuration file.

## --checkpoint
The path to a checkpoint can be added by using the flag `--checkpoint=my_checkpoint.pt`. If no config is provided, the config is retrieved from the checkpoint.
Whenever a checkpoint is provided its model config is used to instantiate the model.

## --untrained
Instead of loading a model as stated by the config, a model can be created out of scratch using the `--untrained` option.
This is usually helpfull for debugging and testing.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --seed
`--seed=1001` specifies which environment seed to use for all episodes that are being played by the agent.

## --num-episodes
This flag specifices the number of to be played episodes using the same seed.

## --video
To render a video, you can use the `--video=target_path` flag. Note that only one episode will be played.

## --framerate
If a video shall be rendered, this flag can be used to modify the video's frame rate.

## --generate_website
As an alternative to rendering a video, a website can be generated that visualizes further information such as the value and the entropy. Note that only one episode will be played. The website's template can be found at `result/template/result_website.html`.

## Example

```
nenjoy --config=./configs/minigrid.yaml --seed=1001 --video=my_video --framerate=10
```

This command
- loads the model as stated in the config,
- sets the seed 1001 for the specified environment,
- renders and runs in realtime the agent playing the environment,
- and outputs my_video.mp4 to the current path using a frame rate of 10.
