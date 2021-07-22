# Train a model

The training is launched via the command `python train.py`.

```
"""
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path of the Config file [default: ./configs/default.yaml].
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --run-id=<path>            Specifies the tag of the tensorboard summaries [default: default].
        --low-mem-fix              Whether to load one mini_batch at a time. This is needed for GPUs with low memory (e.g. 2GB) [default: False].
        --out=<path>               Specifies the path to output files such as summaries and checkpoints. [default: ./]
"""
```

## --config
In general, training, [evaluating](evaluation.md) and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/otc.yaml` argument to specify your configuration file.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments (e.g. Obstacle Tower), because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --run-id
The `--run-id=training_0` adds a tag to the training session, which is helpfull for identifying the [tensorboard](tensorboard.md) summaries and model checkpoints.

## --low-mem-fix
If you are training on a GPU with not enough VRAM you can try the `--low-mem-fix` option. This ensures that only one mini batch at a time is moved to the GPU's memory instead of the entire batch, which exceeds the GPU's memory.

## --out
It specifies the path for outputting files such as the tensorboard summary and the model's checkpoints.
If not specified, these will be saved in the folder `summaries` and `checkpoints`.

## Example

```
python train.py --config=./condigs/otc.yaml --run-id=test --worker-id=100
```

This command
- trains an agent on the Obstacle Tower environment,
- stores checkpoints and summaries to a folder that is named after the run-id
- and sets the offset for the communication port of the environments to 100:100 + n_workers.
