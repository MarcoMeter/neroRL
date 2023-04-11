# Train a model

The training is launched via the command `ntrain` . Given the source code, running `python neroRL/train.py` is an alternative.

```
"""
    Usage:
        ntrain [options]
        ntrain --help

    Options:
        --config=<path>             Path to the config file [default: ./configs/default.yaml].
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
        --out=<path>                Specifies the path to output files such as summaries and checkpoints. [default: ./]
        --seed=<n>      	        Specifies the seed to use during training. If set to smaller than 0, use a random seed. [default: -1]
        --compile                   Whether to compile the model or not (requires PyTorch >= 2.0.0). [default: False]
        --low-mem                   Whether to move one mini_batch at a time to GPU to save memory [default: False].
    """
```

## --config
In general, training, [evaluating](evaluation.md) and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/mortar.yaml` argument to specify your configuration file.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments (e.g. Obstacle Tower), because these environments communicate with Python using sockets via distinct ports that are offset by `--worker-id`.

## --run-id
`--run-id=training_0` adds a tag to the training session, which is helpfull for identifying [tensorboard](tensorboard.md) summaries and model checkpoints.

## --out
It specifies the path for outputting files such as the tensorboard summary and the model's checkpoints.
If not specified, these will be saved in the folder `summaries` and `checkpoints` using the current path.

## --seed
Sets the seed for the entire training procedur for reproducibility.

## --compile
Whether to compile the model before training. As of PyTorch 2.0.0, this feature is extremly unstable.
It is not available on Windows yet.

## --low-mem
Whether to reduce the usage of GPU memory. Given this mode all training data is sampled and stored on the CPU.
Upon optimization, mini batches are moved one by one to the GPU for optimization.
This I/O traffic notably hurts the wall-time.

## Example

```
ntrain --config=./configs/mortar_mayhem.yaml --run-id=test --worker-id=100
```

This command
- trains an agent on the Mortar Mayhem environment (Unity version),
- stores checkpoints and summaries to a folder that is named after the run-id,
- and sets the offset for the communication port of the environments to 100:(100 + n_workers).
