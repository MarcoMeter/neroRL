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
        --checkpoint=<path>         Path to a checkpoint to resume training from [default: None].
        --num-updates=<n>           Number of additional updates to train for if training shall resume from a checkpoint [default: 0].
    """
```

## --config
In general, training, [evaluating](evaluation.md) and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/mortar_mayhem.yaml` argument to specify your configuration file.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments (e.g. Obstacle Tower), because these environments communicate with Python using sockets via distinct ports that are offset by `--worker-id`.

## --run-id
`--run-id=training_0` adds a tag to the training session, which is helpfull for identifying [tensorboard](tensorboard.md) summaries and model checkpoints. If no run-id is provided, the name of the config file is used as run-id instead.

## --out
It specifies the path for outputting files such as the tensorboard summary and the model's checkpoints.
If not specified, these will be saved in the folder `summaries` and `checkpoints` using the current path.

## --seed
Sets the seed for the entire training procedure for reproducibility. This affects all random number generators.

## --compile
Whether to compile the model before training. As of PyTorch 2.0.0, this feature may be unstable and may not work on Windows out-of-the-box.

## --low-mem
Whether to reduce the usage of GPU memory. Given this mode all training data is sampled and stored on the CPU.
Upon optimization, mini batches are moved one by one to the GPU for optimization.
This I/O traffic notably hurts the wall-time.

## --checkpoint
This allows you to resume training from a given checkpoint. The needed config is provided by the checkpoint.

## --num-updates
When resuming training from a checkpoint, you can increase the original number of updates.

## Example A

```
ntrain --config=./configs/mortar_mayhem.yaml --run-id=test --worker-id=100
```

This command
- trains an agent on the Mortar Mayhem environment (Unity version),
- stores checkpoints and summaries to a folder that is named after the run-id,
- and sets the offset for the communication port of the environments to 100:(100 + n_workers).

## Exmaple B

```
ntrain --checkpoint ./checkpoints/cartpole/20230805-125535_2/cartpole-209.pt --num-updates=100
```

This command
- resumes training of an agent that was previously trained on the CartPole environment for 209 updates,
- resumes training with update 210 next,
- and progresses training untill reaching update 310.
