# Evaluate a model

To evaluate a series of model checkpoints utilize the `neval-checkpoints` command. `python neroRL/eval_checkpoints.py` is the alternative if the source code is used.

```
"""
    Usage:
        neval-checkpoints [options]
        neval-checkpoints --help

    Options:
        --worker-id=<n>            Sets the port for each environment instance [default: 2].
        --checkpoints=<path>       Path to the directory containing checkpoints [default: ].
        --config=<path>            Path to the config file [default: ].
        --name=<path>              Specifies the full path to save the output file [default: ./results.res].
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

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments, because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --checkpoints
This is the path to the directory that contains the checkpoints. The configuration of the model is sourced from the checkpoints, while the provided config file details the environment's properties.

## --name
This is the file path to save the results of the evaluation.

## --start-seed

The agent can be trained or evaluated on a distinct range of seed. The start seed is set by this flag. The seeds are consumed by the environment to procedurally sample and generate a new level instance.

## --num-seeds

Starting from the `--start-seed`, the number of seeds are added to specify the range of seed that will be evaluated.

## --repetitions

This is the number of times that one seed is being repeated.

## Example
```
neval-checkpoints --config=./configs/minigrid.yaml --checkpoints=./checkpoints/default/20200527-111513_2 --name=results.res
```

This command
- loads all checkpoints located in the stated directory,
- evaluates these on the Minigrid environment,
- and outputs the file results.res with the evaluation's results.

## Plotting the results
In order to work with the results check out this [jupyter notebook](../notebooks/plot_checkpoint_results.ipynb).
