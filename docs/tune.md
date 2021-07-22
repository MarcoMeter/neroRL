# Hyperparameter Tuning using Grid Search

Hyperparameter tuning, based on grid search, is launched via the command `python tune.py`.
It generates training configs for each possible permutation of hyperparameter choices specified by a tuning config.
After that, training sessions are run sequentially.
You can also just generate and serialize the configs using the `--generate-only` flag.

```
"""
    Usage:
        tune.py [options]
        tune.py --help

    Options:
        --config=<path>             Path of the config file [default: ./configs/default.yaml].
        --tune=<path>               Path to the config file that features the hyperparameter search space for tuning [default: ./configs/tune/example.yaml]
        --num-repetitions=<n>       How many times to repeat the training of one config [default: 1]
        --worker-id=<n>             Sets the port for each environment instance [default: 2].
        --run-id=<path>             Specifies the tag of the tensorboard summaries [default: default].
        --low-mem-fix               Whether to load one mini_batch at a time. This is needed for GPUs with low memory (e.g. 2GB) [default: False].
        --generate-only             Whether to only generate the config files [default: False]
        --out=<path>                Where to output the generated config files [default: ./grid_search/]
    """
```

## --config
In general, [training](training.md), [evaluating](evaluation.md) and [enjoying](enjoy.md) a model relies on a [config file](configuration.md) that specifies the environment and further necessary parameters.
Therefore make use of the `--config=./configs/otc.yaml` argument to specify your configuration file.
In the case of tuning hyperparemeters based on grid search, the specified config is used as a basis.

## --tune
This flag is used to specify a config that features the to be tuned hyperparameters and its choices.
For grid search, all combinations of each parameter's choice is generated.
For each permutation, the base config file as specified by `--config` is overriden for the to be tuned hyperparameters.
An [example](#example-tuning-config) can be found at the bottom of this page.

## --num-repetitions
This determines the number of times a training session is being repeated.
Once all permuted training configs are trained one by one, another lap (up to num-repetitions) is run.

## --worker-id
Setting a `--worker-id=100` is necessary if you run multiple training sessions using Unity environments (e.g. Obstacle Tower), because these environments communicate with Python using sockets via distinct ports that are offset by the `--worker-id`.

## --run-id
The `--run-id=training_0` adds a tag to the training session, which is helpfull for identifying the [tensorboard](tensorboard.md) summaries and model checkpoints.

## --low-mem-fix
If you are training on a GPU with not enough VRAM you can try the `--low-mem-fix` option. This ensures that only one mini batch at a time is moved to the GPU's memory instead of the entire batch, which exceeds the GPU's memory.

## --generate-only
Instead of running training sessions sequentially, you can specify this flag to just generate config files for every possible permutation.
This way you can create your own shell script to run multiple training sessions in parallel or create jobs for a computational cluster.
Using the flag `--out` you can specify a path where all the configs are written to disk.

## --out
Specifies a destination path for the to be created configs or training results like [tensorboard](tensorboard.md) summaries or checkpoints.

## Example

```
python tune.py --config=./configs/minigrid.yaml --tune=./configs/tune/example.yaml --run-id=minigrid --out=./grid_search/
```

This command will conduct sequentially training sessions using the Minigrd environment. Each training draws its hyperparameters from the minigrid.yaml, but those values that are specified by the tuning config are overriden based on the generated permutations. All outputs, like [tensorboard](tensorboard.md) summaries and checkpoints, are saved to the grid_search directory.

## Example Tuning Config

```
# Example: Tune a few hyperparameters and decay schedules
hyperparameters:
  worker_steps: [64, 128]
  num_workers: [8, 16]

learning_rate_schedule:
  initial: [3.0e-3, 1.0e-4]
  final: [1.0e-5, 1.0e-6]

beta_schedule:
  initial: [0.01, 0.001, 0.001]
  final: [0.0001] # As we are using decaying schedules, these choices should not be greater than the initial values.
```

Every single parameter from a config can be used, but it has to be noted that not every parameter combination is desirable.
For example, choosing different environment names and type conflict.

In general, you specify most parameters and its choices under the `hyperparameters` section.
Deyacing schedules, like for the learning rate, have to be treated seperately, like seen above.