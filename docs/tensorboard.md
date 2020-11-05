# Tensorboard

During training, tensorboard summaries are saved to `summaries/run-id/timestamp_worker-id` if not modified by the `--out=<path>` flag.

Run `tensorboad --logdir=summaries --samples_per_plugin=text=100` to watch the training and evaluation statistics in your browser using the URL [http://localhost:6006/](http://localhost:6006/).
The `--samples_per_plugin=text=100` flag assures that all items of the training configuration (e.g. hyperparameters) are visible inside the Text tab of Tensorboard.

The summaries are created at [trainer.py#L440](../neroRL/trainers/PPO/trainer.py#L440).
The minimum and maximum of final episode information is added to the summary and can be controlled in each environment's wrapper.
For example see Obstacle Tower: [obstacle_tower_wrapper.py#L171](../neroRL/environments/obstacle_tower_wrapper.py#L171)
