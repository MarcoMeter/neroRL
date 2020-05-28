# Tensorboard

During training, tensorboard summaries are saved to `summaries/run-id/timestamp_worker-id`.

Run `tensorboad --logdir=summaries` to watch the training and evaluation statistics in your browser using the URL [http://localhost:6006/](http://localhost:6006/).

The summaries are created at [trainer.py#L426](../neroRL/trainers/PPO/trainer.py#L426).
The minimum and maximum of final episode information is added to the summary and can be controlled in each environment's wrapper.
For example see Obstacle Tower: [obstacle_tower_wrapper.py#L171](../neroRL/environments/obstacle_tower_wrapper.py#L171)
