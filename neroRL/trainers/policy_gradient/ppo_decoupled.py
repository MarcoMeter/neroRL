import numpy as np
import torch
from torch import optim

from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import masked_mean
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.monitor import Tag

class DecoupledPPOTrainer(BaseTrainer):
    def __init__(self, configs, worker_id, run_id, out_path):
        super().__init__(configs, worker_id, run_id=run_id, out_path=out_path)

        # Hyperparameter setup
        self.num_policy_epochs = configs["trainer"]["policy_epochs"]
        self.num_value_epochs = configs["trainer"]["value_epochs"]
        self.n_mini_batch = configs["trainer"]["n_mini_batch"]
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0), "Batch Size divided by number of mini batches has a remainder."

        self.policy_lr_schedule = configs["trainer"]["policy_learning_rate_schedule"]
        self.value_lr_schedule = configs["trainer"]["value_learning_rate_schedule"]
        self.beta_schedule = configs["trainer"]["beta_schedule"]
        self.cr_schedule = configs["trainer"]["clip_range_schedule"]

        self.policy_learning_rate = self.policy_lr_schedule["initial"]
        self.value_learning_rate = self.value_lr_schedule["initial"]
        self.beta = self.beta_schedule["initial"]
        self.clip_range = self.cr_schedule["initial"]

        # Instantiate optimizer
        self.policy_optimizer = optim.AdamW(self.model.parameters(), lr=self.policy_learning_rate)
        self.value_optimizer = optim.AdamW(self.model.parameters(), lr=self.value_learning_rate)


    def create_model(self):
        pass

    def train(self):
        train_info = {}

        # Train policy
        for _ in range(self.num_policy_epochs):
            if self.recurrence is not None:
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator(self.n_mini_batch)
            else:
                mini_batch_generator = self.buffer.mini_batch_generator(self.n_mini_batch)
            for mini_batch in mini_batch_generator:
                res = self.train_policy_mini_batch(mini_batch)
                # Collect all values of the training procedure in a list
                for key, (tag, value) in res.items():
                    train_info.setdefault(key, (tag, []))[1].append(value)

        # Train value function
        for _ in range(self.num_value_epochs):
            self.train_value_function()

        # Calculate mean of the collected values
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        return train_info

    def train_policy_mini_batch(self, samples):
        pass

    def train_value_function(self):
        pass

    def step_decay_schedules(self, update):
        self.policy_learning_rate = polynomial_decay(self.policy_lr_schedule["initial"], self.policy_lr_schedule["final"],
                                        self.policy_lr_schedule["max_decay_steps"], self.policy_lr_schedule["power"], update)
        self.value_learning_rate = polynomial_decay(self.value_lr_schedule["initial"], self.value_lr_schedule["final"],
                                        self.value_lr_schedule["max_decay_steps"], self.value_lr_schedule["power"], update)
        self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
        self.clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"],
                                        self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

        # Apply learning rates to optimizers
        for pg in self.policy_optimizer.param_groups:
            pg["lr"] = self.policy_learning_rate
        for pg in self.value_optimizer.param_groups:
            pg["lr"] = self.value_learning_rate

        # TODO monitor both learning rates instead of one
        return self.policy_learning_rate, self.beta, self.clip_range
        # return self.policy_learning_rate, self.value_learning_rate, self.beta, self.clip_range

    def collect_checkpoint_data(self, update):
        return super().collect_checkpoint_data(update)

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
