import torch
from torch import optim

from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import masked_mean
from neroRL.utils.decay_schedules import polynomial_decay

class DecoupledPPOTrainer(BaseTrainer):
    def __init__(self, configs, worker_id, run_id, out_path):
        super().__init__(configs, worker_id, run_id=run_id, out_path=out_path)

    def create_model(self):
        pass

    def train(self):
        pass

    def step_decay_schedules(self, update):
        pass

    def collect_checkpoint_data(self, update):
        return super().collect_checkpoint_data(update)

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
