import torch
import numpy as np

from torch import nn, optim

from neroRL.nn.actor_critic import create_actor_critic_model
from neroRL.utils.monitor import Tag
from neroRL.utils.utils import compute_gradient_stats
from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.decay_schedules import polynomial_decay

class GroundTruthTrainer(BaseTrainer):
    """Trainer for the ground truth estimator only. This is used to analyze whether the model learned
    useful skills. The estimator is trained to approximate ground truth information from the latent space.
    """
    def __init__(self, configs, sample_device, train_device, worker_id, run_id, out_path, seed=0, compile_model=False):
        super().__init__(configs, sample_device, train_device, worker_id, run_id, out_path, seed, compile_model)
        # Hyperparameter setup
        self.epochs = configs["trainer"]["epochs"]
        self.n_mini_batches = configs["trainer"]["n_mini_batches"]
        self.lr_schedule = configs["trainer"]["learning_rate_schedule"]
        self.learning_rate = self.lr_schedule["initial"]
        batch_size = self.n_workers * self.worker_steps
        assert (batch_size % self.n_mini_batches == 0), "Batch Size divided by number of mini batches has a remainder."
        self.max_grad_norm = configs["trainer"]["max_grad_norm"]

        # Init optimizer to only tune the ground truth estimator's parameters
        self.optimizer = optim.Adam(self.model.ground_truth_estimator.parameters(), lr=self.learning_rate)

        # Ground truth estimation members
        self.gt_estimator_schedule = self.configs["trainer"]["ground_truth_estimator_schedule"]
        self.mse_loss = nn.MSELoss()

    def create_model(self) -> None:
        return create_actor_critic_model(self.configs["model"], self.vis_obs_space, self.vec_obs_space,
                                         self.ground_truth_space, self.action_space_shape, self.sample_device)

    def train(self):
        train_info = {}

        # Train policy and value function for e epochs using mini batches
        for epoch in range(self.epochs):
            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.sampler.buffer.recurrent_mini_batch_generator(self.n_mini_batches)
            else:
                mini_batch_generator = self.sampler.buffer.mini_batch_generator(self.n_mini_batches)
            # Conduct the training
            for mini_batch in mini_batch_generator:
                res = self.train_mini_batch(mini_batch)
                # Collect all values of the training procedure in a list
                for key, (tag, value) in res.items():
                    train_info.setdefault(key, (tag, []))[1].append(value)

        # Calculate mean of the collected training statistics
        stats = {}
        for key, (tag, values) in train_info.items():
            stats[key] = (tag, np.mean(values))
            if key == "gt_loss" or key == "gt_error":
                stats[key + "_min"] = (tag, np.min(values))
                stats[key + "_max"] = (tag, np.max(values))

        # Format specific values for logging inside the base class
        formatted_string = "gt_loss={:.3f} gt_loss_max={:.3f} gt_loss_min={:.3f} error={:.3f} error_min={:.3f} error_max={:.3f}".format(
            stats["gt_loss"][1], stats["gt_loss_max"][1], stats["gt_loss_min"][1], stats["gt_error"][1], stats["gt_error_min"][1], stats["gt_error_max"][1])

        # Return the mean of the training statistics
        return stats, formatted_string

    def train_mini_batch(self, samples):
        # Retrieve the agent's memory to feed the model
        memory, mask, memory_indices = None, None, None
        # Case Recurrence: the recurrent cell state is treated as the memory. Only the initial hidden states are selected.
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                memory = samples["hxs"]
            elif self.recurrence["layer_type"] == "lstm":
                memory = (samples["hxs"], samples["cxs"])
        # Case Transformer: the episodic memory is based on activations that were previously gathered throughout an episode
        if self.transformer is not None:
            # Slice memory windows from the whole sequence of episode memories
            memory = samples["memory_window"]
            mask = samples["memory_mask"]
            memory_indices = samples["memory_indices"]

        # Forward model -> policy, value, memory, gae
        with torch.no_grad():
            policy, value, _ = self.model(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                        samples["vec_obs"] if self.vec_obs_space is not None else None,
                                        memory = memory, mask = mask, memory_indices = memory_indices,
                                        sequence_length = self.sampler.buffer.actual_sequence_length)

        # Ground truth estimation loss
        # Forward ground truth estimator
        estimation = self.model.estimate_ground_truth()
        target = samples["ground_truth"]
        # Remove paddings if recurrence is used
        if self.recurrence is not None:
            estimation = estimation[samples["loss_mask"]]
            target = target[samples["loss_mask"]]
        # Compute ground truth estimation loss
        estimation_loss = self.mse_loss(estimation, target)

        # Calculate the error for monitoring
        error = torch.abs(estimation - target).mean()

        # retrieve entire pos encoding table
        # from neroRL.nn.transformer import SinusoidalPosition
        # result = torch.zeros((estimation.shape[0], 1024, 384))
        # self.pos_embedding = SinusoidalPosition(dim = 384)
        # pos_encoding = self.pos_embedding(1024)
        # # subtract estimate from every pos encoding
        # for i in range(estimation.shape[0]):
        #     result[i] = pos_encoding - estimation[i]
        # result = torch.abs(result)
        # t_estimate = torch.argmin(result.mean(dim = 2), dim = 1)
        # # set dtype to float32
        # t_estimate = t_estimate.type(torch.FloatTensor)

        # # calc t_target
        # for i in range(target.shape[0]):
        #     result[i] = pos_encoding - target[i]
        # result = torch.abs(result)
        # t_target = torch.argmin(result.mean(dim = 2), dim = 1)
        # # set dtype to float32
        # t_target = t_target.type(torch.FloatTensor)

        # # t_error = torch.abs(t_estimate - t_target).mean()
        # print(torch.abs(t_estimate - t_target).min())
        # print(torch.abs(t_estimate - t_target).mean())
        # print(torch.abs(t_estimate - t_target).max())


        # Compute gradients
        self.optimizer.zero_grad()
        estimation_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        out = {**compute_gradient_stats({"gt_estimator": self.model.ground_truth_estimator}),
                "gt_loss": (Tag.LOSS, estimation_loss.cpu().data.numpy()),
                "gt_error": (Tag.ERROR, error.cpu().data.numpy())}

        return out

    def step_decay_schedules(self, update):
        self.learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"],
                                        self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
        
        # Apply learning rate to optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

        # Report decayed learning rate
        return {"learning_rate": (Tag.DECAY, self.learning_rate)}

    def collect_checkpoint_data(self, update):
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        # the decoder's optimizer's parameters are not saved
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        # the decoder's optimizer's parameters are not loaded