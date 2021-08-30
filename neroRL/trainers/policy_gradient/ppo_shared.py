import torch
import numpy as np
from torch import optim

from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import masked_mean
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.monitor import Tag

class PPOTrainer(BaseTrainer):
    def __init__(self, configs, worker_id, run_id, low_mem_fix, out_path):
        super().__init__(configs, worker_id, run_id=run_id, low_mem_fix=low_mem_fix, out_path=out_path)

        # Hyperparameter setup
        self.lr_schedule = configs["trainer"]["learning_rate_schedule"]
        self.beta_schedule = configs["trainer"]["beta_schedule"]
        self.cr_schedule = configs["trainer"]["clip_range_schedule"]

        self.learning_rate = self.lr_schedule["initial"]
        self.beta = self.beta_schedule["initial"]
        self.clip_range = self.cr_schedule["initial"]

        # Instantiate optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
            
        Returns:
            {numpy.ndarray} -- Mean training statistics of one training epoch"""
        train_info = {}

        for _ in range(self.epochs):
            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            else:
                mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                res = self.train_mini_batch(mini_batch)
                # Collect all values of the training procedure in a list
                for key, (tag, value) in res.items():
                    train_info.setdefault(key, (tag, []))[1].append(value)

        # Calculate mean of the collected values
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        # Return the mean of the training statistics
        return train_info

    def train_mini_batch(self, samples):
        """ Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
        
        Returns:
            training_stats {dict} -- Losses, entropy, kl-divergence and clip fraction
        """
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        
        policy, value, _ = self.model(samples["vis_obs"] if self.visual_observation_space is not None else None,
                                    samples["vec_obs"] if self.vector_observation_space is not None else None,
                                    recurrent_cell,
                                    self.device,
                                    self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs = []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
        log_probs = torch.stack(log_probs, dim=1)

        # Compute surrogates
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        # Repeat is necessary for multi-discrete action spaces
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        ratio = torch.exp(log_probs - samples["log_probs"])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = masked_mean(policy_loss, samples["loss_mask"])

        # Value
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = masked_mean(vf_loss, samples["loss_mask"])
        vf_loss = .25 * vf_loss

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = masked_mean(torch.stack(entropies, dim=1).sum(1).reshape(-1), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss - vf_loss + self.beta * entropy_bonus)

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Monitor training statistics
        approx_kl = masked_mean((torch.exp(ratio) - 1) - ratio, samples["loss_mask"])
        clip_fraction = (abs((ratio - 1.0)) > self.clip_range).type(torch.FloatTensor).mean()

        return {"policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
                "value_loss": (Tag.LOSS, vf_loss.cpu().data.numpy()),
                "loss": (Tag.LOSS, loss.cpu().data.numpy()),
                "entropy": (Tag.OTHER, entropy_bonus.cpu().data.numpy()),
                "kl_divergence": (Tag.OTHER, approx_kl.cpu().data.numpy()),
                "clip_fraction": (Tag.OTHER, clip_fraction.cpu().data.numpy())}

    def step_decay_schedules(self, update):
        self.learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"],
                                        self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
        self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
        self.clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"],
                                        self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)

        # Apply learning rate to optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

        return self.learning_rate, self.beta, self.clip_range

    def collect_checkpoint_data(self, update):
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        checkpoint_data["optimizer"] = self.optimizer.state_dict()
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])