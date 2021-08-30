import numpy as np
import torch
from torch import optim

from neroRL.nn.actor_critic import create_actor_critic_model
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
        self.value_update_interval = ["trainer"]["value_update_interval"]
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

        # Determine policy and value function parameters
        self.policy_parameters = []
        self.value_parameters = []
        for name, param in self.model.named_parameters():
            if "actor" in name:
                self.policy_parameters.append(param)
            elif "critic" in name:
                self.value_parameters.append(param)

        # Instantiate optimizer
        self.policy_optimizer = optim.AdamW(self.policy_parameters, lr=self.policy_learning_rate)
        self.value_optimizer = optim.AdamW(self.value_parameters, lr=self.value_learning_rate)


    def create_model(self):
        return create_actor_critic_model(self.configs["model"], False,
        self.visual_observation_space, self.vector_observation_space, self.action_space_shape, self.recurrence, self.device)

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
        if self.currentUpdate % self.value_update_interval == 0:
            for _ in range(self.num_value_epochs):
                if self.recurrence is not None:
                    batch_generator = self.buffer.recurrent_mini_batch_generator(1)
                else:
                    batch_generator = self.buffer.mini_batch_generator(1)
                for batch in batch_generator:
                    res = self.train_value_function(batch)
                    for key, (tag, value) in res.items():
                        train_info.setdefault(key, (tag, []))[1].append(value)

        # Calculate mean of the collected values
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        return train_info

    def train_policy_mini_batch(self, samples):
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        
        policy, _, _ = self.model(samples["vis_obs"] if self.visual_observation_space is not None else None,
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

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = masked_mean(torch.stack(entropies, dim=1).sum(1).reshape(-1), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss + self.beta * entropy_bonus)

        # Compute gradients
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_parameters, max_norm=0.5)
        self.policy_optimizer.step()

        # Monitor additional training statistics
        approx_kl = masked_mean((torch.exp(ratio) - 1) - ratio, samples["loss_mask"])
        clip_fraction = (abs((ratio - 1.0)) > self.clip_range).type(torch.FloatTensor).mean()

        return {"policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
                "loss": (Tag.LOSS, loss.cpu().data.numpy()),
                "entropy": (Tag.OTHER, entropy_bonus.cpu().data.numpy()),
                "kl_divergence": (Tag.OTHER, approx_kl.cpu().data.numpy()),
                "clip_fraction": (Tag.OTHER, clip_fraction.cpu().data.numpy())}

    def train_value_function(self, samples):
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        
        _, value, _ = self.model(samples["vis_obs"] if self.visual_observation_space is not None else None,
                                    samples["vec_obs"] if self.vector_observation_space is not None else None,
                                    recurrent_cell,
                                    self.device,
                                    self.buffer.actual_sequence_length)

        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = masked_mean(vf_loss, samples["loss_mask"])

        # Compute gradients
        self.value_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.value_optimizer.step()

        return {"value_loss": (Tag.LOSS, vf_loss.cpu().data.numpy())}

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
