import torch
import numpy as np
from torch import optim

from neroRL.nn.actor_critic import create_actor_critic_model
from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import masked_mean, compute_gradient_stats
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.monitor import Tag

class PPOTrainer(BaseTrainer):
    """PPO implementation according to Schulman et al. 2017. It supports multi-discrete action spaces as well as visual 
    and vector obsverations (either alone or simultaenously). Parameters can be shared or not. If gradients shall be decoupled,
    go for the DecoupledPPOTrainer.
    """
    def __init__(self, configs, worker_id, run_id, out_path, seed = 0):
        """
        Initializes distinct members of the PPOTrainer

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
        """
        super().__init__(configs, worker_id, run_id=run_id, out_path=out_path, seed=seed)

        # Hyperparameter setup
        self.epochs = configs["trainer"]["epochs"]
        self.vf_loss_coef = self.configs["trainer"]["value_coefficient"]
        self.n_mini_batches = configs["trainer"]["n_mini_batches"]
        batch_size = self.n_workers * self.worker_steps
        assert (batch_size % self.n_mini_batches == 0), "Batch Size divided by number of mini batches has a remainder."
        self.max_grad_norm = configs["trainer"]["max_grad_norm"]

        self.lr_schedule = configs["trainer"]["learning_rate_schedule"]
        self.beta_schedule = configs["trainer"]["beta_schedule"]
        self.cr_schedule = configs["trainer"]["clip_range_schedule"]

        self.learning_rate = self.lr_schedule["initial"]
        self.beta = self.beta_schedule["initial"]
        self.clip_range = self.cr_schedule["initial"]

        # Instantiate optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def create_model(self) -> None:
        return create_actor_critic_model(self.configs["model"], self.configs["trainer"]["share_parameters"],
        self.visual_observation_space, self.vector_observation_space, self.action_space_shape, self.recurrence, self.device)

    def train(self):
        train_info = {}

        # Train policy and value function for e epochs using mini batches
        for epoch in range(self.epochs):
            # Refreshes buffer with current model every refresh_buffer_epoch
            if epoch > 0 and epoch % self.refresh_buffer_epoch == 0 and self.refresh_buffer_epoch > 0:
                self.sampler.buffer.refresh(self.model, self.gamma, self.lamda)
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
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        # Format specific values for logging inside the base class
        formatted_string = "loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}".format(
            train_info["loss"][1], train_info["policy_loss"][1], train_info["value_loss"][1], train_info["entropy"][1])

        # Return the mean of the training statistics
        return train_info, formatted_string

    def train_mini_batch(self, samples):
        """Optimizes the policy based on the PPO algorithm

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
        
        policy, value, _, _ = self.model(samples["vis_obs"] if self.visual_observation_space is not None else None,
                                    samples["vec_obs"] if self.vector_observation_space is not None else None,
                                    recurrent_cell,
                                    self.sampler.buffer.actual_sequence_length)
        
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
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = masked_mean(policy_loss, samples["loss_mask"])

        # Value
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = masked_mean(vf_loss, samples["loss_mask"])

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = masked_mean(torch.stack(entropies, dim=1).sum(1).reshape(-1), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss - self.vf_loss_coef * vf_loss + self.beta * entropy_bonus)

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Monitor additional training statistics
        approx_kl = masked_mean((ratio - 1.0) - log_ratio, samples["loss_mask"]) # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > self.clip_range).float().mean()

        if self.model.share_parameters:
            modules = self.model.actor_critic_modules
        else:
            modules = {**self.model.actor_modules, **self.model.critic_modules}

        return {**compute_gradient_stats(modules),
                "policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
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

        return {
            "learning_rate": (Tag.DECAY, self.learning_rate),
            "beta": (Tag.DECAY, self.beta),
            "clip_range": (Tag.DECAY, self.clip_range)
        }


    def collect_checkpoint_data(self, update):
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        checkpoint_data["optimizer"] = self.optimizer.state_dict()
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])