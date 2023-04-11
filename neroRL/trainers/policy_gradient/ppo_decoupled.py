import numpy as np
import torch
from torch import optim
from threading import Thread

from neroRL.nn.actor_critic import create_actor_critic_model
from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import compute_gradient_stats
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.monitor import Tag

class DecoupledPPOTrainer(BaseTrainer):
    """The DecoupledPPOTrainer does not share parameters (i.e. weights) and not gradients among the policy and value function.
    Therefore, it uses slightly different hyperparameters as the regular PPOTrainer to allow more control over updating the
    policy and the value function. Optinally, the actor model can estimate the advantage function as proposed by Raileanu & Fergus, 2021"""
    def __init__(self, configs, sample_device, train_device, worker_id, run_id, out_path, seed = 0, compile_model = False):
        """
        Initializes distinct members of the DecoupledPPOTrainer

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            sample_device {torch.device} -- The device used for sampling training data.
            train_device {torch.device} -- The device used for model optimization.
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            compile_model {bool} -- If True, the model is compiled before training (default: {False})
        """
        # Shall the policy estimate the advantage function? (DAAC algorithm by Raileanu & Fergus, 2021)
        # Assign this before initializing the base class, because this information is needed during model creation
        self.use_daac = "DAAC" in configs["trainer"]

        # Init base class
        super().__init__(configs, sample_device, train_device, worker_id, run_id=run_id, out_path=out_path, seed=seed, compile_model=compile_model)

        # Hyperparameter setup
        self.num_policy_epochs = configs["trainer"]["policy_epochs"]
        self.num_value_epochs = configs["trainer"]["value_epochs"]
        self.value_update_interval = configs["trainer"]["value_update_interval"]
        self.n_policy_mini_batches = configs["trainer"]["n_policy_mini_batches"]
        self.n_value_mini_batches = configs["trainer"]["n_value_mini_batches"]
        batch_size = self.n_workers * self.worker_steps
        assert (batch_size % self.n_policy_mini_batches == 0), "Batch Size divided by number of mini batches has a remainder."
        assert (batch_size % self.n_value_mini_batches == 0), "Batch Size divided by number of mini batches has a remainder."
        self.pi_max_grad_norm = configs["trainer"]["max_policy_grad_norm"]
        self.v_max_grad_norm = configs["trainer"]["max_value_grad_norm"]
        self.run_threaded = configs["trainer"]["run_threaded"]
        if self.use_daac:
            self.adv_coefficient = configs["trainer"]["DAAC"]["adv_coefficient"]
        # Decaying hyperparameter schedules
        self.policy_lr_schedule = configs["trainer"]["policy_learning_rate_schedule"]
        self.value_lr_schedule = configs["trainer"]["value_learning_rate_schedule"]
        self.beta_schedule = configs["trainer"]["beta_schedule"]
        self.pi_cr_schedule = configs["trainer"]["policy_clip_range_schedule"]
        self.v_cr_schedule = configs["trainer"]["value_clip_range_schedule"]
        # Decaying hyperparameter members
        self.policy_learning_rate = self.policy_lr_schedule["initial"]
        self.value_learning_rate = self.value_lr_schedule["initial"]
        self.beta = self.beta_schedule["initial"]
        self.policy_clip_range = self.pi_cr_schedule["initial"]
        self.value_clip_range = self.v_cr_schedule["initial"]

        # Determine policy and value function parameters to assign them to their respective optimizers
        self.policy_parameters = self.model.get_actor_params()
        self.value_parameters = self.model.get_critic_params()

        # Instantiate optimizer
        self.policy_optimizer = optim.AdamW(self.policy_parameters, lr=self.policy_learning_rate)
        self.value_optimizer = optim.AdamW(self.value_parameters, lr=self.value_learning_rate)

    def create_model(self):
        model =  create_actor_critic_model(self.configs["model"], False,
        self.vis_obs_space, self.vec_obs_space, self.action_space_shape,self.sample_device)
        # Optionally, add the advantage estimator head to the model
        if self.use_daac:
            model.add_gae_estimator_head(self.action_space_shape, self.sample_device)
        return model

    def train(self):
        self.train_info = {}

        # Add dummy key to the samples dict to avoid threading issues if the advantage should be normalized batch-wise
        if self.configs["trainer"]["advantage_normalization"] == "batch":
            self.sampler.buffer.samples_flat["normalized_advantages"] = None

        if self.run_threaded:
            # Launch threads for training the actor and critic model simultaenously
            threads = [Thread(target = self.train_policy, daemon = True), Thread(target = self.train_value, daemon = True)]
            for thread in threads:
                thread.start()
            # Wait for the threads to be done
            for thread in threads:
                thread.join()
        else:
            self.train_policy()
            self.train_value()
        
        # Calculate mean of the collected training statistics
        for key, (tag, values) in self.train_info.items():
            self.train_info[key] = (tag, np.mean(values))

        # Format specific values for logging that is done inside the base class
        if self.use_daac:
            formatted_string = "loss={:.3f} a_losss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}".format(
                self.train_info["loss"][1], self.train_info["advantage_loss"][1], self.train_info["policy_loss"][1], self.train_info["value_loss"][1], self.train_info["entropy"][1])
        else:
            formatted_string = "loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}".format(
                self.train_info["loss"][1], self.train_info["policy_loss"][1], self.train_info["value_loss"][1], self.train_info["entropy"][1])

        return self.train_info, formatted_string

    def train_policy(self):
        # Train the actor model using mini batches
        for epoch in range(self.num_policy_epochs):
            # Refreshes buffer with current model every refresh_buffer_epoch
            if epoch > 0 and epoch % self.refresh_buffer_epoch == 0 and self.refresh_buffer_epoch > 0:
                self.sampler.buffer.refresh(self.model, self.gamma, self.lamda)

            # Normalize advantages batch-wise if desired
            # This is done during every epoch just in case refreshing the buffer is used
            if self.configs["trainer"]["advantage_normalization"] == "batch":
                advantages = self.sampler.buffer.samples_flat["advantages"]
                self.sampler.buffer.samples_flat["normalized_advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.sampler.buffer.recurrent_mini_batch_generator(self.n_policy_mini_batches)
            else:
                mini_batch_generator = self.sampler.buffer.mini_batch_generator(self.n_policy_mini_batches)
            # Conduct the training
            for mini_batch in mini_batch_generator:
                res = self.train_policy_mini_batch(mini_batch)
                # Collect all values of the training procedure in a list
                for key, (tag, value) in res.items():
                    self.train_info.setdefault(key, (tag, []))[1].append(value)

    def train_value(self):
        # Train the value function using the whole batch of data instead of mini batches
        if self.current_update % self.value_update_interval == 0:
            for epoch in range(self.num_value_epochs):
                # Refreshes buffer with current model for every refresh_buffer_epoch
                if epoch > 0 and epoch % self.refresh_buffer_epoch == 0 and self.refresh_buffer_epoch > 0:
                    self.sampler.buffer.refresh(self.model, self.gamma, self.lamda)
                if self.recurrence is not None:
                    batch_generator = self.sampler.buffer.recurrent_mini_batch_generator(self.n_value_mini_batches)
                else:
                    batch_generator = self.sampler.buffer.mini_batch_generator(self.n_value_mini_batches)
                for batch in batch_generator:
                    res = self.train_value_mini_batch(batch)
                    for key, (tag, value) in res.items():
                        self.train_info.setdefault(key, (tag, []))[1].append(value)

    def train_policy_mini_batch(self, samples):
        """Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
        
        Returns:
            training_stats {dict} -- Losses, entropy, kl-divergence and clip fraction
        """
        # Retrieve the agent's memory to feed the model
        actor_memory, actor_memory_mask = None, None
        # Case Recurrence: the recurrent cell state is treated as the memory. Only the initial hidden states are selected.
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"]
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"], samples["cxs"])
            (actor_memory, _) = self.model.unpack_recurrent_cell(recurrent_cell)
        # Case Transformer: the episodic memory is based on activations that were previously gathered throughout an episode
        if self.transformer is not None:
            actor_memory = samples["memories"][..., 0]
            actor_memory_mask = samples["memory_mask"]
        
        # Forward model -> policy, value, memory, gae
        policy, _, gae = self.model.forward_actor(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                    samples["vec_obs"] if self.vec_obs_space is not None else None,
                                    actor_memory, actor_memory_mask,
                                    self.sampler.buffer.actual_sequence_length,
                                    samples["actions"])

        # Policy Loss
        # Retrieve and process log_probs from each policy branch
        log_probs, entropies = [], []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples["actions"][:, i]))
            entropies.append(policy_branch.entropy())
        log_probs = torch.stack(log_probs, dim=1)
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)

        # Remove paddings if recurrence is used
        if self.recurrence is not None:
            log_probs = log_probs[samples["loss_mask"]]
            entropies = entropies[samples["loss_mask"]] 

        # Compute surrogates
        # Determine advantage normalization
        if self.configs["trainer"]["advantage_normalization"] == "minibatch":
            normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        elif self.configs["trainer"]["advantage_normalization"] == "no":
            normalized_advantage = samples["advantages"]
        else:
            normalized_advantage = samples["normalized_advantages"]
        # Repeat is necessary for multi-discrete action spaces
        advs = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.policy_clip_range, 1.0 + self.policy_clip_range) * advs
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropies = torch.stack(entropies, dim=1).sum(1).reshape(-1)
        # Mean reduction of entropy bonus
        entropy_bonus = entropies.mean()

        # Advantage estimation as part of the DAAC algorithm (Raileanu & Fergus, 2021)
        if self.use_daac:
            adv_loss = (normalized_advantage - gae)**2
            adv_loss = adv_loss.mean()

        # Complete loss
        if self.use_daac:
            loss = -(policy_loss + self.beta * entropy_bonus) + self.adv_coefficient * adv_loss
        else:
            loss = -(policy_loss + self.beta * entropy_bonus)

        # Compute gradients
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_parameters, max_norm=self.pi_max_grad_norm)
        self.policy_optimizer.step()

        # Monitor additional training statistics
        approx_kl = ((ratio - 1.0) - log_ratio).mean()  # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > self.policy_clip_range).float().mean()

        out = {**compute_gradient_stats(self.model.actor_modules, prefix = "actor"),
                "policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
                "loss": (Tag.LOSS, loss.cpu().data.numpy()),
                "entropy": (Tag.OTHER, entropy_bonus.cpu().data.numpy()),
                "kl_divergence": (Tag.OTHER, approx_kl.cpu().data.numpy()),
                "clip_fraction": (Tag.OTHER, clip_fraction.cpu().data.numpy())}
        if self.use_daac:
            out["advantage_loss"] = (Tag.LOSS, adv_loss.cpu().data.numpy())
        return out

    def train_value_mini_batch(self, samples):
        """Optimizes the value function based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
        
        Returns:
            training_stats {dict} -- Value loss
        """
        # Retrieve the agent's memory to feed the model
        critic_memory, critic_memory_mask = None, None
        # Case Recurrence: the recurrent cell state is treated as the memory. Only the initial hidden states are selected.
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"]
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"], samples["cxs"])
            (critic_memory, _) = self.model.unpack_recurrent_cell(recurrent_cell)
        # Case Transformer: the episodic memory is based on activations that were previously gathered throughout an episode
        if self.transformer is not None:
            critic_memory = samples["memories"][..., 0]
            critic_memory_mask = samples["memory_mask"]
        
        value, _ = self.model.forward_critic(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                    samples["vec_obs"] if self.vec_obs_space is not None else None,
                                    critic_memory, critic_memory_mask,
                                    self.sampler.buffer.actual_sequence_length,
                                    samples["actions"])

        # Remove paddings if recurrence is used
        if self.recurrence is not None:
            value = value[samples["loss_mask"]]

        # Value loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.value_clip_range, max=self.value_clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Compute gradients
        self.value_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_parameters, max_norm=self.v_max_grad_norm)
        self.value_optimizer.step()

        return {**compute_gradient_stats(self.model.critic_modules, prefix = "critic"),
                "value_loss": (Tag.LOSS, vf_loss.cpu().data.numpy())}

    def step_decay_schedules(self, update):
        self.policy_learning_rate = polynomial_decay(self.policy_lr_schedule["initial"], self.policy_lr_schedule["final"],
                                        self.policy_lr_schedule["max_decay_steps"], self.policy_lr_schedule["power"], update)
        self.value_learning_rate = polynomial_decay(self.value_lr_schedule["initial"], self.value_lr_schedule["final"],
                                        self.value_lr_schedule["max_decay_steps"], self.value_lr_schedule["power"], update)
        self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
        self.policy_clip_range = polynomial_decay(self.pi_cr_schedule["initial"], self.pi_cr_schedule["final"],
                                        self.pi_cr_schedule["max_decay_steps"], self.pi_cr_schedule["power"], update)
        self.value_clip_range = polynomial_decay(self.v_cr_schedule["initial"], self.v_cr_schedule["final"],
                                        self.v_cr_schedule["max_decay_steps"], self.v_cr_schedule["power"], update)

        # Apply learning rates to optimizers
        for pg in self.policy_optimizer.param_groups:
            pg["lr"] = self.policy_learning_rate
        for pg in self.value_optimizer.param_groups:
            pg["lr"] = self.value_learning_rate

        return {
            "policy_learning_rate": (Tag.DECAY, self.policy_learning_rate),
            "value_learning_rate": (Tag.DECAY, self.value_learning_rate),
            "beta": (Tag.DECAY, self.beta),
            "policy_clip_range": (Tag.DECAY, self.policy_clip_range),
            "value_clip_range": (Tag.DECAY, self.value_clip_range)
        }

    def collect_checkpoint_data(self, update) -> dict:
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        checkpoint_data["policy_optimizer"] = self.policy_optimizer.state_dict()
        checkpoint_data["value_optimizer"] = self.value_optimizer.state_dict()
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint) -> None:
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])