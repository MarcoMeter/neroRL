import torch
import numpy as np
from torch import nn, optim

from neroRL.nn.actor_critic import create_actor_critic_model
from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import compute_gradient_stats, batched_index_select
from neroRL.utils.decay_schedules import polynomial_decay
from neroRL.utils.monitor import Tag

class PPOTrainer(BaseTrainer):
    """PPO implementation according to Schulman et al. 2017. It supports multi-discrete action spaces as well as visual 
    and vector obsverations (either alone or simultaenously).
    """
    def __init__(self, configs, sample_device, train_device, worker_id, run_id, out_path, seed = 0, compile_model = False):
        """
        Initializes distinct members of the PPOTrainer

        Arguments:
            configs {dict} -- The whole set of configurations (e.g. training and environment configs)
            sample_device {torch.device} -- The device used for sampling training data.
            train_device {torch.device} -- The device used for model optimization.
            worker_id {int} -- Specifies the offset for the port to communicate with the environment, which is needed for Unity ML-Agents environments (default: {1})
            run_id {string} -- The run_id is used to tag the training runs (directory names to store summaries and checkpoints) (default: {"default"})
            out_path {str} -- Determines the target directory for saving summaries, logs and model checkpoints. (default: "./")
            compile_model {bool} -- If true, the model is compiled before training (default: {False})
        """
        super().__init__(configs, sample_device, train_device, worker_id, run_id=run_id, out_path=out_path, seed=seed, compile_model=compile_model)

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

        # Setup visual observation reconstruction members
        if self.use_obs_reconstruction:
            self.obs_recon_schedule = self.configs["trainer"]["obs_reconstruction_schedule"]
            self.obs_recon_coef = self.obs_recon_schedule["initial"]
            self.bce_loss = nn.BCELoss()

        # Setup ground truth estimation members
        if self.use_ground_truth_estimation:
            self.ground_truth_estimation_schedule = self.configs["trainer"]["ground_truth_estimator_schedule"]
            self.ground_truth_estimation_coef = self.ground_truth_estimation_schedule["initial"]
            self.mse_loss = nn.MSELoss()

    def create_model(self) -> None:
        self.use_obs_reconstruction = self.configs["trainer"]["obs_reconstruction_schedule"]["initial"] > 0.0
        self.use_ground_truth_estimation = self.configs["trainer"]["ground_truth_estimator_schedule"]["initial"] > 0.0
        return create_actor_critic_model(self.configs["model"], self.vis_obs_space, self.vec_obs_space,
                                         self.ground_truth_space, self.action_space_shape, self.sample_device)

    def train(self):
        train_info = {}

        # Normalize advantages batch-wise if desired
        if self.configs["trainer"]["advantage_normalization"] == "batch":
            advantages = self.sampler.buffer.samples_flat["advantages"]
            self.sampler.buffer.samples_flat["normalized_advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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
        for key, (tag, values) in train_info.items():
            train_info[key] = (tag, np.mean(values))

        # Format specific values for logging inside the base class
        formatted_string = "loss={:.3f} pi_loss={:.3f} vf_loss={:.3f} entropy={:.3f}".format(
            train_info["loss"][1], train_info["policy_loss"][1], train_info["value_loss"][1], train_info["entropy"][1])
        
        if self.use_ground_truth_estimation:
            formatted_string += " gt_loss={:.3f}".format(train_info["gt_loss"][1])

        if self.use_obs_reconstruction:
            formatted_string += " r_loss={:.3f}".format(train_info["r_loss"][1])

        # Return the mean of the training statistics
        return train_info, formatted_string

    def train_mini_batch(self, samples):
        """Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
        
        Returns:
            training_stats {dict} -- Losses, entropy, kl-divergence and clip fraction
        """
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
        policy, value, _ = self.model(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                    samples["vec_obs"] if self.vec_obs_space is not None else None,
                                    memory = memory, mask = mask, memory_indices = memory_indices,
                                    sequence_length = self.sampler.buffer.actual_sequence_length)
        
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
            value = value[samples["loss_mask"]]
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
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        log_ratio = log_probs - samples["log_probs"]
        ratio = torch.exp(log_ratio)
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = policy_loss.mean()

        # Value loss
        sampled_return = samples["values"] + samples["advantages"]
        clipped_value = samples["values"] + (value - samples["values"]).clamp(min=-self.clip_range, max=self.clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = vf_loss.mean()

        # Entropy Bonus
        entropy_bonus = entropies.mean()

        # Complete loss
        loss = -(policy_loss - self.vf_loss_coef * vf_loss + self.beta * entropy_bonus)

        # Add observation reconstruction loss
        if self.use_obs_reconstruction:
            # Forward decoder
            decoder_output = self.model.reconstruct_observation()
            vis_obs = samples["vis_obs"]
            # Remove paddings if recurrence is used
            if self.recurrence is not None:
                decoder_output = decoder_output[samples["loss_mask"]]
                vis_obs = vis_obs[samples["loss_mask"]]
            # Compute reconstruction loss
            reconstruction_loss = self.bce_loss(decoder_output, vis_obs)
            loss += self.obs_recon_coef * reconstruction_loss

        # plot several images of viso_obs next to the reconstructed images
        # if self.use_obs_reconstruction and self.vis_obs_space is not None:
        #     if self.current_update % 10 == 0 and self.current_update > 1:
        #         self.plot_obs_reconstruction(vis_obs, decoder_output)

        # Add ground truth estimation loss
        if self.use_ground_truth_estimation:
            # Forward ground truth estimator
            estimation = self.model.estimate_ground_truth()
            target = samples["ground_truth"]
            # Remove paddings if recurrence is used
            if self.recurrence is not None:
                estimation = estimation[samples["loss_mask"]]
                target = target[samples["loss_mask"]]
            # Compute ground truth estimation loss
            estimation_loss = self.mse_loss(estimation, target)
            loss += self.ground_truth_estimation_coef * estimation_loss

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        # Monitor additional training statistics
        approx_kl = ((ratio - 1.0) - log_ratio).mean()  # http://joschu.net/blog/kl-approx.html
        clip_fraction = (abs((ratio - 1.0)) > self.clip_range).float().mean()

        # Retrieve modules for monitoring the gradient norm
        modules = self.model.actor_critic_modules

        out = {**compute_gradient_stats(modules),
                "policy_loss": (Tag.LOSS, policy_loss.cpu().data.numpy()),
                "value_loss": (Tag.LOSS, vf_loss.cpu().data.numpy()),
                "loss": (Tag.LOSS, loss.cpu().data.numpy()),
                "entropy": (Tag.OTHER, entropy_bonus.cpu().data.numpy()),
                "kl_divergence": (Tag.OTHER, approx_kl.cpu().data.numpy()),
                "clip_fraction": (Tag.OTHER, clip_fraction.cpu().data.numpy())}
        
        if self.use_obs_reconstruction:
            out["r_loss"] = (Tag.LOSS, reconstruction_loss.cpu().data.numpy())

        if self.use_ground_truth_estimation:
            out["gt_loss"] = (Tag.LOSS, estimation_loss.cpu().data.numpy())

        return out

    def plot_obs_reconstruction(self, vis_obs, decoder_output):
        import matplotlib.pyplot as plt
        # plot the first 5 images of viso_obs next to the reconstructed images
        vis_obs = vis_obs.cpu().data.numpy()
        decoder_output = decoder_output.cpu().data.numpy()
        for i in range(100):
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(vis_obs[i].transpose(1, 2, 0))
            axes[1].imshow(decoder_output[i].transpose(1, 2, 0))
            plt.savefig("./obs/obs_reconstruction_" + str(i) + ".png")
            plt.close()
        assert False

    def step_decay_schedules(self, update):
        self.learning_rate = polynomial_decay(self.lr_schedule["initial"], self.lr_schedule["final"],
                                        self.lr_schedule["max_decay_steps"], self.lr_schedule["power"], update)
        self.beta = polynomial_decay(self.beta_schedule["initial"], self.beta_schedule["final"],
                                        self.beta_schedule["max_decay_steps"], self.beta_schedule["power"], update)
        self.clip_range = polynomial_decay(self.cr_schedule["initial"], self.cr_schedule["final"],
                                        self.cr_schedule["max_decay_steps"], self.cr_schedule["power"], update)
        if self.use_obs_reconstruction:
            self.obs_recon_coef = polynomial_decay(self.obs_recon_schedule["initial"], self.obs_recon_schedule["final"], 
                                            self.obs_recon_schedule["max_decay_steps"], self.obs_recon_schedule["power"], update)
            
        if self.use_ground_truth_estimation:
            self.ground_truth_estimation_coef = polynomial_decay(self.ground_truth_estimation_schedule["initial"], self.ground_truth_estimation_schedule["final"],
                                            self.ground_truth_estimation_schedule["max_decay_steps"], self.ground_truth_estimation_schedule["power"], update)
        
        # Apply learning rate to optimizer
        for pg in self.optimizer.param_groups:
            pg["lr"] = self.learning_rate

        # Report decayed values
        out = {
            "learning_rate": (Tag.DECAY, self.learning_rate),
            "beta": (Tag.DECAY, self.beta),
            "clip_range": (Tag.DECAY, self.clip_range)
        }

        # Report current observation reconstruction coefficient if applicable
        if self.use_obs_reconstruction:
            out["obs_recon_coef"] = (Tag.DECAY, self.obs_recon_coef)

        return out

    def collect_checkpoint_data(self, update):
        checkpoint_data = super().collect_checkpoint_data(update)
        checkpoint_data["model"] = self.model.state_dict()
        checkpoint_data["optimizer"] = self.optimizer.state_dict()
        return checkpoint_data

    def apply_checkpoint_data(self, checkpoint):
        super().apply_checkpoint_data(checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])