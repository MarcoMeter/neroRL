import torch
import numpy as np

from torch import nn, optim

from neroRL.utils.monitor import Tag
from neroRL.utils.utils import compute_gradient_stats, batched_index_select
from neroRL.trainers.policy_gradient.ppo_shared import PPOTrainer
from neroRL.utils.decay_schedules import polynomial_decay

class DecoderTrainer(PPOTrainer):
    """Trainer for the observation reconstruction decoder only. This is used to analyze whether the model learned
    useful representations. The decoder is trained to reconstruct the observations from the latent space.
    """
    def __init__(self, configs, sample_device, train_device, worker_id, run_id, out_path, seed=0, compile_model=False):
        super().__init__(configs, sample_device, train_device, worker_id, run_id, out_path, seed, compile_model)
        # Override optimizer to only tune the decoder's parameters
        self.optimizer = optim.Adam(self.model.vis_decoder.parameters(), lr=self.learning_rate)
        # Obs reconstruction members
        self.obs_recon_schedule = self.configs["trainer"]["obs_reconstruction_schedule"]
        self.obs_recon_coef = self.obs_recon_schedule["initial"]
        self.bce_loss = nn.BCELoss()

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
            if key == "r_loss":
                stats[key + "_min"] = (tag, np.min(values))
                stats[key + "_max"] = (tag, np.max(values))

        # Format specific values for logging inside the base class
        formatted_string = "r_loss={:.3f} r_loss_max={:.3f} r_loss_min={:.3f}".format(
            stats["r_loss"][1], stats["r_loss_max"][1], stats["r_loss_min"][1])

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
            memory = batched_index_select(samples["memories"], 1, samples["memory_indices"])
            mask = samples["memory_mask"]
            memory_indices = samples["memory_indices"]

        # Forward model -> policy, value, memory, gae
        with torch.no_grad():
            policy, value, _ = self.model(samples["vis_obs"] if self.vis_obs_space is not None else None,
                                        samples["vec_obs"] if self.vec_obs_space is not None else None,
                                        memory = memory, mask = mask, memory_indices = memory_indices,
                                        sequence_length = self.sampler.buffer.actual_sequence_length)

        # Observation reconstruction loss
        # Forward decoder
        decoder_output = self.model.reconstruct_observation()
        vis_obs = samples["vis_obs"]
        # Remove paddings if recurrence is used
        if self.recurrence is not None:
            decoder_output = decoder_output[samples["loss_mask"]]
            vis_obs = vis_obs[samples["loss_mask"]]
        # Compute reconstruction loss
        reconstruction_loss = self.bce_loss(decoder_output, vis_obs)

        # plot several images of vis_obs next to the reconstructed images
        # if self.use_obs_reconstruction and self.vis_obs_space is not None:
        #     if self.current_update % 30 == 0 and self.current_update > 1:
        #         self.plot_obs_reconstruction(vis_obs, decoder_output)
        # loss = reconstruction_loss

        # Compute gradients
        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        out = {**compute_gradient_stats({"vis_decoder": self.model.vis_decoder}),
                "r_loss": (Tag.LOSS, reconstruction_loss.cpu().data.numpy())}

        return out
    
    def plot_obs_reconstruction(self, vis_obs, decoder_output):
        import matplotlib.pyplot as plt
        # plot the first 5 images of vis_obs next to the reconstructed images
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