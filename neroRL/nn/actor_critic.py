import numpy as np
import torch
import torch.nn as nn

from neroRL.nn.base import ActorCriticBase
from neroRL.nn.heads import MultiDiscreteActionPolicy, ValueEstimator, GroundTruthEstimator

class ActorCriticSharedWeights(ActorCriticBase):
    """A flexible shared weights actor-critic model that supports:
            - Multi-discrete action spaces
            - Dict observation spaces
            - Recurrent polices (either GRU or LSTM)
            - TransformerXL-based episodic memory
            - Ground truth estimation
            - Observation reconstruction
    """
    def __init__(self, config, obs_space, ground_truth_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Model config
            obs_space {spaces.Dict} -- Dimensions of the visual observation space
            ground_truth_space {box} -- Dimensions of the ground truth space (None if not available)
            action_space_shape {tuple} -- Dimensions of the action space
            use_decoder {bool} -- Whether to use a decoder for observation reconstruction or not (default: {False})
        """
        ActorCriticBase.__init__(self, config)

        # Create the base model
        self.obs_encoders, self.recurrent_layer, self.transformer, self.body, self.vis_decoder = self.create_base_model(config, obs_space)

        # Policy head/output
        self.actor_policy = MultiDiscreteActionPolicy(self.out_features_body, action_space_shape, self.activ_fn)

        # Value function head/output
        self.critic = ValueEstimator(self.out_features_body, self.activ_fn)

        # Ground truth estimator head if configured
        self.ground_truth_estimator = None
        if self.ground_truth_estimator_config is not None:
            self.ground_truth_estimator = GroundTruthEstimator(self.memory_dim, ground_truth_space.shape[0], self.activ_fn)

        # Organize all modules inside a dictionary
        # This will be used for collecting gradient statistics inside the trainer
        self.actor_critic_modules = nn.ModuleDict({
            "body": self.body,
            "actor_head": self.actor_policy,
            "critic_head": self.critic
        })
        
        for key, value in self.obs_encoders.items():
            self.actor_critic_modules["encoder_" + key] = value

        if self.vis_decoder is not None:
            self.actor_critic_modules["vis_decoder"] = self.vis_decoder

        if self.ground_truth_estimator is not None:
            self.actor_critic_modules["ground_truth_estimator"] = self.ground_truth_estimator

        if self.transformer is not None:
            for b, block in enumerate(self.transformer.transformer_blocks):
                self.actor_critic_modules["transformer_" + str(b)] = block
        
        if self.recurrence_config is not None:
            self.actor_critic_modules["recurrent_layer"] = self.recurrent_layer

    def forward(self, obs, memory = None, mask = None, memory_indices = None, sequence_length = 1):
        """Forward pass of the model

            obs {dict} -- Observations as dict
            memory {torch.tensor} -- Reucrrent cell state or episodic memory (None if not available)
            mask {torch.tensor} -- Memory mask (None if the model is not transformer-based)
            memory_indices {torch.tesnor} -- Indices to select the positional encoding that mathes the memory window (None of the model is not transformer-based)
            sequence_length {int} -- Length of the fed sequences

        Returns:
            {list} -- Policy: List featuring categorical distributions respectively for each policy branch
            {torch.tensor} -- Value function: Value
            {torch.tensor or tuple} -- Current memory representation or recurrent cell state (None if memory is not used)
        """
        # Forward observation encoder
        encoded_features = [encoder(obs[key]) for key, encoder in self.obs_encoders.items()]

        # Concatenate if multiple encoders
        if len(encoded_features) > 1:
            h = torch.cat(encoded_features, dim=1)
            attach_to_cnn = False
            if self.decoder_config is not None:
                attach_to_cnn = self.decoder_config.get("attach_to") == "cnn"
            if attach_to_cnn and any(k in ["vis_obs", "visual_observation"] for k in obs.keys()):
                self.decoder_h = h
        else:
            h = encoded_features[0]

        # Forward reccurent layer (GRU or LSTM) if available
        if self.recurrence_config is not None:
            h, memory = self.recurrent_layer(h, memory, sequence_length)

        # Forward transformer if available
        if self.transformer is not None:
            h, memory = self.transformer(h, memory, mask, memory_indices)

        # Store hidden representation for observation reconstruction
        if self.vis_decoder is not None:
            if self.decoder_config["attach_to"] == "memory":
                self.decoder_h = h

        # Store hidden representation for ground truth estimation
        if self.ground_truth_estimator is not None:
            self.ground_truth_estimator_h = h

        # Feed network body
        h = self.body(h)

        # Head: Value function
        value = self.critic(h)
        # Head: Policy branches
        pi = self.actor_policy(h)

        return pi, value, memory
    
    def reconstruct_observation(self):
        """Reconstructs the observation from the visual encoder features
        
        Returns:
            {torch.tensor} -- Reconstructed observation"""
        # Allow gradients to only flow through the decoder?
        if self.decoder_config["detach_gradient"]:
            self.decoder_h = self.decoder_h.detach()
        y = self.vis_decoder(self.decoder_h)
        return y
    
    def estimate_ground_truth(self):
        """Estimates the ground truth from the memory representation
        
        Returns:
            {torch.tensor} -- Estimated ground truth"""
        # Allow gradients to only flow through the estimator?
        if self.ground_truth_estimator_config["detach_gradient"]:
            self.ground_truth_estimator_h = self.ground_truth_estimator_h.detach()
        y = self.ground_truth_estimator(self.ground_truth_estimator_h)
        return y

def create_actor_critic_model(model_config, observation_space, ground_truth_space, action_space_shape, device):
    """Creates a shared weights actor critic model.

    Arguments:
        model_config {dict} -- Model config
        observation_space {spaces.Dict} -- Observation space dictionary
        action_space_shape {tuple} -- Dimensions of the action space
        device {torch.device} -- Current device
        use_decoder {bool} -- Whether to use a decoder for observation reconstruction or not (default: {False})

    Returns:
        {ActorCriticBase} -- The created actor critic model
    """
    return ActorCriticSharedWeights(model_config, observation_space,ground_truth_space, action_space_shape).to(device)