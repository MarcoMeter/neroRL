import torch

from neroRL.trainers.policy_gradient.base import BaseTrainer
from neroRL.utils.utils import masked_mean

class PPOSharedGradientTrainer(BaseTrainer):
    def __init__(self, configs, worker_id, run_id, low_mem_fix, out_path):
        super().__init__(configs, worker_id, run_id=run_id, low_mem_fix=low_mem_fix, out_path=out_path)

    def train(self, clip_range, beta):
        """Trains several PPO epochs over one batch of data while dividing the batch into mini batches.
        
        Arguments:
            learning_rate {float} -- The current learning rate
            clip_range {float} -- The current clip range
            beta {float} -- The current entropy bonus coefficient
            
        Returns:
            {numpy.ndarray} -- Mean training statistics of one training epoch"""
        train_info = []

        for _ in range(self.epochs):
            # Retrieve the to be trained mini_batches via a generator
            # Use the recurrent mini batch generator for training a recurrent policy
            if self.recurrence is not None:
                mini_batch_generator = self.buffer.recurrent_mini_batch_generator()
            else:
                mini_batch_generator = self.buffer.mini_batch_generator()
            for mini_batch in mini_batch_generator:
                res = self._train_mini_batch(clip_range=clip_range,
                                            beta = beta,
                                            samples=mini_batch)
                train_info.append(res)
        # Return the mean of the training statistics
        return train_info

    def _train_mini_batch(self, samples, clip_range, beta):
        """ Optimizes the policy based on the PPO algorithm

        Arguments:
            samples {dict} -- The sampled mini-batch to optimize the model
            learning_rate {float} -- The to be used learning rate
            clip_range {float} -- The to be used clip range
            beta {float} -- The to be used entropy coefficient
        
        Returns:
            training_stats {list} -- Losses, entropy, kl-divergence and clip fraction
        """
        # Retrieve sampled recurrent cell states to feed the model
        recurrent_cell = None
        if self.recurrence is not None:
            if self.recurrence["layer_type"] == "gru":
                recurrent_cell = samples["hxs"].unsqueeze(0)
            elif self.recurrence["layer_type"] == "lstm":
                recurrent_cell = (samples["hxs"].unsqueeze(0), samples["cxs"].unsqueeze(0))
        
        policy, value, _ = self.model(samples['vis_obs'] if self.visual_observation_space is not None else None,
                                    samples['vec_obs'] if self.vector_observation_space is not None else None,
                                    recurrent_cell,
                                    self.device,
                                    self.buffer.actual_sequence_length)
        
        # Policy Loss
        # Retreive and process log_probs from each policy branch
        log_probs = []
        for i, policy_branch in enumerate(policy):
            log_probs.append(policy_branch.log_prob(samples['actions'][:, i]))
        log_probs = torch.stack(log_probs, dim=1)

        # Compute surrogates
        normalized_advantage = (samples["advantages"] - samples["advantages"].mean()) / (samples["advantages"].std() + 1e-8)
        # Repeat is necessary for multi-discrete action spaces
        normalized_advantage = normalized_advantage.unsqueeze(1).repeat(1, len(self.action_space_shape))
        ratio = torch.exp(log_probs - samples['log_probs'])
        surr1 = ratio * normalized_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * normalized_advantage
        policy_loss = torch.min(surr1, surr2)
        policy_loss = masked_mean(policy_loss, samples["loss_mask"])

        # Value
        sampled_return = samples['values'] + samples['advantages']
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range, max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = masked_mean(vf_loss, samples["loss_mask"])
        vf_loss = .25 * vf_loss

        # Entropy Bonus
        entropies = []
        for policy_branch in policy:
            entropies.append(policy_branch.entropy())
        entropy_bonus = masked_mean(torch.stack(entropies, dim=1).sum(1).reshape(-1), samples["loss_mask"])

        # Complete loss
        loss = -(policy_loss - vf_loss + beta * entropy_bonus)

        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Monitor training statistics
        approx_kl = masked_mean((torch.exp(ratio) - 1) - ratio, samples["loss_mask"])
        clip_fraction = (abs((ratio - 1.0)) > clip_range).type(torch.FloatTensor).mean()

        return [policy_loss.cpu().data.numpy(),
                vf_loss.cpu().data.numpy(),
                loss.cpu().data.numpy(),
                entropy_bonus.cpu().data.numpy(),
                approx_kl.cpu().data.numpy(),
                clip_fraction.cpu().data.numpy()]