# neroRL

neroRL is a PyTorch based framework for Deep Reinforcement Learning, which I'm currently developing while pursuing my PhD in this academic field.
Its focus is set on environments that are procedurally generated, while providing some usefull tools for experimenting and analyzing a trained behavior.
One core feature encompasses recurrent policies

# Features
- Environments:
  - [Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)
  - [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
  - [Procgen](https://github.com/openai/procgen)
  - [Gym-Minigrid](https://github.com/maximecb/gym-minigrid) (Vector (one-hot) or Visual Observations (84x84x3))
  - [Gym CartPole](https://github.com/openai/gym) using masked velocity
- Proximal Policy Optimization (PPO)
  - Discrete and Multi-Discrete Action Spaces
  - Vector and Visual Observation Spaces (either alone or simultaneously)
  - [Recurrent Policies using Truncated Backpropagation Through Time](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)
  - Shared and None-Shared Parameters across the Policy and the Value Function
- Decoupled Proximal Policy Optimization
  - Same features as PPO, but the parameters, as well as the gradients, are decoupled
  - Decoupled Advantage Actor-Critic (DAAC, [Raileanu & Fergus, 2021](https://arxiv.org/abs/2102.10330))
    - The actor network estimates the policy and the advantage function

# Obstacle Tower Challenge
Originally, this work started out by achieving the 7th place during the [Obstacle Tower Challenge](https://blogs.unity3d.com/2019/08/07/announcing-the-obstacle-tower-challenge-winners-and-open-source-release/) by using a relatively simple FFCNN. This [video](https://www.youtube.com/watch?v=P2rBDHBHxcM) presents some footage of the approach and the trained behavior:

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=P2rBDHBHxcM
" target="_blank"><img src="http://img.youtube.com/vi/P2rBDHBHxcM/0.jpg" 
alt="Rising to the Obstacle Tower Challenge" width="240" height="180" border="10" /></a></p>

Recently we published a [paper](https://arxiv.org/abs/2004.00567) at CoG 2020 (best paper candidate) that analyzes the taken approach. Additionally the model was trained on 3 level designs and was evaluated on the two left out ones. The results can be reproduced using the [obstacle-tower-challenge](https://github.com/MarcoMeter/neroRL/tree/obstacle-tower-challenge) branch.

# Getting Started

To get started check out the [docs](/docs/)!
