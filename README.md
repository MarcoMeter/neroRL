# neroRL
<p align="center"><img src="/docs/img/nero.png" alt="neroRL" width="180" height="180"></p>

neroRL is a PyTorch based framework for Deep Reinforcement Learning, which I'm currently developing while pursuing my PhD in this academic field.
Its focus is set on environments that are procedurally generated, while providing some usefull tools for experimenting and analyzing a trained behavior.

# Features
- Environments:
  - [Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)
  - [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
  - [Procgen](https://github.com/openai/procgen)
  - [Gym-Minigrid](https://github.com/maximecb/gym-minigrid)
  - [Gym CartPole](https://github.com/openai/gym) using masked velocity
- Proximal Policy Optimization
  - Discrete and Multi-Discrete Action Space
  - Vector and Visual Observation Space (either alone or simultaneously)
  - (Experimental) Recurrent Policy using a GRU layer

# Obstacle Tower Challenge
Originally, this work started out by achieving the 7th place during the [Obstacle Tower Challenge](https://blogs.unity3d.com/2019/08/07/announcing-the-obstacle-tower-challenge-winners-and-open-source-release/) by using a relatively simple FFCNN. This [video](https://www.youtube.com/watch?v=P2rBDHBHxcM) presents some footage of the approach and the trained behavior:

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=P2rBDHBHxcM
" target="_blank"><img src="http://img.youtube.com/vi/P2rBDHBHxcM/0.jpg" 
alt="Rising to the Obstacle Tower Challenge" width="240" height="180" border="10" /></a></p>

Recently we published a [paper](https://arxiv.org/abs/2004.00567) at CoG2020 that analyzes the taken approach. Additionally the model was trained on 3 level designs and was evaluated on the two left out ones.

# Getting Started

To get started check out the [docs](/docs/)!

# Short-Term Development Goals

- Finalizing Recurrent Policy
- Add Layer-Wise Relevance Propagation to reason the agent's decisions
