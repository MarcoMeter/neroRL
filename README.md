# neroRL

neroRL is a PyTorch based research framework for Deep Reinforcement Learning specializing on Transformer and Recurrent Agents based on Proximal Policy Optimization.
Its focus is set on environments that are procedurally generated, while providing some usefull tools for experimenting and analyzing a trained behavior.
This is a research framework.

# Features
- Environments:
  - [Memory Gym](https://github.com/MarcoMeter/drl-memory-gym)
  - [Obstacle Tower](https://github.com/Unity-Technologies/obstacle-tower-env)
  - [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents)
  - [Procgen](https://github.com/openai/procgen)
  - [Minigrid](https://github.com/Farama-Foundation/Minigrid) (Vector (one-hot) or Visual Observations (84x84x3))
  - [Gym CartPole](https://github.com/openai/gym) using masked velocity
  - [DM Ballet](https://github.com/deepmind/deepmind-research/tree/master/hierarchical_transformer_memory/pycolab_ballet)
  - [RandomMaze](https://github.com/zuoxingdong/mazelab)
- Proximal Policy Optimization (PPO)
  - Discrete and Multi-Discrete Action Spaces
  - Vector and Visual Observation Spaces (either alone or simultaneously)
  - [Recurrent Policies using Truncated Backpropagation Through Time](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)
  - [Episodic Memory based on Transformer-XL](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)

# Memory Gym ICLR 2022 Paper

This repository is used to achieve the results on the Memory Gym Environments given the following paper:

```bibtex
@inproceedings{pleines2023memory,
title={Memory Gym: Partially Observable Challenges to Memory-Based Agents},
author={Marco Pleines and Matthias Pallasch and Frank Zimmer and Mike Preuss},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=jHc8dCx6DDr}
}
```

# Obstacle Tower Challenge
Originally, this work started out by achieving the 7th place during the [Obstacle Tower Challenge](https://blogs.unity3d.com/2019/08/07/announcing-the-obstacle-tower-challenge-winners-and-open-source-release/) by using a relatively simple FFCNN. This [video](https://www.youtube.com/watch?v=P2rBDHBHxcM) presents some footage of the approach and the trained behavior:

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=P2rBDHBHxcM
" target="_blank"><img src="http://img.youtube.com/vi/P2rBDHBHxcM/0.jpg" 
alt="Rising to the Obstacle Tower Challenge" width="240" height="180" border="10" /></a></p>

Recently we published a [paper](https://arxiv.org/abs/2004.00567) at CoG 2020 (best paper candidate) that analyzes the taken approach. Additionally the model was trained on 3 level designs and was evaluated on the two left out ones. The results can be reproduced using the [obstacle-tower-challenge](https://github.com/MarcoMeter/neroRL/tree/obstacle-tower-challenge) branch.

```bibtex
@inproceedings{pleines2020otc,
author    = {Marco Pleines and
             Jenia Jitsev and
             Mike Preuss and
             Frank Zimmer},
title     = {Obstacle Tower Without Human Demonstrations: How Far a Deep Feed-Forward
             Network Goes with Reinforcement Learning},
booktitle = {{IEEE} Conference on Games, CoG 2020, Osaka, Japan, August 24-27,
             2020},
pages     = {447--454},
publisher = {{IEEE}},
year      = {2020},
url       = {https://doi.org/10.1109/CoG47356.2020.9231802},
doi       = {10.1109/CoG47356.2020.9231802},
}
```

# Getting Started

To get started check out the [docs](/docs/)!
