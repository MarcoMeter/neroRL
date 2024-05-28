# Preprint Paper

Section 4 details the implemented Transformer-XL architecture in greater detail:
[Memory Gym: Towards Endless Tasks to Benchmark Memory Capabilities of Agents](https://arxiv.org/abs/2309.17207)

A more streamline implementation (less features for simplicity) can be found in this repository:
[TransformerXL as Episodic Memory in Proximal Policy Optimization](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)


## Potential improvements for the Transformer-XL baseline:
- Only one hidde state is computed anew, while all the other ones are cached. Computing more new hidden states could improve performance.
- Use torch's MultiHeadAttention, which should be faster.
- Implement different kinds of positional encodings as rotary or relative ones.