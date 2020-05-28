# Steps to add another environment

1. To support another environment, the first task is to write a wrapper like [CartPole](../neroRL/environments/cartpole_wrapper.py). This class has to inherit [`Env`](../neroRL/environments/env.py) to meet the to be used interface, which is similar to the gym interface. The major difference is that this interface supports vector and visual observation spaces simultaneously.

2. [wrapper.py](../neroRL/environments/wrapper.py) has to be extended. Also, the import for the environments have to be added.

3. At last the environment part of the configuration file should match your new addition.
