# Steps to add another layer

1. To support another type of layer for the actor critic class regardless if the weights are shared or not, the first task is to write a module like an [encoder](../neroRL/trainers/PPO/models/encoder.py), [recurrent layer](../neroRL/trainers/PPO/models/recurrent.py) or [hidden layer](../neroRL/trainers/PPO/models/hidden_layer.py).

2. Now [base.py](../neroRL/trainers/PPO/models/base.py) has to be extended. In the first step, you should import your newly created layer. Afterwards, you go into the proper *create_...* method and add another condition that if your layer is selected in the configuration, a new instance will get returned. At the final part, you should check that the input and output arguments of the method are matched then you are ready to deploy it.

3. In the very last step, you have to change the model part of the configuration file that it matches your new addition.
