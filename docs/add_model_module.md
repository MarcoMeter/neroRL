# Steps to add another model module (visual encoder, vector encoder, recurrent layer, body, head)

1. To support another type of layer for the actor critic class regardless, the first task is to write a module like an [encoder](../neroRL/nn/encoder.py), [recurrent layer](../neroRL/nn/recurrent.py) or [body](../neroRL/nn/body.py). That module has to inherit the [custom](../neroRL/nn/module.py) module class, because some addiontal functionality is desired.

2. Now [base.py](../neroRL/nn/base.py) has to be extended. In the first step, you should import your new created layer. Afterwards, you go into the proper *create_...* method and add another condition that if your layer is selected in the configuration, a new instance will get returned. At the final part, you should check that the input and output arguments of the method are matched then you are ready to deploy it.

3. In the very last step, you have to change the model part of the configuration file that it reflects your new addition.
