# Recurrent Policy Implementation and Usage

The recurrent policy implementation is based on our [baseline/reference implementation](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt).

## Flow of processing the training data

1. Training data
   1. Training data is sampled from the current policy
   2. Sampled data is split into episodes
   3. Episodes are split into sequences (based on the `sequence_length` hyperparameter)
   4. Zero padding is applied to retrieve sequences of fixed length
   5. Recurrent cell states are collected from the beginning of the sequences (truncated bptt)
2. Forward pass of the model
   1. While feeding the model for optimization, the data is flattened to feed an entire batch (faster)
   2. Before feeding it to the recurrent layer, the data is reshaped to `(num_sequences, sequence_length, data)`
3. Loss computation
   1. Zero padded values are masked during the computation of the losses

## Further details

- Padding
    - zeros are added to the end of a sequence
    - Only model input data undergoes padding

- Recurrent cell state initialization choices:
    - zero
    - one
    - mean
        - the mean of the hidden states of the collected training data
    - sample
        - sampled from a gaussian distribution
        - based on the mean of the hidden states of the collected training data
        - the standard deviation is set to 0.01
       