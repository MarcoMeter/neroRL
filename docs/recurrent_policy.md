# Recurrent Policy Implementation and Usage

WIP

## Flow of processing the training data

1. Training data is sampled from the current policy
2. Sampled data is split into episodes
3. (Optional) episodes are split into sequences
4. Apply zero padding to retrieve sequences of fixed length
5. For computing losses, padded values get masked

## Further details

- Padding
    - zeros are added to the end of a sequence
    - every kind of data (e.g. observations, advantages, values, log probabilities, ...) is padded

- Recurrent cell state initialization choices:
    - zero
    - one
    - mean
        - the mean of the hidden states of the collected training data
    - sample
        - sampled from agaussian distribution
        - based on the mean of the hidden states of the collected training data
        - the standard deviation is set to 0.01
       
- Fake Recurrence
    - recurrent cell states are added to the experience tuples of the sampled training data
    - BPTT is omitted by not feeding sequences
       
## TODO

- Add fully connected layer between the visual encoder and the recurrent layer?
- Extrapolate padding instead of zero padding?
- Add pasdding before or after the data of the sequence?
- Fix up checkpoints, because the values for the hidden state initialization need now to be stored inside the checkpoints.
