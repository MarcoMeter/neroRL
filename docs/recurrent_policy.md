# Recurrent Policy Implementation and Usage

WIP

## Flow of processing the training data

1. Training data is sampled from the current policy
2. Sampled data is split into episodes
3. (Optional) episodes are split into sequences
4. Apply zero zero padding to retrieve sequences of fixed length
5. For computing losses, padded values get masked

## Further details

- Padding
    - zero are added to the end of a sequence
    - every kind of data (e.g. observations, advantages, values, log probabilities, ...) is padded

- Hidden state initialization
    - Hidden states are initialized using zeros

## TODO

- Add fully connected layer between the visual encoder and the recurrent layer?
- Implement fake reccurrence, where hidden states are added to the experience tuples while sampling data
- Extrapolate padding instead of zero padding?
- Add pasdding before or after the data of the sequence?
- Implement different options for initializing the hidden states