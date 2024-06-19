# GPT-2
Following master Karpathy with GPT-2 implementation and training

## Walkthrough notes:

- Noisy embedding lines mean that the model could be trained some more

- GPT-2 is decoder only, therefore its architecture is:
![alt text](image.png)
  Also:
  - the positional embeddings are learned
    the way they are noisy in the original model tells that its undertrained
  - layer norms are before, not after blocks
    this is because clean residual pathways are a desirable architecture choice
    this allows gradients to flow from the very top uninterrupted, due to addition
    just passing them down
  - layer norm was added after final self attention
  - h in module dict is the whole gray block
  - mlp is map, attention is reduce

## Other quick wisdom
- torch buffers are basically non-learnable model tensors

## my whims
- train with rope
- play with params on small models
- play with other activation functions
