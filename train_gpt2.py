from gpt2 import GPT, GPTConfig

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class DataLoaderLite:
    def __init__(self, B, T, inputs=None):
        self.B = B
        self.T = T

        enc = tiktoken.get_encoding("gpt2")
        if inputs is None: inputs = "input.txt"
        with open("input.txt", "r") as file:
            text = file.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded a total of {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} tokens")

        self.current_position = 0 
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        # reset if current_position > tokens length
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0 
        return x, y
    
    def reset(self):
        self.current_position = 0
        return self


# infer device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# dataloader
train_loader = DataLoaderLite(4, 32)

# init model
model  = GPT(GPTConfig()) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
print("Model loaded successfully!")

# get logits
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step: {i}, Loss: {loss.item()}")

import sys; sys.exit(0)
model.eval() # put model in eval mode, works for layers like: Dropout, BatchNorm, etc.
model.to(device)

max_length = 30
num_return_sequences = 5

# generating loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)                   # get logits from non-grad forward pass
        logits = logits[:, -1, :]           # take last logits from each batch
        probs  = F.softmax(logits, dim=-1)  # get probabilities from logits
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # get top50 probs and its indices
        ix      = torch.multinomial(topk_probs, 1)   # get random idx from topk distribution (not really dist, since sum=/=1, but yknow)
        xcol    = torch.gather(topk_indices, -1, ix) # get tokens corresponding to sampled ixs
        x       = torch.cat((x, xcol), dim=1)        # concat previous tokens, with sampled ixs tokens
         
# decode generated
for i in range(num_return_sequences):
    tokens  = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"> {decoded}") 
