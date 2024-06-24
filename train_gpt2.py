from gpt2 import GPT, GPTConfig

import tiktoken
import torch
import torch.nn as nn
from datetime import datetime
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
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        self.max_batches = len(tokens) // (B * T)
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
device = "cpu"
print(f"Device: {device}")

# dataloader
train_loader = DataLoaderLite(4, 32)

# init model
model  = GPT(GPTConfig()) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
model.to(device)
print("Model loaded successfully!")

# get logits
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

start = datetime.now()

for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"Step: {i}, Loss: {loss.item()}")

end = datetime.now()
print(f"Runtime {(end-start)} with device {device}")
