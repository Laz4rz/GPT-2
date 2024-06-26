from gpt2 import GPT, GPTConfig
from scratch import check_dtypes

import sys
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
# device = torch.device("mps")
print(f"Device: {device}")

# reproducibility
seed = 1337
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
elif device == "mps":
    torch.mps.manual_seed(seed)
print("Torch seed is", seed)

# dataloader
train_loader = DataLoaderLite(16, 256)

# init model
model = GPT(GPTConfig()) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
model.to(device)
# model = torch.compile(model) if device.type == "cuda" else model # cpu compile is stuck on MBP
print("Model initialized successfully!")

# get logits
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

start_total = datetime.now()
metrics = dict(loss=[], tokens_per_sec=[], batch_time=[])

for i in range(10):
    start = datetime.now()

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    # torch.mps.synchronize() # useful for per epoch timings, only lets cpu continue after gpu finishes work
    
    end = datetime.now()
    tokens_per_sec = (train_loader.B * train_loader.T) / (end-start).total_seconds()
    loss = loss.item()
    batch_time = end-start
    metrics["loss"].append(loss), metrics["tokens_per_sec"].append(tokens_per_sec), metrics["batch_time"].append(batch_time)
    print(f"Step: {i}, Loss: {loss:.6f}, Batch time: {batch_time}, Tokens/sec: {tokens_per_sec:.2f}")

end_total = datetime.now()

mean_batch_time = sum(map(lambda x: x.total_seconds(), metrics["batch_time"])) / len(metrics["batch_time"])
mean_tokens_per_sec = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])
print(f"Runtime: {(end_total-start_total)}\nDevice: {device}\nMean Batch time: {mean_batch_time:.2f}s\nMean tokens/sec: {mean_tokens_per_sec:.2f}")

