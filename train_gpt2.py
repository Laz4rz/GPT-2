from gpt2 import GPT, GPTConfig
from scratch import check_dtypes

import math
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

# gradient accumulation
B = 4  # this controls the accumulating batch size
T = 1024 # this does not, if you're short in token length this is not a way to do it
total_batch_size = B*T*4 # 2**19=~0.5M tokens but a power of 2 as well
assert total_batch_size % (B * T) == 0, "total_batch_size needs to be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# precision of matmuls FP32 -> TF32
torch.set_float32_matmul_precision("high")

# dataloader
train_loader = DataLoaderLite(B=B, T=T)

# init model
model = GPT(GPTConfig(vocab_size=50304)) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
model.to(device)
# model = torch.compile(model) if device.type == "cuda" else model # cpu compile is stuck on MBP
print("Model initialized successfully!")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 12
def get_lr(step):
    # linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # constat min lr for when cosine decay ends
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi + decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# get logits
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = model.configure_optimizers(weight_decay=0.01, lr=6e-4, device=device)

start_total = datetime.now()
metrics = dict(loss=[], tokens_per_sec=[], batch_time=[])
for step in range(max_steps):
    start = datetime.now()
    optimizer.zero_grad()
    loss_accum = 0.0 # only metric
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        # we need a mean of all grad_accum_steps batch elements, which is
        # grad_acum_steps * batch_size, by defualt the loss will be divided
        # by batch_size due to reduce mode "mean", so if we accumulate gradients
        # we still need to divide each of its grad_accums_steps by the number of
        # grad_accum_steps
        loss = loss / grad_accum_steps
        loss_accum += loss.detach() # only a printing metric!
        loss.backward() # it does += automatically, which is why we normally have to zero_grad
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # torch.mps.synchronize() # useful for per epoch timings, only lets cpu continue after gpu finishes work
    # torch.cuda.synchronize()
    end = datetime.now()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (end-start).total_seconds()
    loss = loss.item()
    batch_time = end-start
    metrics["loss"].append(loss), metrics["tokens_per_sec"].append(tokens_per_sec), metrics["batch_time"].append(batch_time)
    print(f"Step: {step}, Loss: {loss_accum:.6f}, Norm: {norm:.4f}, lr: {lr:.4e}, Batch time: {batch_time}, Tokens/sec: {tokens_per_sec:.2f}")

end_total = datetime.now()

mean_batch_time = sum(map(lambda x: x.total_seconds(), metrics["batch_time"])) / len(metrics["batch_time"])
mean_tokens_per_sec = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])
print(f"Runtime: {(end_total-start_total)}\nDevice: {device}\nMean Batch time: {mean_batch_time:.2f}s\nMean tokens/sec: {mean_tokens_per_sec:.2f}")

