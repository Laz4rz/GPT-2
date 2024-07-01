from gpt2 import GPT, GPTConfig
from scratch import check_dtypes

import os
import math
import tiktoken
import torch
import torch.nn as nn
from datetime import datetime
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


class DataLoaderLite:
    def __init__(self, B, T, inputs=None, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        enc = tiktoken.get_encoding("gpt2")
        if inputs is None: inputs = "input.txt"
        with open("input.txt", "r") as file:
            text = file.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded a total of {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        self.max_batches = len(tokens) // (B * T)
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes # im not entirely sure of this, like intuition wise, but this seems similar to how CUDA kernels do it
        # reset if current_position > tokens length
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y
    
    def reset(self):
        self.current_position = 0
        return self
    

# Distributed Data Parallel (DDP) setup
# 1. simple run: python train_gpt2.py
# 2. ddp run: torchrun --standalone --nproc_per_node=2 train_gpt2.py
# standalone is used for single node multi-gpu training
ddp = int(os.environ.get("RANK", -1)) != -1 
# ^if doesnt exist then we return -1, and by -1 or else decide whether ddp
if ddp:
    assert torch.cuda.is_available(), "no CUDA device available"
    init_process_group(backend="nccl") # on backends: https://pytorch.org/docs/stable/distributed.html
    # these env variables are set by torchrun
    ddp_rank = int(os.environ["RANK"]) # RANK is the gloval process id, ie. if you have two machines (nodes) with 4 GPUs each, you will have 8 RANKs 
    ddp_local_rank = int(os.environ["LOCAL_RANK"]) # LOCAL_RANK is an id given to this specific process, ie. if you have two machines with 4 GPUs each, you will have 4 LOCAL_RANKs
    ddp_world_size = int(os.environ["WORLD_SIZE"]) # WORLD_SIZE is the  number of GPUs running the DDP
    print(f"DDP: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # master process is responsible for logs, checkpoints, etc.
else:
    # run without DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # infer device type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps")
    print(f"DDP: disabled")
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
assert total_batch_size % (B * T * ddp_world_size) == 0, "total_batch_size needs to be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


# precision of matmuls FP32 -> TF32
torch.set_float32_matmul_precision("high")


# dataloader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)


# init model
model = GPT(GPTConfig(vocab_size=50304)) # random model initialization, will still produce some readable sentence parts due to tokenizer construction
model.to(device)
# model = torch.compile(model) if device.type == "cuda" else model # cpu compile is stuck on MBP
if ddp:
    DDP(model, device_ids=[ddp_local_rank]) # hmm local rank instead of rank, interesting.....
if master_process: 
    print("Model initialized successfully!")


# create optimizer
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
# apparently AdamW is bugfixed Adam according to Andrej
optimizer = model.configure_optimizers(weight_decay=0.01, lr=6e-4, device=device)


# training loop
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
        if ddp:
            # boolean telling backward() to sync gradients across all ranks if true, we only want it to happen
            # on the last micro_step of the grad_accum_steps
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward() # it does += automatically, which is why we normally have to zero_grad
    if ddp:
        # average loss across all ranks, god I love this
        # if we didnt do this, we would only get master process loss
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) 
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    # torch.mps.synchronize() # useful for per epoch timings, only lets cpu continue after gpu finishes work
    # torch.cuda.synchronize()
    end = datetime.now()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (end-start).total_seconds()
    loss = loss.item()
    batch_time = end-start
    metrics["loss"].append(loss), metrics["tokens_per_sec"].append(tokens_per_sec), metrics["batch_time"].append(batch_time)
    if master_process: 
        print(f"Step: {step}, Loss: {loss_accum:.6f}, Norm: {norm:.4f}, lr: {lr:.4e}, Batch time: {batch_time}, Tokens/sec: {tokens_per_sec:.2f}")

end_total = datetime.now()

# apply all reduce 
mean_batch_time = sum(map(lambda x: x.total_seconds(), metrics["batch_time"])) / len(metrics["batch_time"])
mean_tokens_per_sec = sum(metrics["tokens_per_sec"]) / len(metrics["tokens_per_sec"])
print(f"Runtime: {(end_total-start_total)}\nDevice: {device}\nMean Batch time: {mean_batch_time:.2f}s\nMean tokens/sec: {mean_tokens_per_sec:.2f}")

# destroy process group
if ddp:
    destroy_process_group()
    print("Process group destroyed:", ddp_local_rank)
