import math
from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 # why im not sure at first glance
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # batched q, k, v projections for (all heads) this im not sure why for all 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization (why??? where??? what???)
        # more of a mask than bias, but follows HF/openai naming (also not sure why)
        # buffers are non-learnable parameters
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, embedding dimension (n_embd)
        # calculate q, k, v for all heads in batch and move head forward (to be the batch???)
        # nh is the number of heads, hs is the head size, C (number of channels) = nh * hs
        # in this GPT-2 (124M) model, nh = 12, hs = 64, so nh * hs = C = 768 channels in transformer
        qkv = self.c_attn(x) # TODO: bias of self.c_attn.bias does not match
        q, k, v = qkv.split(self.n_embd, dim=2) # chech what it does, what are the shapes
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        # attention -- materialize the large (T, T) for all queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # contiguous() places the tensor in a contiguous block of memory, used after transpose, view, etc.
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257 # number of tokens, 50k merges BPE, 256 bytes tokens, 1 <|endoftext|>
    n_layer: int = 12       # number of layers
    n_head: int = 12        # number of heads
    n_embd: int = 768       # embedding dimension


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT uses no bias

    def forward(self, idx): # why is this called idx dunno 
        B, T = idx.size()
        assert T < self.config.block_size, f"sequence length ({T=}) is greater than the pretrained position embeddings value ({self.config.block_size=})"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # positional embeddings (T, n_emb)
        tok_emb = self.transformer.wte(idx) # token embeddings (batch size, seq length, n_emb)
        x = pos_emb + tok_emb               # superposition of positions and token embeddings
        for block in self.transformer.h:    # pass sequentially through all attention blocks
            x = block(x)
        x = self.transformer.ln_f(x)        # pass through layernorm
        x = self.lm_head(x)                 # pass through projection from high-dim (latent?) to vocab_size
        return x                            # return logits

    @classmethod
    def from_pretrained(cls, model_type, verbose=False):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], "model_type should be one of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']"
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} model weights from transformers")

        # n_head, n_layer, n_embd are model specific
        config_args = {
            "gpt2"         : dict(n_head=12, n_layer=12, n_embd=768),  # 124M
            "gpt2-medium"  : dict(n_head=16, n_layer=16, n_embd=1024), # 350M
            "gpt2-large"   : dict(n_head=36, n_layer=20, n_embd=1280), # 774M
            "gpt2-xl"      : dict(n_head=48, n_layer=25, n_embd=1600), # 1558M
        }[model_type]

        # load architecture
        config_args["vocab_size"] = 50257 # constant for all GPT-2 models
        config_args["block_size"] = 1024  # constant for all GPT-2 models
        config  = GPTConfig(**config_args)
        model   = GPT(config)
        sd      = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if ".attn.bias" not in k] # remove attn.bias buffer, we omit buffers for some reason i dont know why, it doesnt even matter how hardy I try

        # load huggingface model weights
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf    = hf_model.state_dict()

        # choose, match and copy weights
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if ".attn.bias" not in k and "attn.masked_bias" not in k] # remove attn.bias and attn.masked_bias
        transposed = [".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"]
        # some of the original weights use Conv1D, but we want to load vanilla
        # which is why we need to transpose some of the weights
        assert len(sd_keys) == len(sd_keys_hf), f"mismatch in number of keys: custom {len(sd_keys)} vs hf {len(sd_keys_hf)}"
        for k in sd_keys_hf:
            if verbose: print(f"   Loading: {k}")
            if verbose: print("     from: {:15s}\n       to: {:15s}".format(str(sd_hf[k].shape), str(sd[k].shape)))
            if any(t in k for t in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"mismatch in shape for special: {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t().contiguous()) # .t() works only for 2D weights, .T for any
            else:
                assert sd_hf[k].shape == sd[k].shape, f"mismatch in shape for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# class method allows constructing a class through a class method call
# GPT.from_pretrained('gpt2') would be a way to instantiate a GPT model from a pre-trained checkpoint

# init model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT.from_pretrained('gpt2', verbose=False)
print(f"Device: {device}")
# print(model) # buffers are not visible here, to show them we need to look at model.buffers()
print("Model loaded successfully!")

model.eval() # put model in eval mode, works for layers like: Dropout, BatchNorm, etc.
model.to(device)

num_return_sequences = 5
max_length = 30

# encode
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

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
