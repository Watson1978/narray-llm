"""GPT-2 124M inference in PyTorch eager, fp32.

Unlike python/gpt2.py this is NOT a structural mirror of the Ruby code. It is
the ceiling reference: what an ecosystem with fused kernels does with the same
weights and the same task. So it uses scaled_dot_product_attention, keeps the
KV cache as one preallocated tensor per layer, and leaves the rest to torch.

What it does keep identical, because those are the comparison's premises:

  * the same gpt2_124M.bin (../docs/checkpoint-format-gpt2.md), not HuggingFace
  * fp32 only, TF32 disabled on both matmul and cudnn
  * prefill / decode phases, a [maxT, C] cache per layer, greedy decoding
  * torch.compile is not used: its first-call compilation would make the
    timing protocol a different conversation. Eager only.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
DEVICE = torch.device("cuda" if GPU else "cpu")

# DTYPE mirrors the Ruby side. F.layer_norm folds its statistics in fp32 on its
# own, and SDPA takes the mask value as it comes, so the weights and the cache
# are the only places the dtype has to be threaded through.
DTYPE = os.environ.get("DTYPE", "").lower()
DT = {"": torch.float32, "fp32": torch.float32, "float32": torch.float32,
      "fp16": torch.float16, "float16": torch.float16,
      "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}.get(DTYPE)
if DT is None:
    raise ValueError(f"unknown DTYPE {DTYPE!r}; use fp32, fp16 or bf16")

# Both flags matter and both default to True on Ampere and later. With either
# one on, matmuls run in TF32 and the generated tokens stop matching.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

MAGIC = 20_240_326
VERSION = 3
HEADER_BYTES = 256 * 4
TOKENIZER_MAGIC = 20_240_328
LAYERNORM_EPS = 1e-5


@dataclass(frozen=True)
class Config:
    max_seq_len: int
    vocab_size: int
    num_layers: int
    num_heads: int
    channels: int
    padded_vocab_size: int


def tensor_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    c, l = config.channels, config.num_layers
    return {
        "wte": (config.padded_vocab_size, c), "wpe": (config.max_seq_len, c),
        "ln1w": (l, c), "ln1b": (l, c),
        "qkvw": (l, 3 * c, c), "qkvb": (l, 3 * c),
        "attprojw": (l, c, c), "attprojb": (l, c),
        "ln2w": (l, c), "ln2b": (l, c),
        "fcw": (l, 4 * c, c), "fcb": (l, 4 * c),
        "fcprojw": (l, c, 4 * c), "fcprojb": (l, c),
        "lnfw": (c,), "lnfb": (c,),
    }


def load_checkpoint(path: str, device) -> tuple[Config, dict[str, torch.Tensor]]:
    with open(path, "rb") as f:
        header = struct.unpack("<256i", f.read(HEADER_BYTES))
        if header[0] != MAGIC:
            raise ValueError(f"{path}: bad magic {header[0]}, expected {MAGIC}")
        if header[1] != VERSION:
            raise ValueError(f"{path}: unsupported version {header[1]}")
        config = Config(max_seq_len=header[2], vocab_size=header[3], num_layers=header[4],
                        num_heads=header[5], channels=header[6], padded_vocab_size=header[7])
        params = {}
        for name, shape in tensor_shapes(config).items():
            count = int(np.prod(shape))
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading {name}")
            host = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            params[name] = torch.from_numpy(host).to(device=device, dtype=DT)
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


def load_tokenizer(path: str) -> tuple[int, list[bytes]]:
    with open(path, "rb") as f:
        header = struct.unpack("<256I", f.read(HEADER_BYTES))
        if header[0] != TOKENIZER_MAGIC:
            raise ValueError(f"{path}: bad magic {header[0]}")
        version, vocab_size = header[1], header[2]
        eot = header[3] if version >= 2 else 50_256
        blob = f.read()
    table, offset = [], 0
    for _ in range(vocab_size):
        length = blob[offset]
        offset += 1
        table.append(blob[offset:offset + length])
        offset += length
    return eot, table


def decode_text(table: list[bytes], ids) -> str:
    return b"".join(table[int(i)] for i in ids).decode("utf-8", errors="replace")


class KVCache:
    """One preallocated [maxT, C] pair per layer, as on the Ruby side. Held as
    [1, NH, maxT, hs] because that is the layout SDPA wants. Each layer tracks
    its own position, so nothing depends on the layers staying in lockstep."""

    def __init__(self, num_layers, max_seq_len, num_heads, head_size, device, batch_size=1):
        if batch_size < 1:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        shape = (num_layers, batch_size, num_heads, max_seq_len, head_size)
        self.keys = torch.zeros(shape, dtype=DT, device=device)
        self.values = torch.zeros(shape, dtype=DT, device=device)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.positions = [0] * num_layers

    @staticmethod
    def bytes_for(num_layers, max_seq_len, channels, batch_size=1):
        return 2 * batch_size * num_layers * max_seq_len * channels * 4

    def reset(self):
        self.positions = [0] * len(self.positions)

    def append(self, layer, keys, values):
        rows = keys.shape[2]
        position = self.positions[layer]
        if position + rows > self.max_seq_len:
            raise ValueError(f"kv cache overflow on layer {layer}: {position} + {rows} rows "
                             f"exceeds max_seq_len {self.max_seq_len}")
        self.keys[layer, :, :, position:position + rows] = keys
        self.values[layer, :, :, position:position + rows] = values
        self.positions[layer] = position + rows

    def view(self, layer):
        length = self.positions[layer]
        return self.keys[layer, :, :, :length], self.values[layer, :, :, :length]


class Model:
    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.config, params = load_checkpoint(checkpoint_path, device)
        c = self.config
        self.head_size = c.channels // c.num_heads
        self.wte = params["wte"][:c.vocab_size].contiguous()
        self.wte_t = self.wte.t().contiguous()
        self.wpe = params["wpe"]
        self.lnfw, self.lnfb = params["lnfw"], params["lnfb"]
        self.layers = [{
            "ln1w": params["ln1w"][i].contiguous(), "ln1b": params["ln1b"][i].contiguous(),
            "qkvw_t": params["qkvw"][i].t().contiguous(), "qkvb": params["qkvb"][i].contiguous(),
            "attprojw_t": params["attprojw"][i].t().contiguous(),
            "attprojb": params["attprojb"][i].contiguous(),
            "ln2w": params["ln2w"][i].contiguous(), "ln2b": params["ln2b"][i].contiguous(),
            "fcw_t": params["fcw"][i].t().contiguous(), "fcb": params["fcb"][i].contiguous(),
            "fcprojw_t": params["fcprojw"][i].t().contiguous(),
            "fcprojb": params["fcprojb"][i].contiguous(),
        } for i in range(c.num_layers)]

    def parameter_bytes(self):
        tensors = [self.wte, self.wte_t, self.wpe, self.lnfw, self.lnfb]
        tensors += [t for layer in self.layers for t in layer.values()]
        return sum(t.numel() * t.element_size() for t in tensors)

    def new_cache(self, batch_size=1):
        c = self.config
        return KVCache(c.num_layers, c.max_seq_len, c.num_heads, self.head_size, self.device,
                       batch_size=batch_size)

    def _norm(self, x, weight, bias):
        return F.layer_norm(x, (self.config.channels,), weight, bias, LAYERNORM_EPS)

    def _split_heads(self, qkv, batch_size, seq_len):
        c = self.config
        q, k, v = qkv.split(c.channels, dim=-1)
        shape = (batch_size, seq_len, c.num_heads, self.head_size)
        return (q.view(shape).transpose(1, 2), k.view(shape).transpose(1, 2),
                v.view(shape).transpose(1, 2))

    def _block(self, x, w, layer, batch_size, seq_len, cache, causal):
        c = self.config
        qkv = torch.addmm(w["qkvb"], self._norm(x, w["ln1w"], w["ln1b"]), w["qkvw_t"])
        q, k, v = self._split_heads(qkv, batch_size, seq_len)
        if cache is not None:
            cache.append(layer, k, v)
            k, v = cache.view(layer)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        attn = attn.transpose(1, 2).reshape(batch_size * seq_len, c.channels)

        x = x + torch.addmm(w["attprojb"], attn, w["attprojw_t"])
        hidden = F.gelu(torch.addmm(w["fcb"], self._norm(x, w["ln2w"], w["ln2b"]), w["fcw_t"]),
                        approximate="tanh")
        return x + torch.addmm(w["fcprojb"], hidden, w["fcprojw_t"])

    def forward(self, tokens, last_only=False, cache=None):
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        batch_size, seq_len = ids.shape
        c = self.config
        if seq_len > c.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {c.max_seq_len}")
        if cache is not None and cache.batch_size != batch_size:
            raise ValueError(f"a cache of batch size {cache.batch_size} cannot take "
                             f"{batch_size} rows")

        x = self.wte[ids.reshape(-1)].reshape(batch_size, seq_len, c.channels) + self.wpe[:seq_len]
        x = x.reshape(batch_size * seq_len, c.channels)
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, batch_size, seq_len, cache, causal=seq_len > 1)
        x = self._norm(x, self.lnfw, self.lnfb)
        if last_only:
            x = x.reshape(batch_size, seq_len, c.channels)[:, seq_len - 1].contiguous()
        return (x @ self.wte_t).reshape(batch_size, 1 if last_only else seq_len, c.vocab_size)

    def prefill(self, tokens, cache):
        cache.reset()
        return self.forward(tokens, last_only=True, cache=cache)

    def decode(self, token_id, position, cache):
        c = self.config
        if position >= c.max_seq_len:
            raise ValueError(f"position {position} exceeds max_seq_len {c.max_seq_len}")
        ids = list(token_id) if isinstance(token_id, (list, tuple)) else [token_id]
        if len(ids) != cache.batch_size:
            raise ValueError(f"got {len(ids)} token ids for a cache of batch size "
                             f"{cache.batch_size}")
        batch_size = len(ids)
        if batch_size == 1:
            x = (self.wte[ids[0]] + self.wpe[position]).reshape(1, c.channels)
        else:
            index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            x = self.wte[index] + self.wpe[position]
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, batch_size, 1, cache, causal=False)
        x = self._norm(x, self.lnfw, self.lnfb)
        return (x @ self.wte_t).reshape(batch_size, 1, c.vocab_size)


class Generator:
    def __init__(self, model, eot_token=None):
        self.model = model
        self.eot_token = eot_token

    def generate(self, prompt, max_new_tokens, stop_at_eot=True, cache=True):
        tokens = [int(t) for t in prompt]
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        limit = self.model.config.max_seq_len
        if len(tokens) + max_new_tokens > limit:
            raise ValueError(f"{len(tokens)} prompt tokens + {max_new_tokens} generated tokens "
                             f"exceeds max_seq_len {limit}")
        return (self._cached if cache else self._recomputing)(tokens, max_new_tokens, stop_at_eot)

    def generate_batch(self, prompts, max_new_tokens, stop_at_eot=True):
        """B sequences at once, prompts included. Mirrors the Ruby side: one
        position for the whole batch, so every prompt has to be the same length,
        and a finished sequence keeps stepping and is cut afterwards."""
        rows = [[int(t) for t in prompt] for prompt in prompts]
        if not rows:
            raise ValueError("a batch needs at least one prompt")
        if any(not row for row in rows):
            raise ValueError("every prompt must contain at least one token")
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError(f"every prompt must be the same length, got "
                             f"{[len(row) for row in rows]}")

        length = lengths.pop()
        limit = self.model.config.max_seq_len
        if length + max_new_tokens > limit:
            raise ValueError(f"{length} prompt tokens + {max_new_tokens} generated tokens "
                             f"exceeds max_seq_len {limit}")

        cache = self.model.new_cache(batch_size=len(rows))
        logits = self.model.prefill([row[:] for row in rows], cache)
        produced = [[] for _ in rows]
        for step in range(max_new_tokens):
            # One readback per step for the whole batch, not one per sequence:
            # iterating the device tensor instead would call item() on each
            # element and cost B transfers.
            ids = logits.reshape(len(rows), -1).argmax(dim=1).cpu().tolist()
            for i, one in enumerate(ids):
                produced[i].append(one)
            if step == max_new_tokens - 1:
                break
            if stop_at_eot and self.eot_token is not None and \
                    all(self.eot_token in seq for seq in produced):
                break
            logits = self.model.decode(ids, length + len(produced[0]) - 1, cache)

        return [row + self._cut_at_eot(produced[i], stop_at_eot)
                for i, row in enumerate(rows)]

    def _cut_at_eot(self, sequence, stop_at_eot):
        if not stop_at_eot or self.eot_token is None or self.eot_token not in sequence:
            return sequence
        return sequence[:sequence.index(self.eot_token) + 1]

    def _recomputing(self, tokens, max_new_tokens, stop_at_eot):
        for _ in range(max_new_tokens):
            token = int(self.model.forward([tokens], last_only=True).reshape(-1).argmax())
            tokens.append(token)
            if stop_at_eot and self.eot_token is not None and token == self.eot_token:
                break
        return tokens

    def _cached(self, tokens, max_new_tokens, stop_at_eot):
        cache = self.model.new_cache()
        logits = self.model.prefill([tokens], cache)
        produced = 0
        while produced < max_new_tokens:
            token = int(logits.reshape(-1).argmax())
            tokens.append(token)
            produced += 1
            if stop_at_eot and self.eot_token is not None and token == self.eot_token:
                break
            if produced == max_new_tokens:
                break
            logits = self.model.decode(token, len(tokens) - 1, cache)
        return tokens


def synchronize():
    if GPU:
        torch.cuda.synchronize()


def sdpa_backend(model, seq_len, kv_len):
    """Which SDPA kernel torch actually picks for a given shape.

    can_use_* answers "would this kernel accept these inputs", and the dispatch
    tries them in the priority order below, so the first one that both accepts
    and is enabled is the one that runs. Reported per shape because prefill and
    decode do not have to land on the same kernel."""
    c = model.config
    q = torch.zeros(1, c.num_heads, seq_len, model.head_size,
                    device=model.device, dtype=DT)
    k = torch.zeros(1, c.num_heads, kv_len, model.head_size,
                    device=model.device, dtype=DT)
    causal = seq_len > 1

    if model.device.type != "cuda":
        # can_use_* lives under torch.backends.cuda and has no CPU counterpart,
        # so which kernel the CPU dispatch picks cannot be queried. Say so
        # rather than guess.
        return "(CPU: 判定 API 無し)"

    params = torch.backends.cuda.SDPAParams(q, k, k, None, 0.0, causal, False)
    candidates = (
        ("cudnn", torch.backends.cuda.can_use_cudnn_attention,
         torch.backends.cuda.cudnn_sdp_enabled),
        ("flash", torch.backends.cuda.can_use_flash_attention,
         torch.backends.cuda.flash_sdp_enabled),
        ("efficient", torch.backends.cuda.can_use_efficient_attention,
         torch.backends.cuda.mem_efficient_sdp_enabled),
    )
    for name, can_use, enabled in candidates:
        try:
            if enabled() and can_use(params):
                return name
        except Exception:
            continue
    return "math"
