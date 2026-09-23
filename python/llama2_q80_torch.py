"""PyTorch over llama2.c's version 2 (Q8_0) checkpoint, the one runq.c reads.

Only the matmuls differ from llama2_torch.py. Everything else stays the native
op the framework offers, so this compares backends and not recipes.

PyTorch cannot take the int8 weight as it stands. torch.einsum refuses mixed
dtypes, so the weight is cast per call, which is the same fp32 copy cumo made
before #477. torch._int_mm would avoid it but wants the second operand's column
count to be a multiple of 8, and decoding one token makes it 1.

  python/.venv/bin/python python/bench_llama2.py --impl torch --model stories110M_q80
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from llama2_torch import (GPU, DEVICE, BOS_TOKEN, Config, Generator,  # noqa: F401
                          KVCache, RMSNORM_EPS, apply_rope, decode_text,
                          load_tokenizer, rope_tables, synchronize)
import llama2_torch as _float

MAGIC = 0x616B3432  # "ak42"
VERSION = 2
HEADER_BYTES = 256
Q_MAX = 127.0


@dataclass(frozen=True)
class Quantized:
    """q is [out, in / group_size, group_size]; scales is [out, in / group_size]."""

    q: torch.Tensor
    scales: torch.Tensor
    shape: tuple


def round_half_away(x):
    """C's roundf. torch.round sends halves to even, and adding a signed half
    first is wrong for the largest float below 0.5, where the add itself rounds
    up. Doubling the fraction is exact."""
    t = torch.trunc(x)
    return t + torch.trunc((x - t) * 2.0)


def quantize_groups(x, group_size):
    """runq.c:145. One scale per group of group_size, symmetric around zero."""
    rows = x.reshape(x.numel() // group_size, group_size)
    scale = rows.abs().amax(dim=1, keepdim=True) / Q_MAX
    # A group of exact zeros would divide by zero. Dividing it by one instead
    # gives the zeros it should quantize to.
    positive = torch.ceil(torch.clamp(scale, 0.0, 1.0))
    return round_half_away(rows / (scale + (1.0 - positive))), scale.reshape(-1)


def qmatmul(weight, xq, xs):
    """runq.c:317, folding each group's products in fp32 rather than int32.

    weight.q has to be widened first: einsum will not take int8 against float,
    and an int8 accumulator would wrap on a group sum that reaches 1,032,256.
    """
    ival = torch.einsum("ogk,gk->og", weight.q.float(), xq)
    return (ival * weight.scales * xs).sum(dim=1).reshape(1, weight.shape[0])


def load_checkpoint(path: str, device) -> tuple[Config, int, dict]:
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) != HEADER_BYTES:
            raise ValueError(f"{path}: truncated header")
        magic, version = struct.unpack_from("<Ii", head, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}, expected ak42")
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, expected {VERSION}")
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq = \
            struct.unpack_from("<7i", head, 8)
        shared, = struct.unpack_from("<B", head, 36)
        group_size, = struct.unpack_from("<i", head, 37)
        config = Config(dim=dim, hidden_dim=hidden_dim, num_layers=n_layers,
                        num_heads=n_heads, num_kv_heads=n_kv_heads, vocab_size=vocab,
                        max_seq_len=seq, shared_classifier=shared == 1)

        params: dict = {}

        def read(count, dtype, what):
            size = np.dtype(dtype).itemsize * count
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError(f"{path}: truncated while reading {what}")
            return np.frombuffer(raw, dtype=dtype)

        for name, shape in (("rms_att_weight", (n_layers, dim)),
                            ("rms_ffn_weight", (n_layers, dim)),
                            ("rms_final_weight", (dim,))):
            host = read(int(np.prod(shape)), np.float32, name).reshape(shape).copy()
            params[name] = torch.from_numpy(host).to(device)

        def read_one(shape, what):
            out, inner = shape
            if inner % group_size:
                raise ValueError(f"{what}: row of {inner} is not a multiple of {group_size}")
            groups = inner // group_size
            q = read(out * inner, np.int8, f"{what} (int8)").reshape(out, groups, group_size)
            s = read(out * groups, np.float32, f"{what} (scales)").reshape(out, groups)
            return Quantized(torch.from_numpy(q.copy()).to(device),
                             torch.from_numpy(s.copy()).to(device), shape)

        q_dim = n_heads * config.head_size
        params["q_tokens"] = read_one((vocab, dim), "q_tokens")
        for name, shape in (("wq", (q_dim, dim)), ("wk", (config.kv_dim, dim)),
                            ("wv", (config.kv_dim, dim)), ("wo", (dim, q_dim)),
                            ("w1", (hidden_dim, dim)), ("w2", (dim, hidden_dim)),
                            ("w3", (hidden_dim, dim))):
            params[name] = [read_one(shape, f"{name}[{i}]") for i in range(n_layers)]
        params["wcls"] = params["q_tokens"] if config.shared_classifier \
            else read_one((vocab, dim), "wcls")

        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the weights")
    return config, group_size, params


def dequantize(weight):
    out, _ = weight.shape
    groups = weight.scales.shape[1]
    return (weight.q.float() * weight.scales.reshape(out, groups, 1)).reshape(weight.shape)


class Model(_float.Model):
    """runq.c's forward(). Decode only, one token at a time, as runq.c is."""

    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.config, self.group_size, params = load_checkpoint(checkpoint_path, device)
        c = self.config
        # runq.c:204 dequantizes the embedding once and then reads rows out of it.
        self.token_embedding = dequantize(params["q_tokens"]).contiguous()
        self.rms_final = params["rms_final_weight"].contiguous()
        self.wcls = params["wcls"]
        self.layers = [{
            "rms_att_weight": params["rms_att_weight"][i].contiguous(),
            "rms_ffn_weight": params["rms_ffn_weight"][i].contiguous(),
            "wq": params["wq"][i], "wk": params["wk"][i], "wv": params["wv"][i],
            "wo": params["wo"][i], "w1": params["w1"][i], "w2": params["w2"][i],
            "w3": params["w3"][i],
        } for i in range(c.num_layers)]
        self.cos, self.sin = rope_tables(c.max_seq_len, c.head_size, device)
        self.gqa = c.kv_mul > 1

    def parameter_bytes(self):
        seen, total = set(), 0
        for w in [self.wcls] + [w for layer in self.layers
                                for w in layer.values() if isinstance(w, Quantized)]:
            if id(w) in seen:
                continue
            seen.add(id(w))
            total += w.q.numel() * w.q.element_size() + \
                w.scales.numel() * w.scales.element_size()
        plain = [self.token_embedding, self.rms_final]
        plain += [layer[name] for layer in self.layers
                  for name in ("rms_att_weight", "rms_ffn_weight")]
        for t in plain:
            total += t.numel() * t.element_size()
        return total

    def forward(self, tokens, last_only=False, cache=None):
        raise NotImplementedError(
            "runq.c has no recompute path: quantization is defined per row, so a "
            "prompt runs one token at a time. Use the KV cache.")

    def prefill(self, tokens, cache):
        cache.reset()
        logits = None
        for position, token_id in enumerate(int(t) for t in np.asarray(tokens).reshape(-1)):
            logits = self.decode(token_id, position, cache)
        return logits

    def decode(self, token_id, position, cache):
        c = self.config
        if position >= c.max_seq_len:
            raise ValueError(f"position {position} exceeds max_seq_len {c.max_seq_len}")
        if not 0 <= token_id < c.vocab_size:
            raise ValueError(f"invalid token id {token_id}")

        x = self.token_embedding[token_id].reshape(1, c.dim)
        for layer, w in enumerate(self.layers):
            x = self._decode_block(x, w, layer, position, cache)
        x = self._norm(x, self.rms_final)
        return self._qlinear(self.wcls, x)

    def _qlinear(self, weight, x):
        """runq.c quantizes once per distinct activation and shares it across the
        matmuls that read it: q/k/v off one (runq.c:367), w1/w3 off another
        (runq.c:450). This is for the three that have no partner."""
        xq, xs = quantize_groups(x, self.group_size)
        return qmatmul(weight, xq, xs)

    def _decode_block(self, x, w, layer, position, cache):
        c = self.config
        h = self._norm(x, w["rms_att_weight"])
        hq = quantize_groups(h, self.group_size)
        q = qmatmul(w["wq"], *hq).view(1, 1, c.num_heads, c.head_size).transpose(1, 2)
        k = qmatmul(w["wk"], *hq).view(1, 1, c.num_kv_heads, c.head_size).transpose(1, 2)
        v = qmatmul(w["wv"], *hq).view(1, 1, c.num_kv_heads, c.head_size).transpose(1, 2)

        cos_t = self.cos[position:position + 1]
        sin_t = self.sin[position:position + 1]
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)
        cache.append(layer, k, v)
        k, v = cache.view(layer)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=self.gqa)
        x = x + self._qlinear(w["wo"], attn.transpose(1, 2).reshape(1, c.dim))

        h2 = self._norm(x, w["rms_ffn_weight"])
        h2q = quantize_groups(h2, self.group_size)
        swiglu = F.silu(qmatmul(w["w1"], *h2q)) * qmatmul(w["w3"], *h2q)
        return x + self._qlinear(w["w2"], swiglu)
