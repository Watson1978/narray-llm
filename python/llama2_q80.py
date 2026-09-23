"""Llama 2 over llama2.c's version 2 (Q8_0) checkpoint, the one runq.c reads.

NumPy on the CPU, CuPy on the GPU with GPU=1, the same switch llama2.py uses.
Only the matmuls differ from llama2.py: the matrices are int8 with one fp32
scale per group, and the activation is quantized the same way before them.

  python/.venv/bin/python python/bench_llama2.py --model stories110M_q80
  GPU=1 python/.venv/bin/python python/bench_llama2.py --model stories110M_q80
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from llama2 import (GPU, BOS_TOKEN, Config, Generator, KVCache, contiguous,  # noqa: F401
                    decode_attention, decode_text, load_tokenizer, rope,
                    rope_tables, synchronize, xp)
import llama2 as _float

MAGIC = 0x616B3432  # "ak42"
VERSION = 2
HEADER_BYTES = 256

RMSNORM_EPS = 1e-5
EXP_LIMIT = 88.0
Q_MAX = np.float32(127.0)


@dataclass(frozen=True)
class Quantized:
    """q is [out, in / group_size, group_size]; scales is [out, in / group_size]."""

    q: "xp.ndarray"
    scales: "xp.ndarray"
    shape: tuple


def rmsnorm(x, weight, eps=RMSNORM_EPS):
    """runq.c:182. Takes the reciprocal once, where run.c divides."""
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    return weight * ((np.float32(1.0) / xp.sqrt(ms + eps)) * x)


def silu(x):
    """runq.c:346, again a reciprocal rather than a divide."""
    return x * (np.float32(1.0) / (np.float32(1.0) + xp.exp(xp.clip(-x, -EXP_LIMIT, EXP_LIMIT))))


def round_half_away(x):
    """C's roundf, which numpy's round is not: numpy sends halves to even.

    Adding a signed half first is the usual spelling, but it is wrong for the
    largest float below 0.5, where the addition itself rounds up. Multiplying
    the fraction by two is exact, so this agrees with roundf everywhere.
    """
    t = xp.trunc(x)
    return t + xp.trunc((x - t) * np.float32(2.0))


def quantize_groups(x, group_size):
    """runq.c:145. One scale per group of group_size, symmetric around zero."""
    rows = x.reshape(x.size // group_size, group_size)
    scale = xp.abs(rows).max(axis=1, keepdims=True) / Q_MAX
    # A group of exact zeros would divide by zero. Dividing it by one instead
    # gives the zeros it should quantize to.
    positive = xp.ceil(xp.clip(scale, 0.0, 1.0))
    return round_half_away(rows / (scale + (np.float32(1.0) - positive))), scale.reshape(-1)


def qmatmul(weight, xq, xs):
    """runq.c:317, folding each group's products in fp32 rather than int32.

    group_size * 127 * 127 is at most 1,032,256, inside the 2 ** 24 that fp32
    represents exactly, so no partial sum is ever rounded. einsum keeps the
    int8 side from being materialized as floats, which `w * xq` would do.
    """
    ival = xp.einsum("ogk,gk->og", weight.q, xq)
    return (ival * weight.scales * xs).sum(axis=1).reshape(1, weight.shape[0])


def load_checkpoint(path: str) -> tuple[Config, int, dict]:
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
            raw = f.read(np.dtype(dtype).itemsize * count)
            if len(raw) != np.dtype(dtype).itemsize * count:
                raise ValueError(f"{path}: truncated while reading {what}")
            return np.frombuffer(raw, dtype=dtype)

        for name, shape in (("rms_att_weight", (n_layers, dim)),
                            ("rms_ffn_weight", (n_layers, dim)),
                            ("rms_final_weight", (dim,))):
            params[name] = xp.asarray(read(int(np.prod(shape)), np.float32, name)
                                      .reshape(shape).copy())

        def read_one(shape, what):
            out, inner = shape
            if inner % group_size:
                raise ValueError(f"{what}: row of {inner} is not a multiple of {group_size}")
            groups = inner // group_size
            q = read(out * inner, np.int8, f"{what} (int8)").reshape(out, groups, group_size)
            s = read(out * groups, np.float32, f"{what} (scales)").reshape(out, groups)
            return Quantized(xp.asarray(q.copy()), xp.asarray(s.copy()), shape)

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
    return (weight.q.astype(xp.float32) *
            weight.scales.reshape(out, groups, 1)).reshape(weight.shape)


class Model(_float.Model):
    """runq.c's forward(). Decode only, one token at a time, as runq.c is."""

    def __init__(self, checkpoint_path):
        self.config, self.group_size, params = load_checkpoint(checkpoint_path)
        # runq.c:204 dequantizes the embedding once and then reads rows out of it.
        self.token_embedding = contiguous(dequantize(params["q_tokens"]))
        self.rms_final = contiguous(params["rms_final_weight"])
        self.wcls = params["wcls"]
        self.layers = [{
            "rms_att_weight": contiguous(params["rms_att_weight"][i]),
            "rms_ffn_weight": contiguous(params["rms_ffn_weight"][i]),
            "wq": params["wq"][i], "wk": params["wk"][i], "wv": params["wv"][i],
            "wo": params["wo"][i], "w1": params["w1"][i], "w2": params["w2"][i],
            "w3": params["w3"][i],
        } for i in range(self.config.num_layers)]
        self._full_mask = None
        self._rope = None

    def parameter_bytes(self):
        seen, total = set(), 0
        for w in [self.wcls] + [w for layer in self.layers
                                for w in layer.values() if isinstance(w, Quantized)]:
            if id(w) in seen:
                continue
            seen.add(id(w))
            total += int(w.q.size) + 4 * int(w.scales.size)
        plain = [self.token_embedding, self.rms_final]
        plain += [layer[name] for layer in self.layers
                  for name in ("rms_att_weight", "rms_ffn_weight")]
        return total + 4 * sum(int(t.size) for t in plain)

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
        cos_t, sin_t = self._rope_slice(position, 1)
        for layer, w in enumerate(self.layers):
            x = self._decode_block(x, w, layer, cos_t, sin_t, cache)
        x = rmsnorm(x, self.rms_final)
        return self._qlinear(self.wcls, x)

    def _qlinear(self, weight, x):
        """runq.c quantizes once per distinct activation and shares it across the
        matmuls that read it: q/k/v off one (runq.c:367), w1/w3 off another
        (runq.c:450). This is for the three that have no partner."""
        xq, xs = quantize_groups(x, self.group_size)
        return qmatmul(weight, xq, xs)

    def _decode_block(self, x, w, layer, cos_t, sin_t, cache):
        c = self.config
        h = rmsnorm(x, w["rms_att_weight"])
        hq = quantize_groups(h, self.group_size)
        q = rope(qmatmul(w["wq"], *hq), cos_t, sin_t, c.num_heads)
        k = rope(qmatmul(w["wk"], *hq), cos_t, sin_t, c.num_kv_heads)
        cache.append(layer, k, qmatmul(w["wv"], *hq))

        keys, values = cache.view(layer)
        attn = decode_attention(q, keys, values, c.num_heads, c.num_kv_heads)
        x = x + self._qlinear(w["wo"], attn)

        h2 = rmsnorm(x, w["rms_ffn_weight"])
        h2q = quantize_groups(h2, self.group_size)
        swiglu = silu(qmatmul(w["w1"], *h2q)) * qmatmul(w["w3"], *h2q)
        return x + self._qlinear(w["w2"], swiglu)
