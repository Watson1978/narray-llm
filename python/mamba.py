"""Mamba over kroggen/mamba.c's version 1 checkpoint.

NumPy on the CPU, CuPy on the GPU with GPU=1, the same switch llama2.py uses.

Written out op for op from mamba.c rather than calling mamba_ssm, whose fused
CUDA kernels would make this a comparison of recipes instead of backends. The
other tables in this repository all hold the algorithm fixed and vary only what
runs it, and this one does the same.

  python/.venv/bin/python python/bench_mamba.py
  GPU=1 python/.venv/bin/python python/bench_mamba.py
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

import numpy as np  # host-side token ids, always NumPy (mirrors Ruby's HM)

MAGIC = 0x4D616D62  # "Mamb"
VERSION = 1
HEADER_BYTES = 256
HEADER_INTS = 8

TOKENIZER_MAGIC = 0x4D62546B  # "MbTk"
RMSNORM_EPS = 1e-5
EXP_LIMIT = 88.0
BOS_TOKEN = 0
EOS_TOKEN = 0


@dataclass(frozen=True)
class Config:
    num_layers: int
    vocab_size: int
    dim: int
    d_inner: int
    dt_rank: int
    d_state: int
    d_conv: int
    shared_classifier: bool

    @property
    def rounded_vocab_size(self) -> int:
        """mamba.c:187. The embedding and classifier are stored at this width."""
        remainder = self.vocab_size % 8
        return self.vocab_size if remainder == 0 else self.vocab_size + (8 - remainder)

    @property
    def x_proj_out(self) -> int:
        return self.dt_rank + 2 * self.d_state


def contiguous(a):
    return xp.ascontiguousarray(a)


def rmsnorm(x, weight, eps=RMSNORM_EPS):
    """mamba.c:234. The multiplications go x * weight * ss, left to right."""
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    return (x * weight) * (np.float32(1.0) / xp.sqrt(ms + eps))


def silu(x):
    """mamba.c:230. The exp argument is clipped so fp32 never reaches Inf."""
    return x * (np.float32(1.0) / (np.float32(1.0) + xp.exp(xp.clip(-x, -EXP_LIMIT, EXP_LIMIT))))


def softplus(x):
    """mamba.c:222. Past the clip exp is Inf, and log of that is the argument
    itself, so the excess is added back instead."""
    capped = xp.minimum(x, np.float32(EXP_LIMIT))
    return xp.log(np.float32(1.0) + xp.exp(capped)) + (x - capped)


def load_checkpoint(path: str) -> tuple[Config, dict]:
    with open(path, "rb") as f:
        head = f.read(HEADER_BYTES)
        if len(head) != HEADER_BYTES:
            raise ValueError(f"{path}: truncated header")
        magic, version = struct.unpack_from("<Ii", head, 0)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}, expected Mamb")
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, expected {VERSION}")
        ints = struct.unpack_from(f"<{HEADER_INTS}i", head, 8)
        config = Config(num_layers=ints[0], vocab_size=ints[1], dim=ints[2],
                        d_inner=ints[3], dt_rank=ints[4], d_state=ints[5],
                        d_conv=ints[6], shared_classifier=ints[7] == 1)

        l, d, di = config.num_layers, config.dim, config.d_inner
        rv = config.rounded_vocab_size
        # memory_map_weights (mamba.c:152), in the order the pointers walk.
        shapes = [
            ("embedding", (rv, d)),
            ("in_proj", (l, 2 * di, d)),
            ("conv1d_weight", (l, di, config.d_conv)),
            ("conv1d_bias", (l, di)),
            ("x_proj", (l, config.x_proj_out, di)),
            ("dt_proj_weight", (l, di, config.dt_rank)),
            ("dt_proj_bias", (l, di)),
            ("a", (l, di, config.d_state)),
            ("d", (l, di)),
            ("out_proj", (l, d, di)),
            ("norm", (l, d)),
            ("final_norm", (d,)),
        ]
        params: dict = {}
        for name, shape in shapes:
            count = int(np.prod(shape))
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading {name}")
            params[name] = xp.asarray(np.frombuffer(raw, dtype=np.float32).reshape(shape).copy())
        if config.shared_classifier:
            params["lm_head"] = params["embedding"]
        else:
            count = rv * d
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading lm_head")
            params["lm_head"] = xp.asarray(
                np.frombuffer(raw, dtype=np.float32).reshape(rv, d).copy())
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


def load_tokenizer(path: str) -> list[bytes]:
    """The table mamba.c's tokenizer.py writes. It carries its own count."""
    with open(path, "rb") as f:
        head = f.read(16)
        magic, version, vocab_size, _max_len = struct.unpack("<Iiii", head)
        if magic != TOKENIZER_MAGIC:
            raise ValueError(f"{path}: bad magic 0x{magic:08x}, expected MbTk")
        if version != VERSION:
            raise ValueError(f"{path}: version {version}, expected {VERSION}")
        pieces = []
        for i in range(vocab_size):
            (length,) = struct.unpack("<i", f.read(4))
            pieces.append(f.read(length))
    return pieces


def decode_text(pieces: list[bytes], ids) -> str:
    """mamba.c:570 decode plus mamba.c:583 safe_printf, over the whole run."""
    out = bytearray()
    for prev, token in zip(ids, ids[1:]):
        piece = pieces[int(token)]
        if int(prev) == EOS_TOKEN and piece.startswith(b" "):
            piece = piece[1:]
        if not piece:
            continue
        if len(piece) == 1 and not (0x20 <= piece[0] <= 0x7E or piece[0] in b"\t\n\v\f\r "):
            continue
        out += piece
    return out.decode("utf-8", errors="replace")


class State:
    """The two recurrences a Mamba block carries (mamba.c:64)."""

    def __init__(self, config: Config):
        self.config = config
        shape_conv = (config.num_layers, config.d_inner, config.d_conv)
        shape_ssm = (config.num_layers, config.d_inner, config.d_state)
        self.conv = xp.zeros(shape_conv, dtype=xp.float32)
        self.ssm = xp.zeros(shape_ssm, dtype=xp.float32)

    @staticmethod
    def bytes_for(config: Config) -> int:
        return 4 * config.num_layers * config.d_inner * (config.d_conv + config.d_state)

    def reset(self):
        self.conv[...] = 0.0
        self.ssm[...] = 0.0


class Model:
    def __init__(self, checkpoint_path):
        self.config, params = load_checkpoint(checkpoint_path)
        self.embedding = contiguous(params["embedding"])
        self.final_norm = contiguous(params["final_norm"])
        self.lm_head_t = contiguous(params["lm_head"].T)
        # Stored [out, in]; every matmul here wants [in, out].
        self.layers = [{
            "norm": contiguous(params["norm"][i]),
            "in_proj_t": contiguous(params["in_proj"][i].T),
            "conv1d_weight": contiguous(params["conv1d_weight"][i]),
            "conv1d_bias": contiguous(params["conv1d_bias"][i]),
            "x_proj_t": contiguous(params["x_proj"][i].T),
            "dt_proj_t": contiguous(params["dt_proj_weight"][i].T),
            "dt_proj_bias": contiguous(params["dt_proj_bias"][i]),
            "a": contiguous(params["a"][i]),
            "d": contiguous(params["d"][i]),
            "out_proj_t": contiguous(params["out_proj"][i].T),
        } for i in range(self.config.num_layers)]

    def parameter_bytes(self):
        seen, total = set(), 0
        tensors = [self.embedding, self.final_norm, self.lm_head_t]
        tensors += [t for layer in self.layers for t in layer.values()]
        for t in tensors:
            key = t.__array_interface__["data"][0] if not GPU else t.data.ptr
            if key in seen:
                continue
            seen.add(key)
            total += 4 * int(t.size)
        return total

    def new_state(self):
        return State(self.config)

    def prefill(self, tokens, state):
        state.reset()
        logits = None
        for token in np.asarray(tokens).reshape(-1):
            logits = self.decode(int(token), state)
        return logits

    def decode(self, token_id, state):
        """Answers [vocab_size]. mamba.c computes the rounded width because the
        table is stored that way, then samples over vocab_size."""
        c = self.config
        if not 0 <= token_id < c.rounded_vocab_size:
            raise ValueError(f"invalid token id {token_id}")

        x = self.embedding[token_id]
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, state)
        x = rmsnorm(x, self.final_norm)
        return (x @ self.lm_head_t)[:c.vocab_size]

    # mamba.c:377 forward_layer, then the residual mamba.c:481 folds in.
    def _block(self, x, w, layer, state):
        c = self.config
        di = c.d_inner
        h = rmsnorm(x, w["norm"])
        xz = h @ w["in_proj_t"]
        xc, z = xz[:di], xz[di:]

        conv = state.conv[layer]
        conv[:, :-1] = conv[:, 1:]
        conv[:, -1] = xc
        xc = silu(xp.sum(conv * w["conv1d_weight"], axis=1) + w["conv1d_bias"])

        x_db = xc @ w["x_proj_t"]
        dt = x_db[:c.dt_rank]
        b = x_db[c.dt_rank:c.dt_rank + c.d_state]
        cc = x_db[c.dt_rank + c.d_state:]

        dt = softplus(dt @ w["dt_proj_t"] + w["dt_proj_bias"]).reshape(di, 1)
        ssm = state.ssm[layer]
        ssm *= xp.exp(dt * w["a"])
        ssm += xc.reshape(di, 1) * (dt * b)

        y = xp.sum(ssm * cc, axis=1) + w["d"] * xc
        y = y * silu(z)
        return (y @ w["out_proj_t"]) + x


class Generator:
    def __init__(self, model, stop_token=EOS_TOKEN):
        self.model = model
        self.stop_token = stop_token

    def generate(self, prompt, max_new_tokens, stop_at_eot=True):
        tokens = [int(t) for t in prompt]
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        state = self.model.new_state()
        logits = self.model.prefill(tokens, state)
        produced = 0
        while produced < max_new_tokens:
            # The one readback per generated token.
            token = int(logits.argmax())
            tokens.append(token)
            produced += 1
            if stop_at_eot and token == self.stop_token:
                break
            if produced == max_new_tokens:
                break
            logits = self.model.decode(token, state)
        return tokens


def synchronize():
    if GPU:
        xp.cuda.Stream.null.synchronize()
