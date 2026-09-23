"""Mamba over kroggen/mamba.c's version 1 checkpoint, in PyTorch.

Only the backend differs from mamba.py. mamba_ssm is deliberately not used:
its fused CUDA kernels would make this a comparison of recipes rather than of
backends, and every other table in this repository holds the algorithm fixed.

Where PyTorch has a native op for what mamba.c spells out, it is used, the way
llama2_torch.py uses F.rms_norm and F.silu.

  python/.venv/bin/python python/bench_mamba.py --impl torch
  GPU=1 python/.venv/bin/python python/bench_mamba.py --impl torch
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

MAGIC = 0x4D616D62  # "Mamb"
VERSION = 1
HEADER_BYTES = 256
HEADER_INTS = 8

RMSNORM_EPS = 1e-5
BOS_TOKEN = 0
EOS_TOKEN = 0

from mamba import Config, load_tokenizer, decode_text  # noqa: E402,F401


def load_checkpoint(path: str, device) -> tuple[Config, dict]:
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
            host = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            params[name] = torch.from_numpy(host).to(device)
        if config.shared_classifier:
            params["lm_head"] = params["embedding"]
        else:
            count = rv * d
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading lm_head")
            host = np.frombuffer(raw, dtype=np.float32).reshape(rv, d).copy()
            params["lm_head"] = torch.from_numpy(host).to(device)
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


class State:
    def __init__(self, config: Config, device):
        self.config = config
        self.conv = torch.zeros((config.num_layers, config.d_inner, config.d_conv),
                                dtype=torch.float32, device=device)
        self.ssm = torch.zeros((config.num_layers, config.d_inner, config.d_state),
                               dtype=torch.float32, device=device)

    @staticmethod
    def bytes_for(config: Config) -> int:
        return 4 * config.num_layers * config.d_inner * (config.d_conv + config.d_state)

    def reset(self):
        self.conv.zero_()
        self.ssm.zero_()


class Model:
    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.config, params = load_checkpoint(checkpoint_path, device)
        self.embedding = params["embedding"].contiguous()
        self.final_norm = params["final_norm"].contiguous()
        self.lm_head_t = params["lm_head"].t().contiguous()
        self.layers = [{
            "norm": params["norm"][i].contiguous(),
            "in_proj_t": params["in_proj"][i].t().contiguous(),
            "conv1d_weight": params["conv1d_weight"][i].contiguous(),
            "conv1d_bias": params["conv1d_bias"][i].contiguous(),
            "x_proj_t": params["x_proj"][i].t().contiguous(),
            "dt_proj_t": params["dt_proj_weight"][i].t().contiguous(),
            "dt_proj_bias": params["dt_proj_bias"][i].contiguous(),
            "a": params["a"][i].contiguous(),
            "d": params["d"][i].contiguous(),
            "out_proj_t": params["out_proj"][i].t().contiguous(),
        } for i in range(self.config.num_layers)]

    def parameter_bytes(self):
        seen, total = set(), 0
        tensors = [self.embedding, self.final_norm, self.lm_head_t]
        tensors += [t for layer in self.layers for t in layer.values()]
        for t in tensors:
            key = t.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            total += t.numel() * t.element_size()
        return total

    def new_state(self):
        return State(self.config, self.device)

    def prefill(self, tokens, state):
        state.reset()
        logits = None
        for token in np.asarray(tokens).reshape(-1):
            logits = self.decode(int(token), state)
        return logits

    def decode(self, token_id, state):
        c = self.config
        if not 0 <= token_id < c.rounded_vocab_size:
            raise ValueError(f"invalid token id {token_id}")

        x = self.embedding[token_id]
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, state)
        x = F.rms_norm(x, (c.dim,), self.final_norm, RMSNORM_EPS)
        return (x @ self.lm_head_t)[:c.vocab_size]

    def _block(self, x, w, layer, state):
        c = self.config
        di = c.d_inner
        h = F.rms_norm(x, (c.dim,), w["norm"], RMSNORM_EPS)
        xz = h @ w["in_proj_t"]
        xc, z = xz[:di], xz[di:]

        conv = state.conv[layer]
        conv[:, :-1] = conv[:, 1:].clone()
        conv[:, -1] = xc
        xc = F.silu(torch.sum(conv * w["conv1d_weight"], dim=1) + w["conv1d_bias"])

        x_db = xc @ w["x_proj_t"]
        dt = x_db[:c.dt_rank]
        b = x_db[c.dt_rank:c.dt_rank + c.d_state]
        cc = x_db[c.dt_rank + c.d_state:]

        dt = F.softplus(dt @ w["dt_proj_t"] + w["dt_proj_bias"]).reshape(di, 1)
        ssm = state.ssm[layer]
        ssm *= torch.exp(dt * w["a"])
        ssm += xc.reshape(di, 1) * (dt * b)

        y = torch.sum(ssm * cc, dim=1) + w["d"] * xc
        y = y * F.silu(z)
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
        torch.cuda.synchronize()
