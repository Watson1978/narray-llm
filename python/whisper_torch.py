"""Whisper tiny in PyTorch, over the same safetensors.

Only the backend differs from whisper.py. transformers' own implementation is
deliberately not used: it would make this a comparison of recipes rather than
of backends, and every other table in this repository holds the algorithm
fixed. Where PyTorch has a native op for what the Ruby side spells out, it is
used, the way llama2_torch.py uses F.rms_norm.

  python/.venv/bin/python python/bench_whisper.py --impl torch
  GPU=1 python/.venv/bin/python python/bench_whisper.py --impl torch --spelling unfold
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from whisper import (EMBED, LAYER_NORM_EPS, MEL_DYNAMIC_RANGE, MEL_FLOOR, SPELLINGS)

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
DEVICE = torch.device("cuda" if GPU else "cpu")


def layernorm(x, weight, bias):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, LAYER_NORM_EPS)


def gelu_erf(x):
    # nn.functional.gelu is the erf formulation, which is the one Whisper was
    # trained with.
    return F.gelu(x)


class Conv1d:
    def __init__(self, weight, bias, stride=1, padding=1, spelling="shift", device=DEVICE):
        if spelling not in SPELLINGS:
            raise ValueError(f"unknown spelling {spelling}")
        self.out_channels, self.in_channels, self.kernel = weight.shape
        self.stride, self.padding, self.spelling = stride, padding, spelling
        self.device = device
        self.bias = None if bias is None else bias.reshape(1, -1).to(device)
        self.taps = [weight[:, :, tap].T.contiguous().to(device) for tap in range(self.kernel)]
        self.flat = torch.vstack(self.taps).contiguous()

    def out_frames(self, frames):
        return (frames + 2 * self.padding - self.kernel) // self.stride + 1

    def __call__(self, x):
        frames = x.shape[0]
        if self.spelling == "unfold":
            window = torch.zeros((self.out_frames(frames), self.in_channels * self.kernel),
                                 device=self.device)
            for tap in range(self.kernel):
                span = self.tap_span(frames, tap)
                if span is None:
                    continue
                source, first_t, last_t = span
                window[first_t:last_t + 1,
                       tap * self.in_channels:(tap + 1) * self.in_channels] = x[source]
            product = window @ self.flat
            return product if self.bias is None else product + self.bias

        out = torch.zeros((self.out_frames(frames), self.out_channels), device=self.device)
        if self.bias is not None:
            out = out + self.bias
        for tap in range(self.kernel):
            span = self.tap_span(frames, tap)
            if span is None:
                continue
            source, first_t, last_t = span
            out[first_t:last_t + 1] = out[first_t:last_t + 1] + x[source] @ self.taps[tap]
        return out

    def tap_span(self, frames, tap):
        last = self.out_frames(frames) - 1
        first_t = max(0, -((tap - self.padding) // self.stride))
        last_t = min(last, (frames - 1 - tap + self.padding) // self.stride)
        if first_t > last_t:
            return None
        first = first_t * self.stride + tap - self.padding
        final = last_t * self.stride + tap - self.padding
        return slice(first, final + 1, self.stride), first_t, last_t


class Mel:
    def __init__(self, preprocessor, device=DEVICE):
        self.n_fft = preprocessor["n_fft"]
        self.hop_length = preprocessor["hop_length"]
        self.n_samples = preprocessor["n_samples"]
        self.frames = preprocessor["nb_max_frames"]
        self.mel_bins = preprocessor["feature_size"]
        self.bins = self.n_fft // 2 + 1
        self.device = device
        self.filters = torch.tensor(preprocessor["mel_filters"], device=device)
        n = np.arange(self.n_fft, dtype=np.float32)
        window = (0.5 - 0.5 * np.cos(n * (2.0 * np.pi / self.n_fft))).astype(np.float32)
        self.window = torch.from_numpy(window).reshape(1, -1).to(device)
        angle = np.outer(np.arange(self.n_fft), np.arange(self.bins)) * (-2.0 * np.pi / self.n_fft)
        self.cos = torch.from_numpy(np.cos(angle).astype(np.float32)).to(device)
        self.sin = torch.from_numpy(np.sin(angle).astype(np.float32)).to(device)

    def __call__(self, waveform):
        windows = self.frame(self.reflect(self.fit(waveform)))
        real = windows @ self.cos
        imag = windows @ self.sin
        power = real * real + imag * imag
        return self.logarithm(self.filters @ power.T.contiguous())

    def fit(self, waveform):
        samples = waveform.shape[0]
        if samples > self.n_samples:
            return waveform[:self.n_samples].contiguous()
        if samples == self.n_samples:
            return waveform.contiguous()
        out = torch.zeros(self.n_samples, device=self.device)
        out[:samples] = waveform
        return out

    def reflect(self, signal):
        pad = self.n_fft // 2
        out = torch.empty(self.n_samples + 2 * pad, device=self.device)
        out[pad:pad + self.n_samples] = signal
        out[:pad] = signal[1:pad + 1].flip(0)
        out[pad + self.n_samples:] = signal[-pad - 1:-1].flip(0)
        return out

    def frame(self, padded):
        blocks = self.n_fft // self.hop_length
        rest = self.n_fft % self.hop_length
        needed = self.frames + blocks + (0 if rest == 0 else 1) - 1
        grid = padded[:needed * self.hop_length].contiguous().reshape(needed, self.hop_length)
        out = torch.empty((self.frames, self.n_fft), device=self.device)
        for b in range(blocks):
            out[:, b * self.hop_length:(b + 1) * self.hop_length] = grid[b:b + self.frames]
        if rest:
            out[:, blocks * self.hop_length:] = grid[blocks:blocks + self.frames, :rest]
        return out * self.window

    def logarithm(self, mel):
        spec = torch.log10(torch.clamp(mel, min=MEL_FLOOR))
        floor = float(spec.max()) - MEL_DYNAMIC_RANGE
        return (torch.clamp(spec, min=floor) + 4.0) / 4.0


class Stack:
    def __init__(self, config, heads, device):
        self.config = config
        self.heads = heads
        self.head_dim = config["d_model"] // heads
        self.scaling = self.head_dim ** -0.5
        self.device = device

    def heads_major(self, x):
        return x.reshape(x.shape[0], self.heads, self.head_dim).permute(1, 0, 2).contiguous()

    def linear(self, x, w):
        y = x @ w["weight"]
        return y if w["bias"] is None else y + w["bias"]

    def attend(self, h, w, prefix, keys, values):
        q = self.heads_major(self.linear(h, w[prefix + "q"]) * self.scaling)
        parts = [torch.softmax(q[i] @ keys[i].T, dim=-1) @ values[i] for i in range(self.heads)]
        return self.linear(torch.hstack(parts).contiguous(), w[prefix + "out"])

    def linear_weights(self, store, name):
        bias = store.get(f"{name}.bias")
        return {"weight": store[f"{name}.weight"].T.contiguous().to(self.device),
                "bias": None if bias is None else bias.to(self.device)}

    def norm_weights(self, store, name):
        return {"weight": store[f"{name}.weight"].to(self.device),
                "bias": store[f"{name}.bias"].to(self.device)}


class Encoder(Stack):
    def __init__(self, store, config, spelling, device):
        super().__init__(config, config["encoder_attention_heads"], device)
        self.conv1 = Conv1d(store["model.encoder.conv1.weight"], store["model.encoder.conv1.bias"],
                            stride=1, padding=1, spelling=spelling, device=device)
        self.conv2 = Conv1d(store["model.encoder.conv2.weight"], store["model.encoder.conv2.bias"],
                            stride=2, padding=1, spelling=spelling, device=device)
        self.positions = store["model.encoder.embed_positions.weight"].to(device)
        self.final_norm = self.norm_weights(store, "model.encoder.layer_norm")
        self.layers = [self.layer_weights(store, i) for i in range(config["encoder_layers"])]

    def __call__(self, mel):
        x = mel.T.contiguous()
        x = gelu_erf(self.conv1(x))
        x = gelu_erf(self.conv2(x))
        x = x + self.positions
        for w in self.layers:
            h = layernorm(x, w["attention_norm"]["weight"], w["attention_norm"]["bias"])
            keys = self.heads_major(self.linear(h, w["k"]))
            values = self.heads_major(self.linear(h, w["v"]))
            x = x + self.attend(h, w, "", keys, values)
            h = layernorm(x, w["ff_norm"]["weight"], w["ff_norm"]["bias"])
            x = x + self.linear(gelu_erf(self.linear(h, w["fc1"])), w["fc2"])
        return layernorm(x, self.final_norm["weight"], self.final_norm["bias"])

    def layer_weights(self, store, index):
        at = f"model.encoder.layers.{index}"
        w = {"attention_norm": self.norm_weights(store, f"{at}.self_attn_layer_norm"),
             "ff_norm": self.norm_weights(store, f"{at}.final_layer_norm"),
             "fc1": self.linear_weights(store, f"{at}.fc1"),
             "fc2": self.linear_weights(store, f"{at}.fc2")}
        for part in ("q", "k", "v", "out"):
            w[part] = self.linear_weights(store, f"{at}.self_attn.{part}_proj")
        return w


class Decoder(Stack):
    def __init__(self, store, config, device):
        super().__init__(config, config["decoder_attention_heads"], device)
        self.embedding = store[EMBED].to(device)
        self.positions = store["model.decoder.embed_positions.weight"].to(device)
        self.final_norm = self.norm_weights(store, "model.decoder.layer_norm")
        self.layers = [self.layer_weights(store, i) for i in range(config["decoder_layers"])]

    def new_cache(self, encoder_states):
        width = self.config["d_model"]
        limit = self.config["max_target_positions"]
        cross = [(self.heads_major(self.linear(encoder_states, w["cross_k"])),
                  self.heads_major(self.linear(encoder_states, w["cross_v"])))
                 for w in self.layers]
        return {"cross": cross,
                "keys": [torch.zeros((limit, width), device=self.device) for _ in self.layers],
                "values": [torch.zeros((limit, width), device=self.device) for _ in self.layers],
                "length": 0}

    def __call__(self, token_id, cache):
        position = cache["length"]
        ids = torch.tensor([token_id], device=self.device)
        x = self.embedding[ids].contiguous() + self.positions[position:position + 1]
        for i, w in enumerate(self.layers):
            h = layernorm(x, w["attention_norm"]["weight"], w["attention_norm"]["bias"])
            cache["keys"][i][position] = self.linear(h, w["k"])[0]
            cache["values"][i][position] = self.linear(h, w["v"])[0]
            span = slice(0, position + 1)
            x = x + self.attend(h, w, "", self.heads_major(cache["keys"][i][span]),
                                self.heads_major(cache["values"][i][span]))
            h = layernorm(x, w["cross_norm"]["weight"], w["cross_norm"]["bias"])
            keys, values = cache["cross"][i]
            x = x + self.attend(h, w, "cross_", keys, values)
            h = layernorm(x, w["ff_norm"]["weight"], w["ff_norm"]["bias"])
            x = x + self.linear(gelu_erf(self.linear(h, w["fc1"])), w["fc2"])
        cache["length"] = position + 1
        return layernorm(x, self.final_norm["weight"], self.final_norm["bias"])

    def layer_weights(self, store, index):
        at = f"model.decoder.layers.{index}"
        w = {"attention_norm": self.norm_weights(store, f"{at}.self_attn_layer_norm"),
             "cross_norm": self.norm_weights(store, f"{at}.encoder_attn_layer_norm"),
             "ff_norm": self.norm_weights(store, f"{at}.final_layer_norm"),
             "fc1": self.linear_weights(store, f"{at}.fc1"),
             "fc2": self.linear_weights(store, f"{at}.fc2")}
        for kind, prefix in (("self_attn", ""), ("encoder_attn", "cross_")):
            for part in ("q", "k", "v", "out"):
                w[prefix + part] = self.linear_weights(store, f"{at}.{kind}.{part}_proj")
        return w


class Model:
    def __init__(self, model_dir, spelling="shift", device=DEVICE):
        with open(f"{model_dir}/config.json") as f:
            self.config = json.load(f)
        with open(f"{model_dir}/preprocessor_config.json") as f:
            self.mel = Mel(json.load(f), device=device)
        store = load_file(f"{model_dir}/model.safetensors")
        self.encoder = Encoder(store, self.config, spelling, device)
        self.decoder = Decoder(store, self.config, device)
        self.classifier = store[EMBED].T.contiguous().to(device)
        self.device = device

    def encode(self, mel):
        return self.encoder(mel)

    def decode(self, token_id, cache):
        return self.decoder(token_id, cache) @ self.classifier

    def generate(self, mel, prompt, max_new_tokens, suppress=(), begin_suppress=()):
        cache = self.decoder.new_cache(self.encode(mel))
        logits = None
        for token in prompt:
            logits = self.decode(int(token), cache)
        suppress = list(suppress)
        begin = list(begin_suppress)
        out = []
        for step in range(max_new_tokens):
            row = logits[0].clone()
            hidden = suppress + begin if step == 0 else suppress
            if hidden:
                row[torch.tensor(hidden, device=self.device)] = float("-inf")
            token = int(row.argmax())
            out.append(token)
            if token == self.config["eos_token_id"] or len(out) == max_new_tokens:
                break
            logits = self.decode(token, cache)
        return out


def synchronize():
    if GPU:
        torch.cuda.synchronize()
