"""Whisper tiny over the safetensors openai/whisper-tiny publishes.

NumPy on the CPU, CuPy on the GPU with GPU=1, the same switch the other ports
in this directory use.

Written out to match lib/narray_llm/models/whisper/, not to be idiomatic: the
same two convolution spellings, the same DFT as two real matmuls, the same
per head loop. transformers is not called here; it is the reference the
fixtures came from, and using it would make this a comparison of recipes
rather than of backends.

  python/.venv/bin/python python/bench_whisper.py
  GPU=1 python/.venv/bin/python python/bench_whisper.py --spelling unfold
"""

from __future__ import annotations

import json
import math
import os

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

import numpy as np
from safetensors.numpy import load_file

EMBED = "model.decoder.embed_tokens.weight"
SPELLINGS = ("shift", "unfold")
LAYER_NORM_EPS = 1.0e-5
MEL_FLOOR = 1.0e-10
MEL_DYNAMIC_RANGE = 8.0


def to_device(host):
    return xp.asarray(np.ascontiguousarray(host))


def transposed(host):
    """Stored [out, in]; every matmul here wants [in, out]."""
    return xp.asarray(np.ascontiguousarray(host.T))


def layernorm(x, weight, bias):
    mean = xp.mean(x, axis=-1, keepdims=True)
    centred = x - mean
    variance = xp.mean(centred * centred, axis=-1, keepdims=True)
    return centred / xp.sqrt(variance + np.float32(LAYER_NORM_EPS)) * weight + bias


if GPU:
    from cupyx.scipy.special import erf as _erf
else:
    from scipy.special import erf as _erf

SQRT1_2 = np.float32(1.0 / math.sqrt(2.0))


def gelu_erf(x):
    """The erf formulation, which is what Whisper was trained with. numpy has
    no erf of its own, so scipy provides it on the host and cupyx on the
    device; both call the same function the Ruby side gets from XF::Math."""
    return np.float32(0.5) * x * (np.float32(1.0) + _erf(x * SQRT1_2))


def softmax_rows(x):
    shifted = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(shifted)
    return e / xp.sum(e, axis=-1, keepdims=True)


class Conv1d:
    """Conv1d over [frames, channels]. torch stores [out, in, kernel]."""

    def __init__(self, weight, bias, stride=1, padding=1, spelling="shift"):
        if spelling not in SPELLINGS:
            raise ValueError(f"unknown spelling {spelling}")
        self.out_channels, self.in_channels, self.kernel = weight.shape
        self.stride, self.padding, self.spelling = stride, padding, spelling
        self.bias = None if bias is None else to_device(bias.reshape(1, -1))
        self.taps = [transposed(weight[:, :, tap]) for tap in range(self.kernel)]
        self.flat = xp.ascontiguousarray(xp.vstack(self.taps))

    def out_frames(self, frames):
        return (frames + 2 * self.padding - self.kernel) // self.stride + 1

    def __call__(self, x):
        frames = x.shape[0]
        if self.spelling == "unfold":
            window = xp.zeros((self.out_frames(frames), self.in_channels * self.kernel),
                              dtype=xp.float32)
            for tap in range(self.kernel):
                span = self.tap_span(frames, tap)
                if span is None:
                    continue
                source, first_t, last_t = span
                window[first_t:last_t + 1,
                       tap * self.in_channels:(tap + 1) * self.in_channels] = x[source]
            product = window @ self.flat
            return product if self.bias is None else product + self.bias

        out = xp.zeros((self.out_frames(frames), self.out_channels), dtype=xp.float32)
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
    """The log-mel front end, with the DFT as two real matmuls."""

    def __init__(self, preprocessor):
        self.n_fft = preprocessor["n_fft"]
        self.hop_length = preprocessor["hop_length"]
        self.n_samples = preprocessor["n_samples"]
        self.frames = preprocessor["nb_max_frames"]
        self.mel_bins = preprocessor["feature_size"]
        self.bins = self.n_fft // 2 + 1
        self.filters = to_device(np.asarray(preprocessor["mel_filters"], dtype=np.float32))
        n = np.arange(self.n_fft, dtype=np.float32)
        self.window = to_device((0.5 - 0.5 * np.cos(n * (2.0 * np.pi / self.n_fft))
                                 ).astype(np.float32).reshape(1, -1))
        angle = np.outer(np.arange(self.n_fft), np.arange(self.bins)) * (-2.0 * np.pi / self.n_fft)
        self.cos = to_device(np.cos(angle).astype(np.float32))
        self.sin = to_device(np.sin(angle).astype(np.float32))

    def __call__(self, waveform):
        signal = self.fit(waveform)
        windows = self.frame(self.reflect(signal))
        real = windows @ self.cos
        imag = windows @ self.sin
        power = real * real + imag * imag
        return self.logarithm(self.filters @ xp.ascontiguousarray(power.T))

    def fit(self, waveform):
        samples = waveform.shape[0]
        if samples > self.n_samples:
            return xp.ascontiguousarray(waveform[:self.n_samples])
        if samples == self.n_samples:
            return xp.ascontiguousarray(waveform)
        out = xp.zeros(self.n_samples, dtype=xp.float32)
        out[:samples] = waveform
        return out

    def reflect(self, signal):
        pad = self.n_fft // 2
        out = xp.empty(self.n_samples + 2 * pad, dtype=xp.float32)
        out[pad:pad + self.n_samples] = signal
        out[:pad] = signal[pad:0:-1]
        out[pad + self.n_samples:] = signal[-2:-2 - pad:-1]
        return out

    def frame(self, padded):
        blocks = self.n_fft // self.hop_length
        rest = self.n_fft % self.hop_length
        needed = self.frames + blocks + (0 if rest == 0 else 1) - 1
        grid = xp.ascontiguousarray(padded[:needed * self.hop_length]).reshape(needed, self.hop_length)
        out = xp.empty((self.frames, self.n_fft), dtype=xp.float32)
        for b in range(blocks):
            out[:, b * self.hop_length:(b + 1) * self.hop_length] = grid[b:b + self.frames]
        if rest:
            out[:, blocks * self.hop_length:] = grid[blocks:blocks + self.frames, :rest]
        return out * self.window

    def logarithm(self, mel):
        spec = xp.log10(xp.maximum(mel, np.float32(MEL_FLOOR)))
        floor = float(spec.max()) - MEL_DYNAMIC_RANGE
        return (xp.maximum(spec, np.float32(floor)) + np.float32(4.0)) / np.float32(4.0)


class Stack:
    def __init__(self, store, config, heads):
        self.config = config
        self.heads = heads
        self.head_dim = config["d_model"] // heads
        self.scaling = np.float32(self.head_dim ** -0.5)

    def heads_major(self, x):
        return xp.ascontiguousarray(
            x.reshape(x.shape[0], self.heads, self.head_dim).transpose(1, 0, 2))

    def linear(self, x, w):
        y = x @ w["weight"]
        return y if w["bias"] is None else y + w["bias"]

    def attend(self, h, w, prefix, keys, values):
        q = self.heads_major(self.linear(h, w[prefix + "q"]) * self.scaling)
        parts = [softmax_rows(q[i] @ keys[i].T) @ values[i] for i in range(self.heads)]
        return self.linear(xp.ascontiguousarray(xp.hstack(parts)), w[prefix + "out"])

    @staticmethod
    def linear_weights(store, name):
        bias = store.get(f"{name}.bias")
        return {"weight": transposed(store[f"{name}.weight"]),
                "bias": None if bias is None else to_device(bias)}

    @staticmethod
    def norm_weights(store, name):
        return {"weight": to_device(store[f"{name}.weight"]),
                "bias": to_device(store[f"{name}.bias"])}


class Encoder(Stack):
    def __init__(self, store, config, spelling):
        super().__init__(store, config, config["encoder_attention_heads"])
        self.conv1 = Conv1d(store["model.encoder.conv1.weight"], store["model.encoder.conv1.bias"],
                            stride=1, padding=1, spelling=spelling)
        self.conv2 = Conv1d(store["model.encoder.conv2.weight"], store["model.encoder.conv2.bias"],
                            stride=2, padding=1, spelling=spelling)
        self.positions = to_device(store["model.encoder.embed_positions.weight"])
        self.final_norm = self.norm_weights(store, "model.encoder.layer_norm")
        self.layers = [self.layer_weights(store, i) for i in range(config["encoder_layers"])]

    def __call__(self, mel):
        x = xp.ascontiguousarray(mel.T)
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
    def __init__(self, store, config):
        super().__init__(store, config, config["decoder_attention_heads"])
        self.embedding = to_device(store[EMBED])
        self.positions = to_device(store["model.decoder.embed_positions.weight"])
        self.final_norm = self.norm_weights(store, "model.decoder.layer_norm")
        self.layers = [self.layer_weights(store, i) for i in range(config["decoder_layers"])]

    def new_cache(self, encoder_states):
        width = self.config["d_model"]
        limit = self.config["max_target_positions"]
        cross = [(self.heads_major(self.linear(encoder_states, w["cross_k"])),
                  self.heads_major(self.linear(encoder_states, w["cross_v"])))
                 for w in self.layers]
        return {"cross": cross,
                "keys": [xp.zeros((limit, width), dtype=xp.float32) for _ in self.layers],
                "values": [xp.zeros((limit, width), dtype=xp.float32) for _ in self.layers],
                "length": 0}

    def __call__(self, token_id, cache):
        position = cache["length"]
        x = xp.ascontiguousarray(self.embedding[np.asarray([token_id])]) \
            + self.positions[position:position + 1]
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
    def __init__(self, model_dir, spelling="shift"):
        with open(f"{model_dir}/config.json") as f:
            self.config = json.load(f)
        with open(f"{model_dir}/preprocessor_config.json") as f:
            self.mel = Mel(json.load(f))
        store = load_file(f"{model_dir}/model.safetensors")
        self.encoder = Encoder(store, self.config, spelling)
        self.decoder = Decoder(store, self.config)
        self.classifier = transposed(store[EMBED])

    def encode(self, mel):
        return self.encoder(mel)

    def decode(self, token_id, cache):
        return self.decoder(token_id, cache) @ self.classifier

    def generate(self, mel, prompt, max_new_tokens, suppress=(), begin_suppress=()):
        cache = self.decoder.new_cache(self.encode(mel))
        logits = None
        for token in prompt:
            logits = self.decode(int(token), cache)
        suppress = np.asarray(list(suppress), dtype=np.int64)
        begin = np.asarray(list(begin_suppress), dtype=np.int64)
        out = []
        for step in range(max_new_tokens):
            row = logits[0].copy()
            hidden = np.concatenate([suppress, begin]) if step == 0 else suppress
            if hidden.size:
                row[xp.asarray(hidden) if GPU else hidden] = -np.inf
            token = int(xp.argmax(row))
            out.append(token)
            if token == self.config["eos_token_id"] or len(out) == max_new_tokens:
                break
            logits = self.decode(token, cache)
        return out


def synchronize():
    if GPU:
        xp.cuda.Stream.null.synchronize()
