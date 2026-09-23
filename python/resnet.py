"""ResNet-18 over the safetensors microsoft/resnet-18 publishes.

NumPy on the CPU, CuPy on the GPU with GPU=1, the same switch the other ports
in this directory use.

Written out to match lib/narray_llm/models/resnet/, not to be idiomatic: the
same two convolution spellings, the same channels last layout, the same batch
norm folded into one scale and one shift at load. transformers is not called
here; it is the reference the fixtures came from, and using it would make this
a comparison of recipes rather than of backends.

cumo's third spelling has no counterpart here. CuPy 14.2 exposes no cuDNN
convolution, so on this side cuDNN is what the torch port runs.

  python/.venv/bin/python python/bench_resnet.py
  GPU=1 python/.venv/bin/python python/bench_resnet.py --spelling unfold
"""

from __future__ import annotations

import json
import os

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

import numpy as np
from safetensors.numpy import load_file

SPELLINGS = ("shift", "unfold")
STEM = "resnet.embedder.embedder"
CLASSIFIER = "classifier.1"
NORM_EPS = 1.0e-5
POOL = dict(kernel=3, stride=2, padding=1)
MASK_VALUE = -1.0e9


def to_device(host):
    return xp.asarray(np.ascontiguousarray(host))


def synchronize():
    if GPU:
        xp.cuda.runtime.deviceSynchronize()


class Checkpoint:
    def __init__(self, model_dir):
        self.tensors = load_file(os.path.join(model_dir, "model.safetensors"))
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        self.embedding_size = raw["embedding_size"]
        self.hidden_sizes = raw["hidden_sizes"]
        self.depths = raw["depths"]
        self.downsample_in_first_stage = raw["downsample_in_first_stage"]

    def __getitem__(self, name):
        return self.tensors[name]

    def stride_for(self, stage):
        return 1 if stage == 0 and not self.downsample_in_first_stage else 2

    def in_size(self, stage):
        return self.embedding_size if stage == 0 else self.hidden_sizes[stage - 1]

    def shortcut(self, stage):
        return self.in_size(stage) != self.hidden_sizes[stage] or self.stride_for(stage) != 1


class Conv2d:
    """Conv2d over [N, H, W, in]. torch stores the weight as [out, in, kh, kw]."""

    def __init__(self, weight, bias=None, stride=1, padding=0, spelling="shift"):
        if spelling not in SPELLINGS:
            raise ValueError(f"unknown spelling {spelling}")
        self.out_channels, self.in_channels, self.kernel, _ = weight.shape
        self.stride, self.padding, self.spelling = stride, padding, spelling
        self.bias = None if bias is None else to_device(bias.reshape(1, -1))
        device = to_device(weight)
        self.taps = [xp.ascontiguousarray(device[:, :, i, j].T)
                     for i in range(self.kernel) for j in range(self.kernel)]
        self.unfolded = xp.ascontiguousarray(
            device.transpose(2, 3, 1, 0).reshape(-1, self.out_channels))

    def out_size(self, size):
        return (size + 2 * self.padding - self.kernel) // self.stride + 1

    def __call__(self, x):
        batch, height, width, _ = x.shape
        out_h, out_w = self.out_size(height), self.out_size(width)
        source = x if self.padding == 0 else self.pad(x)
        rows = batch * out_h * out_w
        out = (self.unfold(source, batch, rows, out_h, out_w) if self.spelling == "unfold"
               else self.shift(source, rows, out_h, out_w))
        return out.reshape(batch, out_h, out_w, self.out_channels)

    def window(self, source, i, j, out_h, out_w):
        return source[:, i:i + out_h * self.stride:self.stride,
                      j:j + out_w * self.stride:self.stride, :]

    def shift(self, source, rows, out_h, out_w):
        out = (xp.zeros((rows, self.out_channels), xp.float32) if self.bias is None
               else xp.broadcast_to(self.bias, (rows, self.out_channels)).copy())
        for tap in range(self.kernel * self.kernel):
            i, j = divmod(tap, self.kernel)
            patch = xp.ascontiguousarray(self.window(source, i, j, out_h, out_w)) \
                      .reshape(rows, self.in_channels)
            out += patch @ self.taps[tap]
        return out

    def unfold(self, source, batch, rows, out_h, out_w):
        span = self.kernel * self.kernel * self.in_channels
        windows = xp.empty((batch, out_h, out_w, span), xp.float32)
        for tap in range(self.kernel * self.kernel):
            i, j = divmod(tap, self.kernel)
            at = tap * self.in_channels
            windows[:, :, :, at:at + self.in_channels] = self.window(source, i, j, out_h, out_w)
        y = windows.reshape(rows, span) @ self.unfolded
        return y if self.bias is None else y + self.bias

    def pad(self, x):
        p = self.padding
        return xp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))


def max_pool2d(x, kernel, stride, padding):
    batch, height, width, channels = x.shape
    out_h = (height + 2 * padding - kernel) // stride + 1
    out_w = (width + 2 * padding - kernel) // stride + 1
    source = x if padding == 0 else xp.pad(
        x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), constant_values=MASK_VALUE)
    best = None
    for i in range(kernel):
        for j in range(kernel):
            view = source[:, i:i + out_h * stride:stride, j:j + out_w * stride:stride, :]
            best = view if best is None else xp.maximum(best, view)
    return best


class Norm:
    def __init__(self, checkpoint, prefix):
        scale = checkpoint[f"{prefix}.normalization.weight"] / np.sqrt(
            checkpoint[f"{prefix}.normalization.running_var"] + NORM_EPS)
        shift = checkpoint[f"{prefix}.normalization.bias"] - \
            checkpoint[f"{prefix}.normalization.running_mean"] * scale
        self.scale, self.shift = to_device(scale), to_device(shift)
        self.host_scale, self.host_shift = scale, shift

    def __call__(self, x):
        return x * self.scale + self.shift


class ConvLayer:
    def __init__(self, checkpoint, prefix, stride, padding, spelling,
                 activation=True, fold=False):
        weight = checkpoint[f"{prefix}.convolution.weight"]
        norm = Norm(checkpoint, prefix)
        bias = None
        if fold:
            weight = weight * norm.host_scale.reshape(-1, 1, 1, 1)
            bias = norm.host_shift
        self.conv = Conv2d(weight, bias, stride, padding, spelling)
        self.norm = None if fold else norm
        self.activation = activation

    def __call__(self, x):
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
        return xp.maximum(y, 0.0) if self.activation else y


class BasicLayer:
    def __init__(self, checkpoint, base, stride, shortcut, spelling, fold):
        self.first = ConvLayer(checkpoint, f"{base}.layer.0", stride, 1, spelling, fold=fold)
        self.second = ConvLayer(checkpoint, f"{base}.layer.1", 1, 1, spelling,
                                activation=False, fold=fold)
        self.shortcut = ConvLayer(checkpoint, f"{base}.shortcut", stride, 0, spelling,
                                  activation=False, fold=fold) if shortcut else None

    def __call__(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return xp.maximum(self.second(self.first(x)) + residual, 0.0)


class Model:
    def __init__(self, model_dir, spelling="unfold", fold=False):
        checkpoint = Checkpoint(model_dir)
        self.stem = ConvLayer(checkpoint, STEM, 2, 3, spelling, fold=fold)
        self.layers = []
        for stage, depth in enumerate(checkpoint.depths):
            for layer in range(depth):
                self.layers.append(BasicLayer(
                    checkpoint, f"resnet.encoder.stages.{stage}.layers.{layer}",
                    checkpoint.stride_for(stage) if layer == 0 else 1,
                    layer == 0 and checkpoint.shortcut(stage), spelling, fold))
        self.classifier_t = to_device(checkpoint[f"{CLASSIFIER}.weight"].T)
        self.classifier_b = to_device(checkpoint[f"{CLASSIFIER}.bias"])

    # The caller transfers once, before anything is timed. The Ruby side
    # reads its input onto the device at load, so a per call transfer here
    # would be measuring a different thing.
    def prepare(self, pixels):
        return to_device(pixels)

    def forward(self, pixels):
        x = xp.ascontiguousarray(pixels.transpose(0, 2, 3, 1))
        x = max_pool2d(self.stem(x), **POOL)
        for layer in self.layers:
            x = layer(x)
        return x.mean(axis=(1, 2)) @ self.classifier_t + self.classifier_b

    def classify(self, pixels):
        return [int(v) for v in self.forward(pixels).argmax(axis=1)]
