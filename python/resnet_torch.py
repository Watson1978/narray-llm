"""ResNet-18 in torch, from the same safetensors the other ports read.

docs/method.md puts the torch column outside the correspondence table: it is
not a mirror of the Ruby structure but the ceiling the ecosystem reaches with
the same weights. So this calls conv2d and max_pool2d directly and lets cuDNN
pick whatever it picks, which is the point of having the column.

transformers is not used here. It is the reference the fixtures came from,
and calling it would make this a comparison of recipes.

  GPU=1 python/.venv/bin/python python/bench_resnet.py --impl torch
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
DEVICE = torch.device("cuda" if GPU and torch.cuda.is_available() else "cpu")

STEM = "resnet.embedder.embedder"
CLASSIFIER = "classifier.1"
NORM_EPS = 1.0e-5


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class ConvLayer:
    def __init__(self, tensors, prefix, stride, padding, activation=True, fold=False):
        weight = tensors[f"{prefix}.convolution.weight"]
        scale = tensors[f"{prefix}.normalization.weight"] / torch.sqrt(
            tensors[f"{prefix}.normalization.running_var"] + NORM_EPS)
        shift = tensors[f"{prefix}.normalization.bias"] - \
            tensors[f"{prefix}.normalization.running_mean"] * scale
        self.stride, self.padding, self.activation = stride, padding, activation
        if fold:
            self.weight = (weight * scale.reshape(-1, 1, 1, 1)).contiguous().to(DEVICE)
            self.bias = shift.contiguous().to(DEVICE)
            self.scale = None
        else:
            self.weight = weight.contiguous().to(DEVICE)
            self.bias = None
            self.scale = scale.reshape(1, -1, 1, 1).contiguous().to(DEVICE)
            self.shift = shift.reshape(1, -1, 1, 1).contiguous().to(DEVICE)

    def __call__(self, x):
        y = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
        if self.scale is not None:
            y = y * self.scale + self.shift
        return F.relu(y) if self.activation else y


class BasicLayer:
    def __init__(self, tensors, base, stride, shortcut, fold):
        self.first = ConvLayer(tensors, f"{base}.layer.0", stride, 1, fold=fold)
        self.second = ConvLayer(tensors, f"{base}.layer.1", 1, 1, activation=False, fold=fold)
        self.shortcut = ConvLayer(tensors, f"{base}.shortcut", stride, 0,
                                  activation=False, fold=fold) if shortcut else None

    def __call__(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        return F.relu(self.second(self.first(x)) + residual)


class Model:
    def __init__(self, model_dir, fold=False):
        tensors = load_file(os.path.join(model_dir, "model.safetensors"))
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        depths = config["depths"]
        hidden = config["hidden_sizes"]
        embedding = config["embedding_size"]
        first_stride = config["downsample_in_first_stage"]

        self.stem = ConvLayer(tensors, STEM, 2, 3, fold=fold)
        self.layers = []
        for stage, depth in enumerate(depths):
            stride = 1 if stage == 0 and not first_stride else 2
            inside = embedding if stage == 0 else hidden[stage - 1]
            shortcut = inside != hidden[stage] or stride != 1
            for layer in range(depth):
                self.layers.append(BasicLayer(
                    tensors, f"resnet.encoder.stages.{stage}.layers.{layer}",
                    stride if layer == 0 else 1, layer == 0 and shortcut, fold))
        self.classifier_w = tensors[f"{CLASSIFIER}.weight"].contiguous().to(DEVICE)
        self.classifier_b = tensors[f"{CLASSIFIER}.bias"].contiguous().to(DEVICE)

    # The caller transfers once, before anything is timed. See resnet.py.
    def prepare(self, pixels):
        return torch.as_tensor(pixels).to(DEVICE).contiguous()

    def forward(self, pixels):
        x = pixels
        x = F.max_pool2d(self.stem(x), kernel_size=3, stride=2, padding=1)
        for layer in self.layers:
            x = layer(x)
        return F.linear(x.mean(dim=(2, 3)), self.classifier_w, self.classifier_b)

    def classify(self, pixels):
        with torch.inference_mode():
            return [int(v) for v in self.forward(pixels).argmax(dim=1)]
