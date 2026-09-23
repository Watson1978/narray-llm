"""Dumps what microsoft/resnet-18 computes, for the Ruby side.

There is no C reference for this model, so transformers is the reference
(docs/plans/PLAN-conv2d.md). The values are taken with forward hooks on the real model
rather than from a re-implementation, so nothing here can drift from what
transformers actually computes.

The preprocessing is written here rather than taken from AutoImageProcessor,
which wants torchvision. Installing that could move the torch every PyTorch
number in docs/results/ was measured against. The constants come from the
model's own preprocessor_config.json, so the transform is the published one
even though the code is not.

  python/.venv/bin/python python/resnet_dump.py data/resnet-18 \
      data/resnet-18_state.safetensors
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageClassification

IMAGES = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
CACHE = "data/resnet-18_images"
COUNT = 16

# preprocessor_config.json: size 224, crop_pct 0.875, resample 3 (bicubic).
SIZE = 224
RESIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# One per class from the 1000 the repository publishes, spread across the
# label space rather than taken from the front, which is all fish and birds.
WNIDS = [
    "n01440764_tench", "n01820546_lorikeet", "n02033041_dowitcher",
    "n02096585_Boston_bull", "n02123045_tabby", "n02325366_wood_rabbit",
    "n02749479_assault_rifle", "n03000684_chain_saw", "n03272010_electric_guitar",
    "n03633091_ladle", "n03976467_Polaroid_camera", "n04204347_shopping_cart",
    "n04398044_teapot", "n04550184_wardrobe", "n07747607_orange",
    "n09472597_volcano",
]


def fetch(name: str) -> str:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".JPEG")
    if not os.path.exists(path):
        urllib.request.urlretrieve(IMAGES + name + ".JPEG", path)
    return path


def preprocess(path: str) -> np.ndarray:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = RESIZE / min(w, h)
    im = im.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
    w, h = im.size
    left, top = (w - SIZE) // 2, (h - SIZE) // 2
    im = im.crop((left, top, left + SIZE, top + SIZE))
    pixels = np.asarray(im, np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)


def pixel_values() -> torch.Tensor:
    return torch.from_numpy(np.stack([preprocess(fetch(n)) for n in WNIDS[:COUNT]]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_path")
    args = parser.parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model_dir).eval()
    px = pixel_values()

    seen: dict[str, torch.Tensor] = {}
    watched = {
        "resnet.embedder.embedder.convolution": "conv1",
        "resnet.embedder": "embedder",
        # One 3x3 stride 1 and one 1x1 stride 2, so the other two kernel
        # shapes have a reference of their own. Their inputs are the
        # activations dumped beside them: embedder and stage.0.
        "resnet.encoder.stages.0.layers.0.layer.0.convolution": "conv3x3",
        "resnet.encoder.stages.1.layers.0.shortcut.convolution": "conv1x1",
        "resnet.encoder.stages.0": "stage.0",
        "resnet.encoder.stages.1": "stage.1",
        "resnet.encoder.stages.2": "stage.2",
        "resnet.encoder.stages.3": "stage.3",
        "resnet.pooler": "pooled",
    }

    def hook(label):
        def fn(_module, _inputs, output):
            seen[label] = (output[0] if isinstance(output, tuple) else output).detach().clone()
        return fn

    handles = [mod.register_forward_hook(hook(watched[name]))
               for name, mod in model.named_modules() if name in watched]
    with torch.inference_mode():
        logits = model(px).logits
    for h in handles:
        h.remove()

    out = {"pixel_values": px, "logits": logits.detach().clone()}
    out.update(seen)
    save_file({k: v.contiguous() for k, v in out.items()}, args.out_path)
    for name, tensor in out.items():
        print(f"{name:12s} {tuple(tensor.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
