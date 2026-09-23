"""Class numbers from transformers, for the Ruby side to match.

There is no C reference for this model, so the gate is agreement with
transformers (docs/plans/PLAN-conv2d.md): every image has to come out on the same class
number, not merely close. The true label is recorded beside it but is not the
gate. ResNet-18 gets about seven in ten right, and an implementation that
reproduced the reference's mistakes is what this is asking for.

  python/.venv/bin/python python/resnet_fixtures.py data/resnet-18 \
      python/fixtures/resnet-18_classes.json
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
from transformers import AutoModelForImageClassification

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from resnet_dump import WNIDS, COUNT, pixel_values  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_path")
    args = parser.parse_args()

    model = AutoModelForImageClassification.from_pretrained(args.model_dir).eval()
    with torch.inference_mode():
        logits = model(pixel_values()).logits
    probs = logits.softmax(-1)
    chosen = logits.argmax(-1).tolist()

    images = []
    for i, name in enumerate(WNIDS[:COUNT]):
        images.append({
            "image": name,
            "class": chosen[i],
            "label": model.config.id2label[chosen[i]],
            "probability": round(probs[i, chosen[i]].item(), 6),
        })

    out = {"model": "microsoft/resnet-18", "images": images}
    with open(args.out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    for row in images:
        print(f'  {row["image"]:28s} -> {row["class"]:4d} p={row["probability"]:.3f}  {row["label"][:34]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
