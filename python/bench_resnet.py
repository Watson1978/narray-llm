"""One condition per process, for the ResNet-18 comparison.

  GPU=1 python/.venv/bin/python python/bench_resnet.py --spelling unfold
  GPU=1 python/.venv/bin/python python/bench_resnet.py --impl torch

The class numbers are checked against the fixture before anything is timed,
so a number never comes out of a run that is computing the wrong thing. One
pass over the sixteen images is a few milliseconds, far under the second the
measurement rules ask for, so the timed region repeats (AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "resnet-18")
STATE = os.path.join(DATA_DIR, "resnet-18_state.safetensors")
FIXTURE = os.path.join(ROOT, "python", "fixtures", "resnet-18_classes.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--spelling", choices=("shift", "unfold"), default="unfold")
    parser.add_argument("--fold", action="store_true")
    parser.add_argument("--inner", "--repeat", dest="repeat", type=int, default=100,
                        help="passes inside the region")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--no-tf32", action="store_true",
                        help="torch only: cuDNN keeps fp32 rather than TF32")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    from safetensors.numpy import load_file
    pixels = load_file(STATE)["pixel_values"]

    if args.impl == "torch":
        import torch

        import resnet_torch as impl
        # torch turns TF32 on for cuDNN by default on this card, so a
        # comparison against an fp32 backend has to say which one it measured.
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
        model = impl.Model(MODEL_DIR, fold=args.fold)
        backend = f"torch/{impl.DEVICE.type}"
        spelling = "cudnn-fp32" if args.no_tf32 else "cudnn-tf32"
    else:
        import resnet as impl
        model = impl.Model(MODEL_DIR, spelling=args.spelling, fold=args.fold)
        backend = impl.xp.__name__
        spelling = args.spelling

    pixels = model.prepare(pixels)

    if not args.no_check:
        want = [row["class"] for row in json.load(open(FIXTURE))["images"]]
        got = model.classify(pixels)
        if got != want:
            wrong = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
            print(f"{args.impl}/{spelling}: class numbers differ at {wrong}", file=sys.stderr)
            return 1

    forward = (lambda: model.forward(pixels)) if args.impl == "numpy" else None
    if forward is None:
        import torch

        def forward():
            with torch.inference_mode():
                model.forward(pixels)

    forward()
    forward()
    impl.synchronize()
    times = []
    for _ in range(args.rounds):
        started = time.perf_counter()
        for _ in range(args.repeat):
            forward()
        impl.synchronize()
        times.append(time.perf_counter() - started)

    best = min(times)
    images = int(pixels.shape[0]) * args.repeat
    print(f"{args.impl}\t{backend}\t{spelling}\t{'folded' if args.fold else 'plain'}\t"
          f"{best:.4f}\t{images / best:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
