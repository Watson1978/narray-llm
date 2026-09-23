"""One condition per process, for the Whisper comparison.

AGENTS.md wants one process per condition and best-of-N inside it, so this
takes the condition on the command line and prints a single line. An
interleaving harness runs it once per (implementation, spelling, condition).

A single encode is 6 ms, far under the second AGENTS.md's rule 7 says a
condition needs before its spread settles, so --inner loops the work inside
one timed interval.

  python/.venv/bin/python python/bench_whisper.py --mode decode
  GPU=1 python/.venv/bin/python python/bench_whisper.py --impl torch --mode encode
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(DATA_DIR, "whisper-tiny")
DUMP = os.path.join(DATA_DIR, "whisper-tiny_encoder_state.safetensors")
FIXTURE = os.path.join(ROOT, "python", "fixtures", "whisper-tiny_greedy.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--spelling", choices=("shift", "unfold"), default="shift")
    parser.add_argument("--mode", choices=("mel", "encode", "decode"), default="decode")
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--inner", type=int, default=1)
    parser.add_argument("--rounds", "--repeat", dest="repeat", type=int, default=3)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    impl = importlib.import_module("whisper_torch" if args.impl == "torch" else "whisper")
    if args.impl == "torch":
        import torch
        from safetensors.torch import load_file
        context = torch.inference_mode
        to_device = lambda a: a.to(impl.DEVICE)  # noqa: E731
    else:
        import contextlib
        from safetensors.numpy import load_file
        context = contextlib.nullcontext
        to_device = impl.to_device

    model = impl.Model(MODEL_DIR, spelling=args.spelling)
    store = load_file(DUMP)
    mel = to_device(store["input_features"])
    waveform = to_device(store["waveform"])
    fixture = json.loads(pathlib.Path(FIXTURE).read_text())

    with context():
        if not args.no_check and not check(model, mel, fixture, args):
            return 1

        if args.mode == "mel":
            run = lambda: model.mel(waveform)  # noqa: E731
            produced, unit = model.mel.frames, "frames"
        elif args.mode == "encode":
            run = lambda: model.encode(mel)  # noqa: E731
            produced, unit = model.config["max_source_positions"], "positions"
        else:
            run = lambda: model.generate(  # noqa: E731
                mel, fixture["prompt"], args.length,
                suppress=fixture["suppress_tokens"],
                begin_suppress=fixture["begin_suppress_tokens"])
            produced, unit = args.length, "tokens"

        run()
        impl.synchronize()
        times = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            for _ in range(args.inner):
                run()
            impl.synchronize()
            times.append((time.perf_counter() - started) / args.inner)

    best = min(times)
    backend = f"torch/{'cuda' if impl.GPU else 'cpu'}" if args.impl == "torch" else impl.xp.__name__
    print(f"{args.impl}\t{backend}\t{args.spelling}\t{args.mode}\tlen={produced}\t"
          f"{produced / best:.2f}\t{best:.6f}\t{unit}")
    return 0


def check(model, mel, fixture, args) -> bool:
    """The token sequence has to be the one transformers produces."""
    got = model.generate(mel, fixture["prompt"], fixture["max_new_tokens"],
                         suppress=fixture["suppress_tokens"],
                         begin_suppress=fixture["begin_suppress_tokens"])
    if got != fixture["tokens"]:
        first = next((i for i, (a, b) in enumerate(zip(got, fixture["tokens"])) if a != b), len(got))
        print(f"{args.impl}/{args.spelling}: token sequence diverged at {first}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
