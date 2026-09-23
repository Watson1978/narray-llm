"""One condition per process, for the Mamba comparison.

CLAUDE.md wants one process per condition and best-of-N inside it, so this
takes the condition on the command line and prints a single line. An
interleaving harness runs it once per (implementation, condition).

  python/.venv/bin/python python/bench_mamba.py --length 64
  GPU=1 python/.venv/bin/python python/bench_mamba.py --impl torch --length 256

--impl numpy selects python/mamba.py (NumPy, or CuPy with GPU=1); --impl torch
selects python/mamba_torch.py. The generated tokens are checked against the
fixture before anything is timed, so a run that drifted cannot report a number.

There is no --cache switch. Mamba has no recompute path to compare against:
its two recurrences are the only way it advances, and mamba.c has no other
mode either.
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

DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TOKENIZER = "mamba_tokenizer.bin"
BOS = 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--model", default="mamba-130m")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--rounds", "--repeat", dest="repeat", type=int, default=3)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.impl == "torch":
        import torch
        impl = importlib.import_module("mamba_torch")
        context = torch.inference_mode
    else:
        impl = importlib.import_module("mamba")
        import contextlib
        context = contextlib.nullcontext

    checkpoint = os.path.join(DATA_DIR, f"{args.model}.bin")
    model = impl.Model(checkpoint)
    generator = impl.Generator(model)

    with context():
        # mamba.c does not stop at the delimiter, so neither does this.
        tokens = generator.generate([BOS], args.length, stop_at_eot=False)
        if not args.no_check and not check(args.model, tokens):
            return 1

        generator.generate([BOS], 2, stop_at_eot=False)
        impl.synchronize()
        times = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            generator.generate([BOS], args.length, stop_at_eot=False)
            impl.synchronize()
            times.append(time.perf_counter() - started)

    best = min(times)
    generated = len(tokens) - 1
    backend = describe(args.impl, impl)
    print(f"{args.impl}\t{backend}\t{args.model}\tlen={generated}\t"
          f"{generated / best:.2f}\t{best:.4f}\t"
          f"{','.join(f'{t:.4f}' for t in times)}")
    return 0


def describe(name, impl) -> str:
    if name == "torch":
        return f"torch/{'cuda' if impl.GPU else 'cpu'}"
    return impl.xp.__name__


def check(model_name, tokens) -> bool:
    """The token sequence has to be the one mamba.c produces.

    A run that drifted is not a slower run, it is a different computation, and
    quoting tokens/sec for it would be meaningless.
    """
    path = pathlib.Path(FIXTURES) / f"{model_name}_greedy.json"
    if not path.exists():
        print(f"missing fixture {path}; run script/mamba_fixtures.rb", file=sys.stderr)
        return False
    expected = json.loads(path.read_text())["tokens"]
    got = [int(t) for t in tokens]
    if len(got) > len(expected):
        print(f"fixture has {len(expected)} tokens, run produced {len(got)}", file=sys.stderr)
        return False
    if got != expected[:len(got)]:
        first = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)
        print(f"token sequence diverged at {first}: got {got[first]}, "
              f"expected {expected[first]}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
