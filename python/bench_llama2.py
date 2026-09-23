"""One condition per process, for the Llama 2 comparison.

CLAUDE.md wants one process per condition and best-of-N inside it, so this
takes the condition on the command line and prints a single line. An
interleaving harness runs it once per (implementation, condition).

  python/.venv/bin/python python/bench_llama2.py --length 64 --cache 1
  GPU=1 python/.venv/bin/python python/bench_llama2.py --impl torch --length 256 --cache 0

--impl numpy selects python/llama2.py (NumPy, or CuPy with GPU=1); --impl torch
selects python/llama2_torch.py. The generated tokens are checked against the
fixture before anything is timed, so a run that drifted cannot report a number.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pathlib
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
TOKENIZERS = {"stories260K": "tok512.bin"}
DEFAULT_TOKENIZER = "tokenizer.bin"
Q80_MAGIC = 0x616B3432
BOS = 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--model", default="stories110M")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--cache", type=int, default=1)
    parser.add_argument("--rounds", "--repeat", dest="repeat", type=int, default=3)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    checkpoint = os.path.join(DATA_DIR, f"{args.model}.bin")
    # runq.c's int8 checkpoints carry their own magic, so the file says which
    # implementation to build rather than a flag having to agree with it.
    with open(checkpoint, "rb") as probe:
        quantized = struct.unpack("<I", probe.read(4))[0] == Q80_MAGIC

    if args.impl == "torch":
        import torch
        impl = importlib.import_module("llama2_q80_torch" if quantized else "llama2_torch")
        context = torch.inference_mode
    else:
        impl = importlib.import_module("llama2_q80" if quantized else "llama2")
        import contextlib
        context = contextlib.nullcontext

    tokenizer_path = os.path.join(DATA_DIR, TOKENIZERS.get(args.model, DEFAULT_TOKENIZER))
    model = impl.Model(checkpoint)
    pieces = impl.load_tokenizer(tokenizer_path, model.config.vocab_size)
    generator = impl.Generator(model)
    cache = bool(args.cache)

    with context():
        tokens = generator.generate([BOS], args.length, cache=cache)
        if not args.no_check and not check(args.model, args.length, tokens):
            return 1

        # Warm up, then best-of-N with no synchronization inside a run.
        generator.generate([BOS], 2, cache=cache)
        impl.synchronize()
        times = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            generator.generate([BOS], args.length, cache=cache)
            impl.synchronize()
            times.append(time.perf_counter() - started)

    best = min(times)
    # Divided by what was actually produced, not what was asked for: the loop
    # stops early if the model emits the delimiter (stories110M does, at 243).
    generated = len(tokens) - 1
    backend = describe(args.impl, impl)
    print(f"{args.impl}\t{backend}\t{args.model}\tlen={generated}\tcache={int(cache)}\t"
          f"{generated / best:.2f}\t{best:.4f}\t"
          f"{','.join(f'{t:.4f}' for t in times)}")
    return 0


def describe(name, impl) -> str:
    if name == "torch":
        return f"torch/{'cuda' if impl.GPU else 'cpu'}"
    return impl.xp.__name__


def check(model_name, length, tokens) -> bool:
    """The token sequence has to be the one llama2.c produces.

    A run that drifted is not a slower run, it is a different computation, and
    quoting tokens/sec for it would be meaningless.
    """
    path = pathlib.Path(FIXTURES) / f"{model_name}_greedy.json"
    if not path.exists():
        print(f"missing fixture {path}; run script/llama2_fixtures.rb", file=sys.stderr)
        return False
    expected = json.loads(path.read_text())["tokens"]
    got = [int(t) for t in tokens]
    if len(got) > len(expected):
        print(f"fixture has {len(expected)} tokens, run produced {len(got)}", file=sys.stderr)
        return False
    # A short run is fine only when it stopped on the delimiter, which is a
    # property of the sequence and so has to match the fixture too.
    if len(got) < length + 1 and got[-1] != expected[len(got) - 1]:
        print(f"run stopped at {len(got)} tokens but the fixture does not", file=sys.stderr)
        return False
    if got != expected[:len(got)]:
        first = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)
        print(f"token sequence diverged at {first}: got {got[first]}, "
              f"expected {expected[first]}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
