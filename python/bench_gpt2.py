"""One condition per process, for the GPT-2 comparison.

The same shape as bench_llama2.py: bench_gpt2_sweep.py and bench_gpt2_torch_sweep.py walk every
condition inside one process, which CLAUDE.md's rule 11 says measures the
allocator as much as the work. This takes the condition on the command line
and prints a single line, so an interleaving harness can run it once per
(implementation, condition).

  GPU=1 python/.venv/bin/python python/bench_gpt2.py --impl numpy --length 64 --cache 1
  GPU=1 python/.venv/bin/python python/bench_gpt2.py --impl torch --batch 8 --length 128
  GPU=1 python/.venv/bin/python python/bench_gpt2.py --impl torch --length 256 --cache 0
  GPU=1 DTYPE=fp16 python/.venv/bin/python python/bench_gpt2.py --impl numpy --length 256

The generated tokens are checked against python/fixtures/gpt2_124M_greedy.json
before anything is timed, so a run that drifted cannot report a number. The
fixture is fp32, so under DTYPE the check reports where the sequence parts
instead of failing the run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "fixtures", "gpt2_124M_greedy.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--cache", type=int, default=1)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--rounds", "--repeat", dest="repeat", type=int, default=3)
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    if args.impl == "torch":
        import torch
        import gpt2_torch as impl
        context = torch.inference_mode
        dtype = str(impl.DT).removeprefix("torch.")
    else:
        import contextlib
        import gpt2 as impl
        context = contextlib.nullcontext
        dtype = impl.DT.__name__
    reduced = dtype != "float32"

    model = impl.Model(os.path.join(DATA_DIR, "gpt2_124M.bin"))
    eot, _table = impl.load_tokenizer(os.path.join(DATA_DIR, "gpt2_tokenizer.bin"))
    generator = impl.Generator(model, eot_token=eot)
    cache = bool(args.cache)

    batch = args.batch
    with context():
        if batch is None:
            tokens = generator.generate([eot], args.length, cache=cache)
            if not args.no_check and not check(args.length, tokens) and not reduced:
                return 1
            run = lambda: generator.generate([eot], args.length, cache=cache)  # noqa: E731
            warm = lambda: generator.generate([eot], 2, cache=cache)  # noqa: E731
            generated = len(tokens) - 1
        else:
            if not cache:
                print("batched generation needs the cache", file=sys.stderr)
                return 1
            prompts = [[eot] for _ in range(batch)]
            sequences = generator.generate_batch([row[:] for row in prompts], args.length,
                                                 stop_at_eot=False)
            # The whole batch runs the same prompt, so each row has to be the
            # sequence the unbatched fixture already pins down.
            if not args.no_check and not reduced:
                for row in sequences:
                    if not check(args.length, row):
                        return 1
            run = lambda: generator.generate_batch(  # noqa: E731
                [row[:] for row in prompts], args.length, stop_at_eot=False)
            warm = lambda: generator.generate_batch(  # noqa: E731
                [row[:] for row in prompts], 2, stop_at_eot=False)
            generated = len(sequences[0]) - 1

        warm()
        impl.synchronize()
        times = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            run()
            impl.synchronize()
            times.append(time.perf_counter() - started)

    best = min(times)
    rows = batch or 1
    backend = f"torch/{'cuda' if impl.GPU else 'cpu'}" if args.impl == "torch" else impl.xp.__name__
    print(f"{args.impl}\t{backend}\t{dtype}\tgpt2_124M\tlen={generated}\tcache={int(cache)}\t"
          f"batch={rows}\t{generated * rows / best:.2f}\t{best:.4f}\t"
          f"{','.join(f'{t:.4f}' for t in times)}")
    return 0


def check(length, tokens) -> bool:
    path = pathlib.Path(FIXTURE)
    if not path.exists():
        print(f"missing fixture {path}; run script/gpt2_fixtures.rb", file=sys.stderr)
        return False
    expected = json.loads(path.read_text())["sequences"].get(str(length))
    if expected is None:
        print(f"fixture has no sequence for length {length}", file=sys.stderr)
        return False
    got = [int(t) for t in tokens[1:]]
    if got != expected:
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), len(got))
        print(f"token sequence diverged at {first}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
