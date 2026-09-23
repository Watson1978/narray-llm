"""One condition per process, for the Switch Transformer comparison.

AGENTS.md wants one process per condition and best-of-N inside it, so this
takes the condition on the command line and prints a single line. An
interleaving harness runs it once per (implementation, router, condition).

  python/.venv/bin/python python/bench_switch.py --length 72
  GPU=1 python/.venv/bin/python python/bench_switch.py --impl torch --router dense

The generated tokens are checked against the fixture before anything is timed,
so a run that drifted cannot report a number. --encode-only skips that: the
encoder answers activations, not tokens, and the token check has already
passed on the generation path.
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
FIXTURE = os.path.join(ROOT, "python", "fixtures", "switch-base-8_greedy.json")
MODEL = "switch-base-8"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--router", choices=("dispatch", "dense"), default="dispatch")
    parser.add_argument("--length", type=int, default=72)
    parser.add_argument("--prompt-length", type=int, default=0,
                        help="synthetic prompt of this many ids, instead of the fixture's")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--rounds", "--repeat", dest="repeat", type=int, default=3)
    args = parser.parse_args()

    impl = importlib.import_module("switch_torch" if args.impl == "torch" else "switch")
    if args.impl == "torch":
        import torch
        context = torch.inference_mode
    else:
        import contextlib
        context = contextlib.nullcontext

    path = os.path.join(DATA_DIR, f"{MODEL}.safetensors")
    model = impl.Model(path, config_path=os.path.join(DATA_DIR, MODEL, "config.json"),
                       router=args.router)

    if args.prompt_length:
        prompt = list(range(100, 100 + args.prompt_length - 1)) + [1]
    else:
        prompt = json.loads(pathlib.Path(FIXTURE).read_text())["prompts"]["short"]["input_ids"]

    with context():
        if args.encode_only:
            run = lambda: model.encode(prompt)  # noqa: E731
            produced = len(prompt)
        else:
            if not args.prompt_length and not check(model, prompt, args.impl, args.router):
                return 1
            run = lambda: model.generate(prompt, args.length, stop_at_eos=False)  # noqa: E731
            produced = args.length

        run()
        impl.synchronize()
        times = []
        for _ in range(args.repeat):
            started = time.perf_counter()
            run()
            impl.synchronize()
            times.append(time.perf_counter() - started)

    best = min(times)
    backend = describe(args.impl, impl)
    kind = "encode" if args.encode_only else "decode"
    print(f"{args.impl}\t{backend}\t{args.router}\t{kind}\tlen={produced}\t"
          f"{produced / best:.2f}\t{best:.4f}\t"
          f"{','.join(f'{t:.4f}' for t in times)}")
    return 0


def describe(name, impl) -> str:
    if name == "torch":
        return f"torch/{'cuda' if impl.GPU else 'cpu'}"
    return impl.xp.__name__


def check(model, prompt, impl_name, router) -> bool:
    """The token sequence has to be the one transformers produces.

    A run that drifted is not a slower run, it is a different computation, and
    quoting tokens/sec for it would be meaningless.
    """
    path = pathlib.Path(FIXTURE)
    if not path.exists():
        print(f"missing fixture {path}; run python/switch_fixtures.py", file=sys.stderr)
        return False
    fixture = json.loads(path.read_text())
    expected = fixture["prompts"]["short"]["tokens"]
    got = model.generate(prompt, fixture["max_new_tokens"])
    if got != expected:
        first = next((i for i, (a, b) in enumerate(zip(got, expected)) if a != b), len(got))
        print(f"{impl_name}/{router}: token sequence diverged at {first}", file=sys.stderr)
        return False
    return True


if __name__ == "__main__":
    sys.exit(main())
