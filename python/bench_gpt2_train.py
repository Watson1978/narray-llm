"""One condition per process, for the training comparison.

  GPU=1 IDIOMATIC=1 python/.venv/bin/python python/bench_gpt2_train.py --impl numpy
  GPU=1 python/.venv/bin/python python/bench_gpt2_train.py --impl torch --stop-after backward

The loss sequence llm.c pins is checked before anything is timed, so a number
never comes out of a run that is computing the wrong thing. Steps are not
repeatable -- each moves the weights -- so the time is the total over the steps
after the first, which pays for handles, the allocator and the optimiser's
first allocation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("NARRAY_LLM_DATA") or os.path.join(ROOT, "data")

# test_gpt2.c:89-99.
EXPECTED_LOSSES = [
    5.270007133483887, 4.059706687927246, 3.3751230239868164, 2.8007826805114746,
    2.315382242202759, 1.8490285873413086, 1.3946564197540283, 0.9991465210914612,
    0.6240804195404053, 0.37651097774505615,
]
LOSS_TOLERANCE = 1e-2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--stop-after", choices=("forward", "backward", "update"),
                        default="update")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()

    import gpt2
    import gpt2_train
    config, _ = gpt2.load_checkpoint(os.path.join(DATA_DIR, "gpt2_124M.bin"))
    state = gpt2_train.DebugState(os.path.join(DATA_DIR, "gpt2_124M_debug_state.bin"), config)

    if args.impl == "torch":
        import gpt2_torch
        import gpt2_torch_train
        model = gpt2_torch.Model(os.path.join(DATA_DIR, "gpt2_124M.bin"))
        trainer = gpt2_torch_train.Trainer(model)
        backend = f"torch/{'cuda' if gpt2_torch.GPU else 'cpu'}"
        synchronize = gpt2_torch.synchronize

        def run(step):
            loss = trainer.forward(state.inputs, state.targets)
            value = float(loss)
            if args.stop_after != "forward":
                trainer.backward(loss)
                if args.stop_after != "backward":
                    trainer.update()
            return value
    else:
        model = gpt2.Model(os.path.join(DATA_DIR, "gpt2_124M.bin"))
        trainer = gpt2_train.Trainer(model)
        backend = gpt2.xp.__name__
        synchronize = gpt2.synchronize

        def run(step):
            loss, acts = trainer.forward(state.inputs, state.targets)
            if args.stop_after != "forward":
                grads = trainer.backward(acts)
                if args.stop_after != "backward":
                    trainer.step(grads)
            return loss

    times = []
    for step in range(args.steps):
        started = time.perf_counter()
        loss = run(step)
        synchronize()
        times.append(time.perf_counter() - started)

        # Without the update the weights never move, so only the first loss is
        # the sequence's.
        index = step if args.stop_after == "update" else 0
        want = EXPECTED_LOSSES[index] if index < len(EXPECTED_LOSSES) else None
        if not args.no_check and want is not None and abs(loss - want) >= LOSS_TOLERANCE:
            print(f"step {step}: loss {loss} but llm.c has {want}", file=sys.stderr)
            return 1

    timed = times[1:]
    per = sum(timed) / len(timed)
    print(f"{args.impl}\t{backend}\t{args.stop_after}\tsteps={args.steps}\t"
          f"{per:.5f}\t{1.0 / per:.2f}\t{trainer.bytes() / 1048576:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
