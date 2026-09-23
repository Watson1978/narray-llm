"""Greedy token sequences from transformers, for the Ruby side to match.

There is no C reference for this architecture, so the gate is agreement with
transformers (docs/plans/PLAN-switch.md). Greedy is forced here rather than left to
generation_config.json, so the fixture says exactly what it is.

  python/.venv/bin/python python/switch_fixtures.py data/switch-base-8 \
      python/fixtures/switch-base-8_greedy.json
"""

from __future__ import annotations

import json
import sys

import torch
from transformers import SwitchTransformersForConditionalGeneration

# Masked span filling is what these checkpoints were trained on, so the
# prompts carry sentinel ids. The third is long enough to put more than
# expert_capacity tokens on one expert.
PROMPTS = {
    "short": [8774, 15, 3, 9, 794, 1],
    "sentinel": [37, 3, 9, 3155, 13, 32099, 19, 32098, 5, 1],
    "long": list(range(100, 227)) + [1],
}
MAX_NEW_TOKENS = 24


def main(model_dir: str, out_path: str) -> int:
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32).eval()
    out = {"model": "google/switch-base-8", "max_new_tokens": MAX_NEW_TOKENS, "prompts": {}}
    with torch.inference_mode():
        for name, ids in PROMPTS.items():
            generated = model.generate(
                input_ids=torch.tensor([ids], dtype=torch.long),
                max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                early_stopping=False, no_repeat_ngram_size=0, length_penalty=1.0,
            )
            tokens = generated[0].tolist()
            out["prompts"][name] = {"input_ids": ids, "tokens": tokens}
            print(f"{name:9s} {len(ids):4d} -> {tokens}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
