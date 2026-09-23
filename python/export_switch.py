"""google/switch-base-8's pickle to a float32 safetensors file.

The published checkpoint is a pytorch_model.bin with no safetensors beside
it, and its tensors are bfloat16. Ruby reads safetensors in 35 lines
(docs/next-models.md), but Numo has no half precision at all, so the
conversion widens to float32 here rather than in the loader. Widening
bfloat16 to float32 is exact.

The embedding is stored four times under four names that share one storage
(shared, encoder.embed_tokens, decoder.embed_tokens, lm_head). Only
shared.weight is written; the loader knows the other three are the same.

  python/.venv/bin/python python/export_switch.py \
      data/switch-base-8 data/switch-base-8.safetensors
"""

from __future__ import annotations

import sys

import torch
from safetensors.torch import save_file

TIED = ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight")
SHARED = "shared.weight"


def main(model_dir: str, out_path: str) -> int:
    state = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu", weights_only=True)

    shared = state.get(SHARED)
    if shared is None:
        print(f"{SHARED} is missing", file=sys.stderr)
        return 1
    for name in TIED:
        alias = state.get(name)
        if alias is None:
            print(f"{name} is missing", file=sys.stderr)
            return 1
        if alias.untyped_storage().data_ptr() != shared.untyped_storage().data_ptr():
            print(f"{name} is not tied to {SHARED}; the loader would be wrong", file=sys.stderr)
            return 1

    out = {name: tensor.to(torch.float32).contiguous()
           for name, tensor in state.items() if name not in TIED}
    save_file(out, out_path, metadata={"format": "pt", "source": "google/switch-base-8"})

    count = sum(t.numel() for t in out.values())
    print(f"{len(out)} tensors, {count:,} parameters, {count * 4:,} bytes of float32")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
