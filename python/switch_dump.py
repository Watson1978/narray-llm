"""Dumps what the encoder of google/switch-base-8 computes, for the Ruby side.

There is no C reference for this model, so transformers is the reference
(docs/plans/PLAN-switch.md). The values are taken with forward hooks on the real model
rather than from a re-implementation, so nothing here can drift from what
transformers actually computes.

The output is a safetensors file, which the Ruby side already reads.

  python/.venv/bin/python python/switch_dump.py data/switch-base-8 \
      data/switch-base-8_encoder_state.safetensors --tokens 8774,15,3,9,794
"""

from __future__ import annotations

import argparse
import json

import torch
from safetensors.torch import save_file
from transformers import SwitchTransformersForConditionalGeneration


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_path")
    parser.add_argument("--tokens", default="8774,15,3,9,794,1",
                        help="comma separated input ids, ending with eos (1)")
    args = parser.parse_args()

    ids = [int(t) for t in args.tokens.split(",")]
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.float32)
    model.eval()
    encoder = model.encoder

    out: dict[str, torch.Tensor] = {}
    handles = []

    def keep(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            out[name] = value.detach().to(torch.float32).squeeze(0).contiguous()
        return hook

    def keep_router(name):
        def hook(_module, _inputs, output):
            probs, expert_index, logits = output
            out[f"{name}.probs"] = probs.detach().to(torch.float32).contiguous()
            out[f"{name}.expert_index"] = expert_index.detach().to(torch.int64).contiguous()
            out[f"{name}.logits"] = logits.detach().to(torch.float32).contiguous()
        return hook

    handles.append(encoder.embed_tokens.register_forward_hook(keep("embed")))
    handles.append(encoder.final_layer_norm.register_forward_hook(keep("final_norm")))
    for i, block in enumerate(encoder.block):
        handles.append(block.layer[0].register_forward_hook(keep(f"block.{i}.after_attention")))
        handles.append(block.register_forward_hook(keep(f"block.{i}.output")))
        mlp = block.layer[1].mlp
        if hasattr(mlp, "router"):
            handles.append(mlp.router.register_forward_hook(keep_router(f"block.{i}.router")))

    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long)
        result = encoder(input_ids=input_ids)
    for handle in handles:
        handle.remove()

    out["input_ids"] = input_ids.squeeze(0).to(torch.int64).contiguous()
    out["encoder_last_hidden_state"] = \
        result.last_hidden_state.detach().to(torch.float32).squeeze(0).contiguous()

    # The bias the first block computes is shared by every block in the stack.
    bias = encoder.block[0].layer[0].SelfAttention.compute_bias(len(ids), len(ids))
    out["relative_bias"] = bias.detach().to(torch.float32).squeeze(0).contiguous()

    save_file({k: v.clone() for k, v in out.items()},
              args.out_path, metadata={"tokens": json.dumps(ids)})
    print(f"{len(out)} tensors for {len(ids)} tokens -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
