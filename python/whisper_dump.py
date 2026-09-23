"""Dumps what the encoder of openai/whisper-tiny computes, for the Ruby side.

There is no C reference for this model, so transformers is the reference
(docs/plans/PLAN-whisper.md). The values are taken with forward hooks on the real model
rather than from a re-implementation, so nothing here can drift from what
transformers actually computes.

The waveform is synthetic and built from a formula, so the mel is the same on
any machine with no audio file to carry around. The real feature extractor
turns it into mel bins, so the front end being exercised is the published one.

  python/.venv/bin/python python/whisper_dump.py data/whisper-tiny \
      data/whisper-tiny_encoder_state.safetensors
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

SAMPLE_RATE = 16000
SECONDS = 30

# The prompt whisper builds for English transcription with no timestamps,
# followed by a few ids chosen to exercise positions rather than to mean
# anything. Teacher forcing over this gives one reference logit row per
# position, which is a stronger check than a token sequence that repeats.
DECODER_INPUT_IDS = [50258, 50259, 50359, 50363, 2411, 50, 1002, 13, 400, 293]


def waveform() -> np.ndarray:
    """A chirp under an envelope, plus a steady tone. Deterministic."""
    t = np.arange(SAMPLE_RATE * SECONDS, dtype=np.float64) / SAMPLE_RATE
    sweep = np.sin(2 * np.pi * (200.0 + 60.0 * t) * t)
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    envelope = 0.5 * (1.0 - np.cos(2 * np.pi * t / SECONDS))
    return ((sweep + tone) * envelope).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("out_path")
    args = parser.parse_args()

    extractor = WhisperFeatureExtractor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.float32)
    model.eval()
    encoder = model.model.encoder

    audio = waveform()
    features = extractor(audio, sampling_rate=SAMPLE_RATE,
                         return_tensors="pt").input_features.to(torch.float32)

    out: dict[str, torch.Tensor] = {}
    handles = []

    def keep(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            out[name] = value.detach().to(torch.float32).squeeze(0).contiguous()
        return hook

    handles.append(encoder.conv1.register_forward_hook(keep("conv1")))
    handles.append(encoder.conv2.register_forward_hook(keep("conv2")))
    handles.append(encoder.layer_norm.register_forward_hook(keep("final_norm")))
    for i, layer in enumerate(encoder.layers):
        handles.append(layer.register_forward_hook(keep(f"layer.{i}.output")))
        handles.append(layer.self_attn_layer_norm.register_forward_hook(keep(f"layer.{i}.attn_norm")))

    with torch.inference_mode():
        result = encoder(input_features=features)
    for handle in handles:
        handle.remove()

    # The waveform itself, so the Ruby side works from the same float32
    # samples rather than from the same formula evaluated twice.
    out["waveform"] = torch.from_numpy(audio).contiguous()
    out["input_features"] = features.squeeze(0).contiguous()
    out["encoder_last_hidden_state"] = \
        result.last_hidden_state.detach().to(torch.float32).squeeze(0).contiguous()

    # The decoder over a fixed sequence, teacher forced. A token sequence that
    # repeats one id is a weak check on its own, so the logits at every
    # position are dumped too and the Ruby side has to match them one by one.
    forced = DECODER_INPUT_IDS
    with torch.inference_mode():
        decoded = model(input_features=features,
                        decoder_input_ids=torch.tensor([forced], dtype=torch.long))
    out["decoder_input_ids"] = torch.tensor(forced, dtype=torch.int64).contiguous()
    out["decoder_logits"] = decoded.logits.detach().to(torch.float32).squeeze(0).contiguous()

    save_file({k: v.clone() for k, v in out.items()}, args.out_path)
    print(f"{len(out)} tensors, mel {tuple(features.shape[1:])}, "
          f"decoder {tuple(out['decoder_logits'].shape)} -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
