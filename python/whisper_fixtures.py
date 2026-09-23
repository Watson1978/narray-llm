"""Greedy token sequences from transformers, for the Ruby side to match.

There is no C reference for this model, so the gate is agreement with
transformers (docs/plans/PLAN-whisper.md). The language and the task are pinned rather
than detected: generation_config.json leaves the language slot None, which
makes generate run a detection pass, and a fixture that depends on it would
not say what it is.

  python/.venv/bin/python python/whisper_fixtures.py data/whisper-tiny \
      python/fixtures/whisper-tiny_greedy.json
"""

from __future__ import annotations

import json
import sys

import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from whisper_dump import SAMPLE_RATE, waveform  # noqa: E402

MAX_NEW_TOKENS = 32


def main(model_dir: str, out_path: str) -> int:
    extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float32).eval()
    features = extractor(waveform(), sampling_rate=SAMPLE_RATE,
                         return_tensors="pt").input_features.to(torch.float32)

    with torch.inference_mode():
        generated = model.generate(
            features, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
            language="en", task="transcribe", return_timestamps=False,
        )
    tokens = generated[0].tolist()

    config = model.generation_config
    out = {
        "model": "openai/whisper-tiny",
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompt": [config.decoder_start_token_id, 50259, 50359, config.no_timestamps_token_id],
        "suppress_tokens": sorted(config.suppress_tokens),
        "begin_suppress_tokens": sorted(config.begin_suppress_tokens),
        "eos_token_id": config.eos_token_id,
        "tokens": tokens,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{len(tokens)} tokens -> {tokens}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
