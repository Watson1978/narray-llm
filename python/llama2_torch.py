"""Llama 2 inference in PyTorch eager, fp32.

Unlike python/llama2.py this is NOT a structural mirror of the Ruby code. It is
the ceiling reference: what an ecosystem with fused kernels does with the same
weights and the same task. So it uses scaled_dot_product_attention (with
enable_gqa, so grouped queries never widen the cache), F.rms_norm and F.silu,
and leaves the rest to torch.

What it does keep identical, because those are the comparison's premises:

  * the same stories*.bin (../docs/checkpoint-format-llama2.md)
  * fp32 only, TF32 disabled on both matmul and cudnn
  * prefill / decode phases, a [maxT, kv_dim] cache per layer, greedy decoding
  * RoPE in llama2.c's interleaved form: adjacent pairs, not split halves. The
    HuggingFace layout rotates x[:hs/2] against x[hs/2:] and would silently
    produce different text from the same weights.
  * torch.compile is not used: its first-call compilation would make the
    timing protocol a different conversation. Eager only.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
DEVICE = torch.device("cuda" if GPU else "cpu")

# Both flags matter and both default to True on Ampere and later. With either
# one on, matmuls run in TF32 and the generated tokens stop matching.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HEADER_BYTES = 7 * 4
RMSNORM_EPS = 1e-5
ROPE_THETA = 10_000.0
BOS_TOKEN = 1

TENSOR_NAMES = (
    "token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo",
    "rms_ffn_weight", "w1", "w2", "w3", "rms_final_weight", "wcls",
)


@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_seq_len: int
    shared_classifier: bool

    @property
    def head_size(self) -> int:
        return self.dim // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.dim * self.num_kv_heads // self.num_heads

    @property
    def kv_mul(self) -> int:
        return self.num_heads // self.num_kv_heads


def tensor_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    d, l, h = config.dim, config.num_layers, config.hidden_dim
    q, kv = config.num_heads * config.head_size, config.kv_dim
    return {
        "token_embedding_table": (config.vocab_size, d), "rms_att_weight": (l, d),
        "wq": (l, q, d), "wk": (l, kv, d), "wv": (l, kv, d), "wo": (l, d, q),
        "rms_ffn_weight": (l, d), "w1": (l, h, d), "w2": (l, d, h), "w3": (l, h, d),
        "rms_final_weight": (d,), "wcls": (config.vocab_size, d),
    }


def load_checkpoint(path: str, device) -> tuple[Config, dict[str, torch.Tensor]]:
    with open(path, "rb") as f:
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = \
            struct.unpack("<7i", f.read(HEADER_BYTES))
        config = Config(dim=dim, hidden_dim=hidden_dim, num_layers=n_layers,
                        num_heads=n_heads, num_kv_heads=n_kv_heads,
                        vocab_size=abs(vocab_size), max_seq_len=seq_len,
                        shared_classifier=vocab_size > 0)
        shapes = tensor_shapes(config)
        params = {}
        for name in TENSOR_NAMES:
            if name == "wcls":
                f.read(4 * 2 * (config.max_seq_len * config.head_size // 2))  # freq_cis, unused
                if config.shared_classifier:
                    params[name] = params["token_embedding_table"]
                    continue
            shape = shapes[name]
            count = int(np.prod(shape))
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading {name}")
            array = np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()
            params[name] = torch.from_numpy(array).to(device)
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


def load_tokenizer(path: str, vocab_size: int) -> list[bytes]:
    pieces = []
    with open(path, "rb") as f:
        f.read(4)  # max_token_length
        for _ in range(vocab_size):
            f.read(4)  # score
            (length,) = struct.unpack("<i", f.read(4))
            pieces.append(f.read(length))
    return pieces


def decode_text(pieces: list[bytes], ids) -> str:
    out = bytearray()
    for prev, token in zip(ids, ids[1:]):
        piece = pieces[int(token)]
        if int(prev) == BOS_TOKEN and piece.startswith(b" "):
            piece = piece[1:]
        if piece.startswith(b"<0x") and len(piece) >= 5:
            piece = bytes([int(piece[3:5], 16)])
        if not piece:
            continue
        if len(piece) == 1 and not (0x20 <= piece[0] <= 0x7E or piece[0] in (9, 10, 11, 12, 13)):
            continue
        out += piece
    return out.decode("utf-8", errors="replace")


def rope_tables(max_pos, head_size, device):
    half = head_size // 2
    exponent = torch.arange(half, dtype=torch.float32, device=device) * \
        (-2.0 * math.log(ROPE_THETA) / head_size)
    angle = torch.arange(max_pos, dtype=torch.float32, device=device).unsqueeze(1) * \
        torch.exp(exponent)
    return torch.cos(angle), torch.sin(angle)


def apply_rope(x, cos_t, sin_t):
    """x is [1, heads, t, head_size]; cos_t / sin_t are [t, head_size / 2].

    llama2.c rotates x[2i] against x[2i + 1] (run.c:270-280), so the pairs are
    adjacent. Reshaping the last axis to (half, 2) is that grouping.
    """
    shape = x.shape
    pairs = x.reshape(*shape[:-1], shape[-1] // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    c = cos_t.reshape(1, 1, cos_t.shape[0], cos_t.shape[1])
    s = sin_t.reshape(1, 1, sin_t.shape[0], sin_t.shape[1])
    return torch.stack((even * c - odd * s, even * s + odd * c), dim=-1).reshape(shape)


class KVCache:
    """One preallocated pair per layer, [1, num_kv_heads, maxT, hs], the layout
    SDPA wants. With grouped queries this is kv_mul times smaller than the
    query width, and enable_gqa lets SDPA read it without widening."""

    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_size, device):
        shape = (num_layers, 1, num_kv_heads, max_seq_len, head_size)
        self.keys = torch.zeros(shape, dtype=torch.float32, device=device)
        self.values = torch.zeros(shape, dtype=torch.float32, device=device)
        self.max_seq_len = max_seq_len
        self.positions = [0] * num_layers

    @staticmethod
    def bytes_for(num_layers, max_seq_len, channels):
        return 2 * num_layers * max_seq_len * channels * 4

    def reset(self):
        self.positions = [0] * len(self.positions)

    def append(self, layer, keys, values):
        rows = keys.shape[2]
        position = self.positions[layer]
        if position + rows > self.max_seq_len:
            raise ValueError(f"kv cache overflow on layer {layer}: {position} + {rows} rows "
                             f"exceeds max_seq_len {self.max_seq_len}")
        self.keys[layer, :, :, position:position + rows] = keys
        self.values[layer, :, :, position:position + rows] = values
        self.positions[layer] = position + rows

    def view(self, layer):
        length = self.positions[layer]
        return self.keys[layer, :, :, :length], self.values[layer, :, :, :length]


class Model:
    def __init__(self, checkpoint_path, device=DEVICE):
        self.device = device
        self.config, params = load_checkpoint(checkpoint_path, device)
        c = self.config
        self.token_embedding = params["token_embedding_table"].contiguous()
        self.wcls_t = params["wcls"].t().contiguous()
        self.rms_final = params["rms_final_weight"].contiguous()
        self.layers = [{
            "rms_att_weight": params["rms_att_weight"][i].contiguous(),
            "wq_t": params["wq"][i].t().contiguous(),
            "wk_t": params["wk"][i].t().contiguous(),
            "wv_t": params["wv"][i].t().contiguous(),
            "wo_t": params["wo"][i].t().contiguous(),
            "rms_ffn_weight": params["rms_ffn_weight"][i].contiguous(),
            "w1_t": params["w1"][i].t().contiguous(),
            "w2_t": params["w2"][i].t().contiguous(),
            "w3_t": params["w3"][i].t().contiguous(),
        } for i in range(c.num_layers)]
        self.cos, self.sin = rope_tables(c.max_seq_len, c.head_size, device)
        self.gqa = c.kv_mul > 1

    def parameter_bytes(self):
        tensors = [self.token_embedding, self.wcls_t, self.rms_final]
        tensors += [t for layer in self.layers for t in layer.values()]
        return sum(t.numel() * t.element_size() for t in tensors)

    def new_cache(self):
        c = self.config
        return KVCache(c.num_layers, c.max_seq_len, c.num_kv_heads, c.head_size, self.device)

    def _norm(self, x, weight):
        return F.rms_norm(x, (self.config.dim,), weight, RMSNORM_EPS)

    def _block(self, x, w, layer, seq_len, start, cache, causal):
        c = self.config
        h = self._norm(x, w["rms_att_weight"])
        q = (h @ w["wq_t"]).view(1, seq_len, c.num_heads, c.head_size).transpose(1, 2)
        k = (h @ w["wk_t"]).view(1, seq_len, c.num_kv_heads, c.head_size).transpose(1, 2)
        v = (h @ w["wv_t"]).view(1, seq_len, c.num_kv_heads, c.head_size).transpose(1, 2)

        cos_t = self.cos[start:start + seq_len]
        sin_t = self.sin[start:start + seq_len]
        q = apply_rope(q, cos_t, sin_t)
        k = apply_rope(k, cos_t, sin_t)
        if cache is not None:
            cache.append(layer, k, v)
            k, v = cache.view(layer)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=self.gqa)
        attn = attn.transpose(1, 2).reshape(seq_len, c.dim)
        x = x + attn @ w["wo_t"]

        h2 = self._norm(x, w["rms_ffn_weight"])
        return x + (F.silu(h2 @ w["w1_t"]) * (h2 @ w["w3_t"])) @ w["w2_t"]

    def forward(self, tokens, last_only=False, cache=None):
        ids = torch.as_tensor(tokens, dtype=torch.long, device=self.device).reshape(-1)
        seq_len = int(ids.shape[0])
        c = self.config
        if seq_len > c.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {c.max_seq_len}")

        x = self.token_embedding[ids]
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, seq_len, 0, cache, causal=seq_len > 1)
        x = self._norm(x, self.rms_final)
        if last_only:
            x = x[seq_len - 1:seq_len]
        return x @ self.wcls_t

    def prefill(self, tokens, cache):
        cache.reset()
        return self.forward(tokens, last_only=True, cache=cache)

    def decode(self, token_id, position, cache):
        c = self.config
        if position >= c.max_seq_len:
            raise ValueError(f"position {position} exceeds max_seq_len {c.max_seq_len}")
        x = self.token_embedding[token_id].reshape(1, c.dim)
        for layer, w in enumerate(self.layers):
            x = self._block(x, w, layer, 1, position, cache, causal=False)
        x = self._norm(x, self.rms_final)
        return x @ self.wcls_t


class Generator:
    def __init__(self, model, stop_token=BOS_TOKEN):
        self.model = model
        self.stop_token = stop_token

    def generate(self, prompt, max_new_tokens, stop_at_eot=True, cache=True):
        tokens = [int(t) for t in prompt]
        if not tokens:
            raise ValueError("prompt must contain at least one token")
        limit = self.model.config.max_seq_len
        if len(tokens) + max_new_tokens > limit:
            raise ValueError(f"{len(tokens)} prompt tokens + {max_new_tokens} generated tokens "
                             f"exceeds max_seq_len {limit}")
        return (self._cached if cache else self._recomputing)(tokens, max_new_tokens, stop_at_eot)

    def _argmax(self, logits):
        return int(logits.reshape(self.model.config.vocab_size).argmax().item())

    def _recomputing(self, tokens, max_new_tokens, stop_at_eot):
        for _ in range(max_new_tokens):
            token = self._argmax(self.model.forward(tokens, last_only=True))
            tokens.append(token)
            if stop_at_eot and token == self.stop_token:
                break
        return tokens

    def _cached(self, tokens, max_new_tokens, stop_at_eot):
        cache = self.model.new_cache()
        logits = self.model.prefill(tokens, cache)
        produced = 0
        while produced < max_new_tokens:
            token = self._argmax(logits)
            tokens.append(token)
            produced += 1
            if stop_at_eot and token == self.stop_token:
                break
            if produced == max_new_tokens:
                break
            logits = self.model.decode(token, len(tokens) - 1, cache)
        return tokens


def synchronize():
    if GPU:
        torch.cuda.synchronize()


def sdpa_backend(model, seq_len, kv_len):
    """Which backend SDPA would pick for these shapes, asked at the real sizes.

    Reported rather than assumed: the choice is what makes this the ceiling
    reference, and flash is fp16/bf16 only so it cannot be picked here.
    """
    if not GPU:
        return "cpu (not queried)"
    c = model.config
    causal = seq_len > 1
    q = torch.empty((1, c.num_heads, seq_len, c.head_size), dtype=torch.float32, device=DEVICE)
    k = torch.empty((1, c.num_kv_heads, kv_len, c.head_size), dtype=torch.float32, device=DEVICE)
    params = torch.backends.cuda.SDPAParams(q, k, k, None, 0.0, causal, model.gqa)
    for name, can_use, enabled in (
        ("cudnn", torch.backends.cuda.can_use_cudnn_attention,
         torch.backends.cuda.cudnn_sdp_enabled),
        ("flash", torch.backends.cuda.can_use_flash_attention,
         torch.backends.cuda.flash_sdp_enabled),
        ("efficient", torch.backends.cuda.can_use_efficient_attention,
         torch.backends.cuda.mem_efficient_sdp_enabled),
    ):
        try:
            if enabled() and can_use(params, False):
                return name
        except RuntimeError:
            continue
    return "math"
