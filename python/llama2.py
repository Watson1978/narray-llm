"""Llama 2 inference on NumPy or CuPy.

A deliberate mirror of the Ruby implementation in ../lib/narray_llm/models/llama2,
so that a benchmark compares the two ecosystems rather than two different
algorithms:

  * one array module chosen once (GPU=1 selects CuPy, otherwise NumPy), used
    everywhere afterwards, exactly like the Ruby XM constant
  * the same stories*.bin, read with the layout in ../docs/checkpoint-format-llama2.md
  * no batch dimension, because run.c has none: activations are [t, dim]
  * prefill / decode phases with a [maxT, kv_dim] KV cache per layer
  * the key is cached after RoPE, which is what run.c stores
  * grouped-query attention by broadcasting, not by widening the cache
  * greedy decoding, float32 only

Nothing here is tuned. Where a faster or more idiomatic formulation exists, the
comment says so and the mirror is kept.
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

import numpy as np  # host-side token ids, always NumPy (mirrors Ruby's HM)

HEADER_INTS = 7
HEADER_BYTES = HEADER_INTS * 4

RMSNORM_EPS = 1e-5
ROPE_THETA = 10_000.0
EXP_LIMIT = 88.0
MASK_VALUE = -1.0e9

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

    @property
    def grouped_query(self) -> bool:
        return self.num_kv_heads != self.num_heads


def tensor_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    d, l, h = config.dim, config.num_layers, config.hidden_dim
    q = config.num_heads * config.head_size
    kv = config.kv_dim
    return {
        "token_embedding_table": (config.vocab_size, d),
        "rms_att_weight": (l, d),
        "wq": (l, q, d),
        "wk": (l, kv, d),
        "wv": (l, kv, d),
        "wo": (l, d, q),
        "rms_ffn_weight": (l, d),
        "w1": (l, h, d),
        "w2": (l, d, h),
        "w3": (l, h, d),
        "rms_final_weight": (d,),
        "wcls": (config.vocab_size, d),
    }


def load_checkpoint(path: str) -> tuple[Config, dict[str, "xp.ndarray"]]:
    with open(path, "rb") as f:
        header = struct.unpack("<7i", f.read(HEADER_BYTES))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
        # A negative vocab_size flags a classifier that is not shared (run.c:147).
        config = Config(dim=dim, hidden_dim=hidden_dim, num_layers=n_layers,
                        num_heads=n_heads, num_kv_heads=n_kv_heads,
                        vocab_size=abs(vocab_size), max_seq_len=seq_len,
                        shared_classifier=vocab_size > 0)
        shapes = tensor_shapes(config)
        params = {}
        for name in TENSOR_NAMES:
            if name == "wcls":
                # freq_cis_real and freq_cis_imag, unused since run.c computes
                # RoPE on the fly, but present in the file (run.c:136).
                f.read(4 * 2 * (config.max_seq_len * config.head_size // 2))
                if config.shared_classifier:
                    params[name] = params["token_embedding_table"]
                    continue
            shape = shapes[name]
            count = int(np.prod(shape))
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading {name}")
            params[name] = xp.asarray(np.frombuffer(raw, dtype=np.float32).reshape(shape).copy())
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


def load_tokenizer(path: str, vocab_size: int) -> list[bytes]:
    """The file carries no vocabulary size; run.c:387 takes it from the config."""
    pieces = []
    with open(path, "rb") as f:
        struct.unpack("<i", f.read(4))  # max_token_length, unused for decoding
        for _ in range(vocab_size):
            struct.unpack("<f", f.read(4))  # score, only the encoder needs it
            (length,) = struct.unpack("<i", f.read(4))
            pieces.append(f.read(length))
    return pieces


def decode_text(pieces: list[bytes], ids) -> str:
    """run.c:418 decode plus run.c:431 safe_printf, over a whole sequence."""
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


# --- ops (mirrors lib/narray_llm/ops.rb) ---

def contiguous(a):
    return xp.ascontiguousarray(a)


def rmsnorm(x, weight, eps=RMSNORM_EPS):
    """run.c:178. Does not centre the row."""
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    return x / xp.sqrt(ms + eps) * weight


def silu(x):
    """run.c:337. The exp argument is clipped so fp32 never reaches Inf."""
    return x / (1.0 + xp.exp(xp.clip(-x, -EXP_LIMIT, EXP_LIMIT)))


def rope_tables(max_pos, head_size):
    half = head_size // 2
    exponent = xp.arange(half, dtype=xp.float32) * (-2.0 * math.log(ROPE_THETA) / head_size)
    angle = xp.arange(max_pos, dtype=xp.float32).reshape(max_pos, 1) * xp.exp(exponent)
    return xp.cos(angle), xp.sin(angle)


def rope(x, cos_t, sin_t, num_heads):
    """Rotates adjacent pairs within each head (run.c:265-280).

    Put back with concatenate rather than by assigning into a strided slice, to
    mirror what the Ruby side does for Cumo's sake.
    """
    t = x.shape[0]
    head_size = x.shape[1] // num_heads
    half = head_size // 2
    pairs = x.reshape(t, num_heads, half, 2)
    even = pairs[:, :, :, 0]
    odd = pairs[:, :, :, 1]
    c = cos_t.reshape(t, 1, half)
    s = sin_t.reshape(t, 1, half)
    rotated = xp.concatenate([(even * c - odd * s).reshape(t, num_heads, half, 1),
                              (even * s + odd * c).reshape(t, num_heads, half, 1)], axis=3)
    return rotated.reshape(t, num_heads * head_size)


def repeat_kv_heads(x, num_kv_heads, kv_mul):
    """run.c:295 reads head h / kv_mul for query head h.

    Broadcasting into a zeroed array rather than xp.repeat, mirroring the Ruby
    side, which avoids handing an index array to the device.
    """
    if kv_mul == 1:
        return x
    t = x.shape[0]
    head_size = x.shape[1] // num_kv_heads
    widened = xp.zeros((t, num_kv_heads, kv_mul, head_size), dtype=xp.float32) + \
        x.reshape(t, num_kv_heads, 1, head_size)
    return widened.reshape(t, num_kv_heads * kv_mul * head_size)


def softmax_rows(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    e = xp.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(t):
    row = xp.arange(t, dtype=xp.float32).reshape(t, 1)
    col = xp.arange(t, dtype=xp.float32).reshape(1, t)
    return MASK_VALUE * xp.ceil(xp.clip(col - row, 0.0, 1.0))


def one_hot(token_ids, num_classes):
    ids = xp.asarray(np.asarray(token_ids, dtype=np.float32)).reshape(-1, 1)
    col = xp.arange(num_classes, dtype=xp.float32).reshape(1, num_classes)
    return 1.0 - xp.ceil(xp.clip(xp.abs(col - ids), 0.0, 1.0))


def attention(q, k, v, mask, num_heads, num_kv_heads, head_size):
    t = q.shape[0]
    scale = 1.0 / math.sqrt(head_size)
    kv_mul = num_heads // num_kv_heads

    def heads_first(a):
        return contiguous(a.reshape(t, num_heads, head_size).transpose(1, 0, 2))

    queries = heads_first(q)
    keys = heads_first(repeat_kv_heads(k, num_kv_heads, kv_mul))
    values = heads_first(repeat_kv_heads(v, num_kv_heads, kv_mul))

    scores = queries @ contiguous(keys.transpose(0, 2, 1)) * scale + mask
    weights = softmax_rows(scores)
    return contiguous((weights @ values).transpose(1, 0, 2)).reshape(t, num_heads * head_size)


def decode_attention(q, keys, values, num_heads, num_kv_heads):
    channels = q.shape[1]
    head_size = channels // num_heads
    length = keys.shape[0]
    scale = 1.0 / math.sqrt(head_size)

    if num_kv_heads == num_heads:
        scores = (keys * q).reshape(length, num_heads, head_size).sum(axis=2).T * scale
        weights = softmax_rows(scores)
        return (values.reshape(length, num_heads, head_size) *
                weights.T.reshape(length, num_heads, 1)).sum(axis=0).reshape(1, channels)

    kv_mul = num_heads // num_kv_heads
    scores = (keys.reshape(length, num_kv_heads, 1, head_size) *
              q.reshape(1, num_kv_heads, kv_mul, head_size)).sum(axis=3)
    scores = scores.reshape(length, num_heads).T * scale
    weights = softmax_rows(scores)
    return (values.reshape(length, num_kv_heads, 1, head_size) *
            weights.T.reshape(length, num_kv_heads, kv_mul, 1)).sum(axis=0).reshape(1, channels)


class KVCache:
    def __init__(self, num_layers, max_seq_len, channels):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.channels = channels
        self.keys = [xp.zeros((max_seq_len, channels), dtype=xp.float32) for _ in range(num_layers)]
        self.values = [xp.zeros((max_seq_len, channels), dtype=xp.float32) for _ in range(num_layers)]
        self.positions = [0] * num_layers

    @staticmethod
    def bytes_for(num_layers, max_seq_len, channels):
        return 2 * num_layers * max_seq_len * channels * 4

    def reset(self):
        self.positions = [0] * self.num_layers

    def append(self, layer, keys, values):
        rows = keys.shape[0] if keys.ndim > 1 else 1
        position = self.positions[layer]
        if position + rows > self.max_seq_len:
            raise ValueError(f"kv cache overflow on layer {layer}: {position} + {rows} rows "
                             f"exceeds max_seq_len {self.max_seq_len}")
        self.keys[layer][position:position + rows] = keys.reshape(rows, self.channels)
        self.values[layer][position:position + rows] = values.reshape(rows, self.channels)
        self.positions[layer] = position + rows

    def view(self, layer):
        length = self.positions[layer]
        return self.keys[layer][:length], self.values[layer][:length]


class Model:
    def __init__(self, checkpoint_path):
        self.config, params = load_checkpoint(checkpoint_path)
        c = self.config
        self.token_embedding = contiguous(params["token_embedding_table"])
        self.wcls_t = contiguous(params["wcls"].T)
        self.rms_final = params["rms_final_weight"]
        # Weights are stored [out, in]; every matmul wants [in, out].
        self.layers = [{
            "rms_att_weight": contiguous(params["rms_att_weight"][i]),
            "wq_t": contiguous(params["wq"][i].T),
            "wk_t": contiguous(params["wk"][i].T),
            "wv_t": contiguous(params["wv"][i].T),
            "wo_t": contiguous(params["wo"][i].T),
            "rms_ffn_weight": contiguous(params["rms_ffn_weight"][i]),
            "w1_t": contiguous(params["w1"][i].T),
            "w2_t": contiguous(params["w2"][i].T),
            "w3_t": contiguous(params["w3"][i].T),
        } for i in range(c.num_layers)]
        self._full_mask = None
        self._rope = None

    def parameter_bytes(self):
        tensors = [self.token_embedding, self.wcls_t, self.rms_final]
        tensors += [t for layer in self.layers for t in layer.values()]
        return 4 * sum(int(t.size) for t in tensors)

    def new_cache(self):
        c = self.config
        return KVCache(c.num_layers, c.max_seq_len, c.kv_dim)

    def _causal_mask(self, seq_len):
        if self._full_mask is None:
            self._full_mask = causal_mask(self.config.max_seq_len)
        return contiguous(self._full_mask[:seq_len, :seq_len])

    def _rope_slice(self, start, count):
        if self._rope is None:
            self._rope = rope_tables(self.config.max_seq_len, self.config.head_size)
        cos_t, sin_t = self._rope
        return contiguous(cos_t[start:start + count]), contiguous(sin_t[start:start + count])

    def forward(self, tokens, last_only=False, cache=None):
        ids = np.asarray(tokens, dtype=np.int32).reshape(-1)
        t = int(ids.shape[0])
        c = self.config
        if t > c.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {c.max_seq_len}")

        x = one_hot(ids, c.vocab_size) @ self.token_embedding
        cos_t, sin_t = self._rope_slice(0, t)
        mask = self._causal_mask(t)
        for layer, w in enumerate(self.layers):
            sink = (lambda k, v, _l=layer: cache.append(_l, k, v)) if cache is not None else None
            x = self._block(x, w, cos_t, sin_t, mask, kv_sink=sink)

        x = rmsnorm(x, self.rms_final)
        if last_only:
            x = contiguous(x[t - 1:t])
        return x @ self.wcls_t

    def _block(self, x, w, cos_t, sin_t, mask, kv_sink=None):
        c = self.config
        h = rmsnorm(x, w["rms_att_weight"])
        q = h @ w["wq_t"]
        k = h @ w["wk_t"]
        v = h @ w["wv_t"]
        q = rope(q, cos_t, sin_t, c.num_heads)
        k = rope(k, cos_t, sin_t, c.num_kv_heads)
        # After RoPE: run.c rotates s->k in place and it already points into the
        # cache row (run.c:259, :279).
        if kv_sink is not None:
            kv_sink(k, v)

        attn = attention(q, k, v, mask, c.num_heads, c.num_kv_heads, c.head_size)
        x = x + attn @ w["wo_t"]

        h2 = rmsnorm(x, w["rms_ffn_weight"])
        swiglu = silu(h2 @ w["w1_t"]) * (h2 @ w["w3_t"])
        return x + swiglu @ w["w2_t"]

    def prefill(self, tokens, cache):
        cache.reset()
        return self.forward(tokens, last_only=True, cache=cache)

    def decode(self, token_id, position, cache):
        c = self.config
        if position >= c.max_seq_len:
            raise ValueError(f"position {position} exceeds max_seq_len {c.max_seq_len}")
        if not 0 <= token_id < c.vocab_size:
            raise ValueError(f"invalid token id {token_id}")

        # A Python int index, so no device-side gather (mirrors the Ruby side).
        x = self.token_embedding[token_id].reshape(1, c.dim)
        cos_t, sin_t = self._rope_slice(position, 1)
        for layer, w in enumerate(self.layers):
            x = self._decode_block(x, w, layer, cos_t, sin_t, cache)
        x = rmsnorm(x, self.rms_final)
        return x @ self.wcls_t

    def _decode_block(self, x, w, layer, cos_t, sin_t, cache):
        c = self.config
        h = rmsnorm(x, w["rms_att_weight"])
        q = rope(h @ w["wq_t"], cos_t, sin_t, c.num_heads)
        k = rope(h @ w["wk_t"], cos_t, sin_t, c.num_kv_heads)
        v = h @ w["wv_t"]
        cache.append(layer, k, v)

        keys, values = cache.view(layer)
        attn = decode_attention(q, keys, values, c.num_heads, c.num_kv_heads)
        x = x + attn @ w["wo_t"]

        h2 = rmsnorm(x, w["rms_ffn_weight"])
        swiglu = silu(h2 @ w["w1_t"]) * (h2 @ w["w3_t"])
        return x + swiglu @ w["w2_t"]


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
        return (self._generate_cached if cache else self._generate_recomputing)(
            tokens, max_new_tokens, stop_at_eot)

    def _argmax(self, logits):
        row = logits.reshape(self.model.config.vocab_size)
        # The one readback per generated token.
        return int(row.argmax())

    def _generate_recomputing(self, tokens, max_new_tokens, stop_at_eot):
        for _ in range(max_new_tokens):
            token = self._argmax(self.model.forward(tokens, last_only=True))
            tokens.append(token)
            if stop_at_eot and token == self.stop_token:
                break
        return tokens

    def _generate_cached(self, tokens, max_new_tokens, stop_at_eot):
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
        xp.cuda.Stream.null.synchronize()
