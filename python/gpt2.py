"""GPT-2 124M inference on NumPy or CuPy.

A deliberate mirror of the Ruby implementation in ../lib/narray_llm, so that a
benchmark compares the two ecosystems rather than two different algorithms:

  * one array module chosen once (GPU=1 selects CuPy, otherwise NumPy), used
    everywhere afterwards, exactly like the Ruby XM constant
  * the same gpt2_124M.bin, read with the layout in ../docs/checkpoint-format-gpt2.md
  * prefill / decode phases with a [maxT, C] KV cache per layer
  * decode runs one query against the cache with no causal mask
  * greedy decoding
  * attention loops over heads by default. IDIOMATIC=1 selects the batched
    formulation, mirroring what the Ruby side does on its current HEAD, so the
    two ecosystems can be compared with the heads looped and with them batched
  * float32 by default. DTYPE=fp16 selects float16, mirroring the Ruby side's
    DTYPE switch, and carries the same two consequences: the causal mask value
    and the layernorm statistics

Nothing here is tuned. Where a faster or more idiomatic formulation exists, the
comment says so and the mirror is kept.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
IDIOMATIC = os.environ.get("IDIOMATIC", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

DTYPE = os.environ.get("DTYPE", "").lower()
if DTYPE in ("", "fp32", "float32"):
    DT = xp.float32
elif DTYPE in ("fp16", "float16"):
    DT = xp.float16
else:
    raise ValueError(f"unknown DTYPE {DTYPE!r}; use fp32 or fp16 "
                     "(numpy and cupy have no bfloat16)")
FP16 = DT is xp.float16

import numpy as np  # host-side token ids, always NumPy (mirrors Ruby's HM)

MAGIC = 20_240_326
VERSION = 3
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4

TOKENIZER_MAGIC = 20_240_328
LAYERNORM_EPS = 1e-5
GELU_SCALING_FACTOR = float(np.sqrt(2.0 / np.pi))
MASK_VALUE = -1.0e4 if FP16 else -1.0e9

TENSOR_NAMES = (
    "wte", "wpe", "ln1w", "ln1b", "qkvw", "qkvb", "attprojw", "attprojb",
    "ln2w", "ln2b", "fcw", "fcb", "fcprojw", "fcprojb", "lnfw", "lnfb",
)


@dataclass(frozen=True)
class Config:
    max_seq_len: int
    vocab_size: int
    num_layers: int
    num_heads: int
    channels: int
    padded_vocab_size: int


def tensor_shapes(config: Config) -> dict[str, tuple[int, ...]]:
    c, l = config.channels, config.num_layers
    return {
        "wte": (config.padded_vocab_size, c),
        "wpe": (config.max_seq_len, c),
        "ln1w": (l, c), "ln1b": (l, c),
        "qkvw": (l, 3 * c, c), "qkvb": (l, 3 * c),
        "attprojw": (l, c, c), "attprojb": (l, c),
        "ln2w": (l, c), "ln2b": (l, c),
        "fcw": (l, 4 * c, c), "fcb": (l, 4 * c),
        "fcprojw": (l, c, 4 * c), "fcprojb": (l, c),
        "lnfw": (c,), "lnfb": (c,),
    }


def load_checkpoint(path: str) -> tuple[Config, dict[str, "xp.ndarray"]]:
    with open(path, "rb") as f:
        header = struct.unpack("<256i", f.read(HEADER_BYTES))
        if header[0] != MAGIC:
            raise ValueError(f"{path}: bad magic {header[0]}, expected {MAGIC}")
        if header[1] != VERSION:
            raise ValueError(f"{path}: unsupported version {header[1]}")
        config = Config(max_seq_len=header[2], vocab_size=header[3], num_layers=header[4],
                        num_heads=header[5], channels=header[6], padded_vocab_size=header[7])

        params = {}
        for name, shape in tensor_shapes(config).items():
            count = int(np.prod(shape))
            raw = f.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated while reading {name}")
            # copy: frombuffer answers a read-only view, and training writes
            # the parameters in place.
            host = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
            params[name] = xp.asarray(host, dtype=DT)
        if f.read(1):
            raise ValueError(f"{path}: trailing bytes after the parameters")
    return config, params


def load_tokenizer(path: str) -> tuple[int, list[bytes]]:
    """Returns (eot_token, table). Decoding only, like the Ruby side."""
    with open(path, "rb") as f:
        header = struct.unpack("<256I", f.read(HEADER_BYTES))
        if header[0] != TOKENIZER_MAGIC:
            raise ValueError(f"{path}: bad magic {header[0]}")
        version, vocab_size = header[1], header[2]
        eot = header[3] if version >= 2 else 50_256
        blob = f.read()
    table, offset = [], 0
    for _ in range(vocab_size):
        length = blob[offset]
        offset += 1
        table.append(blob[offset:offset + length])
        offset += length
    return eot, table


def decode_text(table: list[bytes], ids) -> str:
    return b"".join(table[int(i)] for i in ids).decode("utf-8", errors="replace")


# --- ops (mirrors lib/narray_llm/ops.rb) ---

def contiguous(a):
    return xp.ascontiguousarray(a)


def linear(x, weight_t, bias=None):
    y = x @ weight_t
    return y if bias is None else y + bias


def layernorm(x, weight, bias, eps=LAYERNORM_EPS):
    if FP16:
        # GPT-2's outliers put (x - mean) ** 2 past the fp16 ceiling, so fold
        # the statistics in fp32 the way Cumo's fused layer_norm does.
        mean = x.mean(axis=1, keepdims=True, dtype=xp.float32)
        variance = x.var(axis=1, keepdims=True, dtype=xp.float32)
        rstd = (1.0 / xp.sqrt(variance + eps)).astype(DT)
        return (x - mean.astype(DT)) * rstd * weight + bias

    mean = x.mean(axis=1, keepdims=True)
    centered = x - mean
    variance = (centered * centered).mean(axis=1, keepdims=True)
    rstd = 1.0 / xp.sqrt(variance + eps)
    return centered * rstd * weight + bias


def gelu(x):
    cube = 0.044715 * x * x * x
    return 0.5 * x * (1.0 + xp.tanh(GELU_SCALING_FACTOR * (x + cube)))


def softmax_rows(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    e = xp.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def causal_mask(t):
    """Mirrors the Ruby clip+ceil construction, which exists to avoid Bit
    arrays in Cumo. xp.triu would be the idiomatic form here and costs the
    same; the mask is built once at maxT either way."""
    row = xp.arange(t, dtype=xp.float32).reshape(t, 1)
    col = xp.arange(t, dtype=xp.float32).reshape(1, t)
    return (MASK_VALUE * xp.ceil(xp.clip(col - row, 0.0, 1.0))).astype(DT)


def one_hot(token_ids, num_classes):
    """Mirrors the Ruby arithmetic construction. xp.eye[ids] or a scatter would
    be idiomatic here; on the Ruby side an NArray index forces a device sync."""
    ids = xp.asarray(np.asarray(token_ids, dtype=np.float32).reshape(-1, 1))
    col = xp.arange(num_classes, dtype=xp.float32).reshape(1, num_classes)
    # Built in fp32: a token id past 2048 moves when it is held in fp16.
    return (1.0 - xp.ceil(xp.clip(xp.abs(col - ids), 0.0, 1.0))).astype(DT)


def attention_batched(qkv, batch_size, seq_len, num_heads, mask):
    """Every head as one entry of a stacked matmul. Mirrors the Ruby side's
    current attention: three 4-D transposed copies, then [B*NH, T, hs] products.

    Deliberately not einsum. einsum would express this in one line, but it also
    picks its own contraction order, and the point here is to run the same
    operations as the Ruby version."""
    channels = qkv.shape[1] // 3
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))
    packed = qkv.reshape(batch_size, seq_len, 3 * channels)

    blocks = []
    for block in range(3):
        part = packed[:, :, block * channels:(block + 1) * channels]
        part = part.reshape(batch_size, seq_len, num_heads, head_size).transpose(0, 2, 1, 3)
        blocks.append(contiguous(part).reshape(batch_size * num_heads, seq_len, head_size))
    queries, keys, values = blocks

    scores = queries @ keys.transpose(0, 2, 1) * scale + mask
    weights = softmax_rows(scores)
    out = weights @ values
    out = out.reshape(batch_size, num_heads, seq_len, head_size).transpose(0, 2, 1, 3)
    return contiguous(out).reshape(batch_size * seq_len, channels)


def decode_attention_batched(q, keys, values, num_heads):
    """All heads at once for a single query. Mirrors the Ruby side, which drops
    the matmul entirely here: per head the product is [1, hs] x [hs, t], too
    small to be worth a GEMM call, so broadcasting q over the cached rows and
    summing inside each head's channel block does the same arithmetic."""
    channels = q.shape[1]
    head_size = channels // num_heads
    length = keys.shape[0]
    scale = 1.0 / float(np.sqrt(head_size))

    scores = (keys * q).reshape(length, num_heads, head_size).sum(axis=2).T * scale
    weights = softmax_rows(scores)
    weighted = values.reshape(length, num_heads, head_size) * weights.T.reshape(length, num_heads, 1)
    return weighted.sum(axis=0).reshape(1, channels)


def attention_looped(qkv, batch_size, seq_len, num_heads, mask):
    """One [T, hs] x [hs, T] product per head, as the Ruby version did before
    the heads were batched."""
    channels = qkv.shape[1] // 3
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))
    packed = qkv.reshape(batch_size, seq_len, 3 * channels)
    out = xp.zeros((batch_size, seq_len, channels), dtype=DT)

    for b in range(batch_size):
        for h in range(num_heads):
            lo = h * head_size
            hi = lo + head_size
            q = contiguous(packed[b, :, lo:hi])
            k = contiguous(packed[b, :, channels + lo:channels + hi])
            v = contiguous(packed[b, :, 2 * channels + lo:2 * channels + hi])
            out[b, :, lo:hi] = softmax_rows(q @ k.T * scale + mask) @ v

    return out.reshape(batch_size * seq_len, channels)


def decode_attention_batch(q, keys, values, num_heads):
    """The same step for a batch. q is [B, C] and the cache views are [t, B, C],
    so the reduction axis stays 0 and softmax needs t moved last."""
    length, batch, channels = keys.shape
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))

    k4 = keys.reshape(length, batch, num_heads, head_size)
    q4 = q.reshape(1, batch, num_heads, head_size)
    scores = (k4 * q4).sum(axis=3).transpose(1, 2, 0) * scale
    weights = softmax_rows(scores)
    weighted = values.reshape(length, batch, num_heads, head_size) * \
        contiguous(weights.transpose(2, 0, 1)).reshape(length, batch, num_heads, 1)
    return weighted.sum(axis=0).reshape(batch, channels)


def decode_attention_looped(q, keys, values, num_heads):
    """One new query against the cache. No mask: the cache holds only the past."""
    channels = q.shape[1]
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))
    out = xp.zeros((1, channels), dtype=DT)

    for h in range(num_heads):
        lo = h * head_size
        hi = lo + head_size
        qh = contiguous(q[:, lo:hi])
        kh = contiguous(keys[:, lo:hi])
        vh = contiguous(values[:, lo:hi])
        out[:, lo:hi] = softmax_rows(qh @ kh.T * scale) @ vh

    return out


def _cache_rows(rows, batch_size, seq_len):
    """The block hands over [B*T, C] laid out batch major and the cache wants
    [T, B, C]. One sequence needs no move."""
    if batch_size == 1:
        return rows
    return contiguous(rows.reshape(batch_size, seq_len, rows.shape[1]).transpose(1, 0, 2))


attention = attention_batched if IDIOMATIC else attention_looped
decode_attention = decode_attention_batched if IDIOMATIC else decode_attention_looped


def transformer_block(x, w, batch_size, seq_len, num_heads, mask, kv_sink=None):
    ln1 = layernorm(x, w["ln1w"], w["ln1b"])
    qkv = linear(ln1, w["qkvw_t"], w["qkvb"])
    if kv_sink is not None:
        c = qkv.shape[1] // 3
        kv_sink(qkv[:, c:2 * c], qkv[:, 2 * c:3 * c])
    attn = attention(qkv, batch_size, seq_len, num_heads, mask)
    residual2 = x + linear(attn, w["attprojw_t"], w["attprojb"])

    ln2 = layernorm(residual2, w["ln2w"], w["ln2b"])
    hidden = gelu(linear(ln2, w["fcw_t"], w["fcb"]))
    return residual2 + linear(hidden, w["fcprojw_t"], w["fcprojb"])


class KVCache:
    """Time is the outer axis, as on the Ruby side: a growing slice of
    [maxT, B, C] stays contiguous whatever B is. With B of 1 the batch axis is
    dropped on the way out so the unbatched path sees the shapes it always had.
    """

    def __init__(self, num_layers, max_seq_len, channels, batch_size=1):
        if batch_size < 1:
            raise ValueError(f"batch size must be positive, got {batch_size}")
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.channels = channels
        self.batch_size = batch_size
        shape = (max_seq_len, batch_size, channels)
        self.keys = [xp.zeros(shape, dtype=DT) for _ in range(num_layers)]
        self.values = [xp.zeros(shape, dtype=DT) for _ in range(num_layers)]
        self.positions = [0] * num_layers

    @staticmethod
    def bytes_for(num_layers, max_seq_len, channels, batch_size=1):
        return 2 * batch_size * num_layers * max_seq_len * channels * 4

    def reset(self):
        self.positions = [0] * self.num_layers

    def _as_block(self, tensor):
        if tensor.ndim == 3:
            return tensor
        if self.batch_size > 1:
            if tensor.ndim == 2 and tensor.shape[0] == self.batch_size:
                return tensor.reshape(1, self.batch_size, self.channels)
            raise ValueError(f"batched append needs [{self.batch_size}, C] or "
                             f"[n, {self.batch_size}, C], got {tensor.shape}")
        rows = tensor.shape[0] if tensor.ndim > 1 else 1
        return tensor.reshape(rows, 1, self.channels)

    def append(self, layer, keys, values):
        k = self._as_block(keys)
        v = self._as_block(values)
        rows = k.shape[0]
        position = self.positions[layer]
        if position + rows > self.max_seq_len:
            raise ValueError(f"kv cache overflow on layer {layer}: {position} + {rows} rows "
                             f"exceeds max_seq_len {self.max_seq_len}")
        self.keys[layer][position:position + rows] = k
        self.values[layer][position:position + rows] = v
        self.positions[layer] = position + rows

    def view(self, layer):
        length = self.positions[layer]
        if self.batch_size == 1:
            return self.keys[layer][:length, 0], self.values[layer][:length, 0]
        return self.keys[layer][:length], self.values[layer][:length]


class Model:
    def __init__(self, checkpoint_path):
        self.config, params = load_checkpoint(checkpoint_path)
        c = self.config
        # llm.c keeps Vp rows in wte but never uses the padding.
        self.wte = contiguous(params["wte"][:c.vocab_size])
        self.wte_t = contiguous(self.wte.T)
        self.wpe = params["wpe"]
        self.lnfw = params["lnfw"]
        self.lnfb = params["lnfb"]
        # Weights are stored [out, in]; every matmul wants [in, out].
        self.layers = [{
            "ln1w": contiguous(params["ln1w"][i]), "ln1b": contiguous(params["ln1b"][i]),
            "qkvw_t": contiguous(params["qkvw"][i].T), "qkvb": contiguous(params["qkvb"][i]),
            "attprojw_t": contiguous(params["attprojw"][i].T),
            "attprojb": contiguous(params["attprojb"][i]),
            "ln2w": contiguous(params["ln2w"][i]), "ln2b": contiguous(params["ln2b"][i]),
            "fcw_t": contiguous(params["fcw"][i].T), "fcb": contiguous(params["fcb"][i]),
            "fcprojw_t": contiguous(params["fcprojw"][i].T),
            "fcprojb": contiguous(params["fcprojb"][i]),
        } for i in range(c.num_layers)]
        self._full_mask = None

    def parameter_bytes(self):
        tensors = [self.wte, self.wte_t, self.wpe, self.lnfw, self.lnfb]
        tensors += [t for layer in self.layers for t in layer.values()]
        return 4 * sum(int(t.size) for t in tensors)

    def new_cache(self, batch_size=1):
        c = self.config
        return KVCache(c.num_layers, c.max_seq_len, c.channels, batch_size=batch_size)

    def _causal_mask(self, seq_len):
        if self._full_mask is None:
            self._full_mask = causal_mask(self.config.max_seq_len)
        return contiguous(self._full_mask[:seq_len, :seq_len])

    def _embed(self, ids, batch_size, seq_len):
        c = self.config
        tokens = one_hot(np.asarray(ids).reshape(-1), c.vocab_size) @ self.wte
        positions = self.wpe[:seq_len]
        return (tokens.reshape(batch_size, seq_len, c.channels) + positions).reshape(
            batch_size * seq_len, c.channels)

    def forward(self, tokens, last_only=False, cache=None):
        ids = np.asarray(tokens, dtype=np.int32)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)
        batch_size, seq_len = ids.shape
        c = self.config
        if seq_len > c.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {c.max_seq_len}")
        if cache is not None and cache.batch_size != batch_size:
            raise ValueError(f"a cache of batch size {cache.batch_size} cannot take "
                             f"{batch_size} rows")

        x = self._embed(ids, batch_size, seq_len)
        mask = self._causal_mask(seq_len)
        for layer, w in enumerate(self.layers):
            sink = None
            if cache is not None:
                sink = (lambda k, v, _l=layer: cache.append(
                    _l, _cache_rows(k, batch_size, seq_len),
                    _cache_rows(v, batch_size, seq_len)))
            x = transformer_block(x, w, batch_size, seq_len, c.num_heads, mask, kv_sink=sink)

        x = layernorm(x, self.lnfw, self.lnfb)
        if last_only:
            x = contiguous(x.reshape(batch_size, seq_len, c.channels)[:, seq_len - 1, :])
        logits = x @ self.wte_t
        return logits.reshape(batch_size, 1 if last_only else seq_len, c.vocab_size)

    def prefill(self, tokens, cache):
        cache.reset()
        return self.forward(tokens, last_only=True, cache=cache)

    def decode(self, token_id, position, cache):
        """One id, or one per sequence for a batched cache. Every sequence sits
        at the same position, as on the Ruby side."""
        c = self.config
        if position >= c.max_seq_len:
            raise ValueError(f"position {position} exceeds max_seq_len {c.max_seq_len}")
        ids = list(token_id) if isinstance(token_id, (list, tuple)) else [token_id]
        if len(ids) != cache.batch_size:
            raise ValueError(f"got {len(ids)} token ids for a cache of batch size "
                             f"{cache.batch_size}")
        for one in ids:
            if not 0 <= one < c.vocab_size:
                raise ValueError(f"invalid token id {one}")

        # Python int indices, so no device-side gather (mirrors the Ruby side).
        if len(ids) == 1:
            x = (self.wte[ids[0]] + self.wpe[position]).reshape(1, c.channels)
        else:
            x = contiguous(self.wte[ids]) + self.wpe[position]
        for layer, w in enumerate(self.layers):
            x = self._decode_block(x, w, layer, cache)
        x = layernorm(x, self.lnfw, self.lnfb)
        return (x @ self.wte_t).reshape(len(ids), 1, c.vocab_size)

    def _decode_block(self, x, w, layer, cache):
        c = self.config.channels
        ln1 = layernorm(x, w["ln1w"], w["ln1b"])
        qkv = linear(ln1, w["qkvw_t"], w["qkvb"])
        cache.append(layer, qkv[:, c:2 * c], qkv[:, 2 * c:3 * c])
        keys, values = cache.view(layer)
        step = decode_attention_batch if cache.batch_size > 1 else decode_attention
        attn = step(contiguous(qkv[:, 0:c]), keys, values, self.config.num_heads)
        residual2 = x + linear(attn, w["attprojw_t"], w["attprojb"])

        ln2 = layernorm(residual2, w["ln2w"], w["ln2b"])
        hidden = gelu(linear(ln2, w["fcw_t"], w["fcb"]))
        return residual2 + linear(hidden, w["fcprojw_t"], w["fcprojb"])


class Generator:
    def __init__(self, model, eot_token=None):
        self.model = model
        self.eot_token = eot_token

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

    def generate_batch(self, prompts, max_new_tokens, stop_at_eot=True):
        """B sequences at once, prompts included. Mirrors the Ruby side: one
        position for the whole batch, so every prompt has to be the same length,
        and a finished sequence keeps stepping and is cut afterwards."""
        rows = [[int(t) for t in prompt] for prompt in prompts]
        if not rows:
            raise ValueError("a batch needs at least one prompt")
        if any(not row for row in rows):
            raise ValueError("every prompt must contain at least one token")
        lengths = {len(row) for row in rows}
        if len(lengths) != 1:
            raise ValueError(f"every prompt must be the same length, got "
                             f"{[len(row) for row in rows]}")

        length = lengths.pop()
        limit = self.model.config.max_seq_len
        if length + max_new_tokens > limit:
            raise ValueError(f"{length} prompt tokens + {max_new_tokens} generated tokens "
                             f"exceeds max_seq_len {limit}")

        cache = self.model.new_cache(batch_size=len(rows))
        logits = self.model.prefill([row[:] for row in rows], cache)
        produced = [[] for _ in rows]
        for step in range(max_new_tokens):
            ids = self._argmax_rows(logits, len(rows))
            for i, one in enumerate(ids):
                produced[i].append(one)
            if step == max_new_tokens - 1:
                break
            if stop_at_eot and self.eot_token is not None and \
                    all(self.eot_token in seq for seq in produced):
                break
            logits = self.model.decode(ids, length + len(produced[0]) - 1, cache)

        return [row + self._cut_at_eot(produced[i], stop_at_eot)
                for i, row in enumerate(rows)]

    def _cut_at_eot(self, sequence, stop_at_eot):
        if not stop_at_eot or self.eot_token is None or self.eot_token not in sequence:
            return sequence
        return sequence[:sequence.index(self.eot_token) + 1]

    def _argmax_rows(self, logits, batch_size):
        rows = logits.reshape(batch_size, self.model.config.vocab_size)
        # One readback per step for the whole batch, not one per sequence:
        # iterating the device array instead would call int() on each element
        # and cost B transfers.
        return rows.argmax(axis=1).get().tolist() if GPU else \
            rows.argmax(axis=1).tolist()

    def _argmax(self, logits):
        row = logits.reshape(self.model.config.vocab_size)
        # The one readback per generated token.
        return int(row.argmax())

    def _generate_recomputing(self, tokens, max_new_tokens, stop_at_eot):
        for _ in range(max_new_tokens):
            token = self._argmax(self.model.forward([tokens], last_only=True))
            tokens.append(token)
            if stop_at_eot and self.eot_token is not None and token == self.eot_token:
                break
        return tokens

    def _generate_cached(self, tokens, max_new_tokens, stop_at_eot):
        cache = self.model.new_cache()
        logits = self.model.prefill([tokens], cache)
        produced = 0
        while produced < max_new_tokens:
            token = self._argmax(logits)
            tokens.append(token)
            produced += 1
            if stop_at_eot and self.eot_token is not None and token == self.eot_token:
                break
            if produced == max_new_tokens:
                break
            logits = self.model.decode(token, len(tokens) - 1, cache)
        return tokens


def synchronize():
    if GPU:
        xp.cuda.Stream.null.synchronize()
