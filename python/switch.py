"""Switch Transformer base-8 over the safetensors python/export_switch.py writes.

NumPy on the CPU, CuPy on the GPU with GPU=1, the same switch the other ports
in this directory use.

Written out to match lib/narray_llm/models/switch/, not to be idiomatic: the
same per head loop, the same two ways of spending a sparse layer, the same
order of operations. The other tables in this repository hold the algorithm
fixed and vary only what runs it, and this one does the same. transformers is
not called here; it is the reference the fixtures came from.

  python/.venv/bin/python python/bench_switch.py
  GPU=1 python/.venv/bin/python python/bench_switch.py --router dense
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")

if GPU:
    import cupy as xp
else:
    import numpy as xp

import numpy as np  # host side ids and buckets, always NumPy (mirrors Ruby's HM)
from safetensors.numpy import load_file

SHARED = "shared.weight"
ROUTERS = ("dispatch", "dense")


@dataclass(frozen=True)
class Config:
    num_layers: int
    num_decoder_layers: int
    d_model: int
    d_ff: int
    d_kv: int
    num_heads: int
    num_experts: int
    expert_capacity: int
    encoder_sparse_step: int
    decoder_sparse_step: int
    relative_attention_num_buckets: int
    relative_attention_max_distance: int
    layer_norm_epsilon: float
    vocab_size: int
    pad_token_id: int
    eos_token_id: int
    decoder_start_token_id: int

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.d_kv

    def sparse_encoder_layer(self, block: int) -> bool:
        return block % self.encoder_sparse_step == 1

    def sparse_decoder_layer(self, block: int) -> bool:
        return block % self.decoder_sparse_step == 1


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = json.load(f)
    fields = Config.__dataclass_fields__
    return Config(**{k: raw[k] for k in fields})


def buckets(query_length, key_length, config, bidirectional, offset=0):
    """_relative_position_bucket. The three factors are multiplied in the
    reference's order: folding the divide and the scale into one constant puts
    a distance of exactly 64 one bucket down."""
    total = config.relative_attention_num_buckets
    half = total // 2 if bidirectional else total
    exact = half // 2
    span = math.log(config.relative_attention_max_distance / exact)
    out = np.empty(query_length * key_length, dtype=np.int64)
    for q in range(query_length):
        for k in range(key_length):
            distance = k - (q + offset)
            bucket = 0
            if bidirectional:
                if distance > 0:
                    bucket = half
                distance = abs(distance)
            else:
                distance = -min(distance, 0)
            if distance < exact:
                bucket += distance
            else:
                large = int(math.log(distance / exact) / span * (half - exact))
                bucket += min(exact + large, half - 1)
            out[q * key_length + k] = bucket
    return out


def rmsnorm(x, weight, eps):
    """SwitchTransformersLayerNorm: mean of squares, no centring, no bias."""
    variance = xp.mean(x * x, axis=-1, keepdims=True)
    return weight * (x * (np.float32(1.0) / xp.sqrt(variance + eps)))


def softmax_rows(x):
    shifted = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(shifted)
    return e / xp.sum(e, axis=-1, keepdims=True)


def relu(x):
    return xp.maximum(x, np.float32(0.0))


def transposed(host):
    """Stored [out, in]; every matmul here wants [in, out]. The copy is made on
    the host, because the store is NumPy whichever backend runs the model."""
    return xp.asarray(np.ascontiguousarray(host.T))


class Stack:
    def __init__(self, store, config, prefix, router, expert_capacity):
        if router not in ROUTERS:
            raise ValueError(f"unknown router {router}")
        self.config = config
        self.prefix = prefix
        self.router = router
        self.expert_capacity = expert_capacity
        self.before = xp.asarray(np.triu(np.ones((config.num_experts, config.num_experts),
                                                  dtype=np.float32), 1))
        self.embedding = xp.asarray(store[SHARED])
        self.final_norm = xp.asarray(store[f"{prefix}.final_layer_norm.weight"])
        self.bias_table = xp.asarray(
            store[f"{prefix}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"])

    def bias_from(self, query_length, key_length, bidirectional, offset=0):
        index = buckets(query_length, key_length, self.config, bidirectional, offset)
        rows = self.bias_table[index]
        return xp.ascontiguousarray(
            rows.reshape(query_length, key_length, self.config.num_heads).transpose(2, 0, 1))

    def attend(self, h, w, prefix, keys, values, bias):
        """keys and values come in head major as [heads, keys, d_kv], so each
        head is a contiguous slice rather than a column span that has to be
        copied. No scaling: Switch folds the relative bias in and leaves the
        query-key product as it is."""
        q = self.heads_major(h @ w[f"{prefix}q_t"])
        parts = []
        for head in range(self.config.num_heads):
            scores = q[head] @ keys[head].T + bias[head]
            parts.append(softmax_rows(scores) @ values[head])
        return xp.ascontiguousarray(xp.hstack(parts)) @ w[f"{prefix}o_t"]

    def heads_major(self, x):
        return xp.ascontiguousarray(
            x.reshape(x.shape[0], self.config.num_heads, self.config.d_kv).transpose(1, 0, 2))

    def project_kv(self, x, w, prefix):
        return x @ w[f"{prefix}k_t"], x @ w[f"{prefix}v_t"]

    def project_kv_heads(self, x, w, prefix):
        return [self.heads_major(a) for a in self.project_kv(x, w, prefix)]

    def feed_forward(self, h, w):
        if "experts" not in w:
            return self.dense(h, w)
        probs = softmax_rows(h @ w["router_t"])
        if self.router == "dense":
            return self.mixture_dense(h, w, probs)
        return self.mixture_dispatch(h, w, probs)

    def dense(self, h, w):
        return relu(h @ w["wi_t"]) @ w["wo_t"]

    def mixture_dispatch(self, h, w, probs):
        tokens = h.shape[0]
        gate = xp.max(probs, axis=1, keepdims=True)
        # The one readback: the expert numbers have to reach the host before
        # they can be used as subscripts.
        chosen = xp.asnumpy(xp.argmax(probs, axis=1)) if GPU else np.argmax(probs, axis=1)
        out = xp.zeros_like(h)
        seats = [[] for _ in range(self.config.num_experts)]
        for token, expert in enumerate(chosen):
            bucket = seats[int(expert)]
            if self.expert_capacity is None or len(bucket) < self.expert_capacity:
                bucket.append(token)
        for expert, rows in enumerate(seats):
            if not rows:
                continue
            index = np.asarray(rows)
            out[index] = self.dense(h[index], w["experts"][expert]) * gate[index]
        return out

    def mixture_dense(self, h, w, probs):
        top = xp.max(probs, axis=1, keepdims=True)
        selected = self.first_maximum(probs, top)
        within = self.capacity_mask(selected)
        weights = selected * within * top
        out = xp.zeros_like(h)
        for expert in range(self.config.num_experts):
            out = out + self.dense(h, w["experts"][expert]) * weights[:, expert:expert + 1]
        return out

    def first_maximum(self, probs, top):
        """1 on the expert holding the row maximum, and on the first one when
        several tie. argmax takes the first, so a mask that kept every tie
        would send a token to more than one expert.

        The running count comes from a matmul with a strictly lower triangular
        block of ones, matching the Ruby side, where cumsum synchronizes."""
        hit = np.float32(1.0) - xp.ceil(xp.clip(xp.abs(probs - top), 0.0, 1.0))
        return hit * (np.float32(1.0) - xp.ceil(xp.clip(hit @ self.before, 0.0, 1.0)))

    def capacity_mask(self, selected):
        if self.expert_capacity is None:
            return xp.ones_like(selected)
        return np.float32(1.0) - xp.ceil(
            xp.clip(xp.cumsum(selected, axis=0) - self.expert_capacity, 0.0, 1.0))

    def attention_weights(self, store, name, prefix, kind):
        return {f"{prefix}{p}_t": transposed(store[f"{name}.{kind}.{p}.weight"])
                for p in ("q", "k", "v", "o")}

    def feed_forward_weights(self, store, prefix, sparse):
        def t(name):
            return transposed(store[f"{prefix}.{name}"])

        if not sparse:
            return {"wi_t": t("wi.weight"), "wo_t": t("wo.weight")}
        return {"router_t": t("router.classifier.weight"),
                "experts": [{"wi_t": t(f"experts.expert_{e}.wi.weight"),
                             "wo_t": t(f"experts.expert_{e}.wo.weight")}
                            for e in range(self.config.num_experts)]}


class Encoder(Stack):
    def __init__(self, store, config, router, expert_capacity):
        super().__init__(store, config, "encoder", router, expert_capacity)
        self.blocks = [self.block_weights(store, b) for b in range(config.num_layers)]

    def forward(self, ids):
        x = xp.ascontiguousarray(self.embedding[np.asarray(ids)])
        bias = self.bias_from(len(ids), len(ids), True)
        for w in self.blocks:
            h = rmsnorm(x, w["attention_norm"], self.config.layer_norm_epsilon)
            keys, values = self.project_kv_heads(h, w, "")
            x = x + self.attend(h, w, "", keys, values, bias)
            h = rmsnorm(x, w["ff_norm"], self.config.layer_norm_epsilon)
            x = x + self.feed_forward(h, w)
        return rmsnorm(x, self.final_norm, self.config.layer_norm_epsilon)

    def block_weights(self, store, block):
        name = f"encoder.block.{block}"
        w = self.attention_weights(store, f"{name}.layer.0", "", "SelfAttention")
        w["attention_norm"] = xp.asarray(store[f"{name}.layer.0.layer_norm.weight"])
        w["ff_norm"] = xp.asarray(store[f"{name}.layer.1.layer_norm.weight"])
        w.update(self.feed_forward_weights(store, f"{name}.layer.1.mlp",
                                           self.config.sparse_encoder_layer(block)))
        return w


class Decoder(Stack):
    def __init__(self, store, config, router, expert_capacity):
        super().__init__(store, config, "decoder", router, expert_capacity)
        self.blocks = [self.block_weights(store, b) for b in range(config.num_decoder_layers)]

    def new_cache(self, encoder_states, max_seq_len):
        cross = [self.project_kv_heads(encoder_states, w, "cross_") for w in self.blocks]
        keys = [xp.zeros((max_seq_len, self.config.inner_dim), dtype=xp.float32)
                for _ in self.blocks]
        values = [xp.zeros((max_seq_len, self.config.inner_dim), dtype=xp.float32)
                  for _ in self.blocks]
        return {"cross": cross, "keys": keys, "values": values, "length": 0}

    def decode(self, token_id, cache):
        position = cache["length"]
        x = xp.ascontiguousarray(self.embedding[np.asarray([token_id])])
        bias = self.bias_from(1, position + 1, False, position)
        zero = xp.zeros((self.config.num_heads, 1, 1), dtype=xp.float32)
        for i, w in enumerate(self.blocks):
            h = rmsnorm(x, w["attention_norm"], self.config.layer_norm_epsilon)
            keys, values = self.project_kv(h, w, "")
            cache["keys"][i][position] = keys[0]
            cache["values"][i][position] = values[0]
            span = slice(0, position + 1)
            x = x + self.attend(h, w, "", self.heads_major(cache["keys"][i][span]),
                                self.heads_major(cache["values"][i][span]), bias)

            h = rmsnorm(x, w["cross_norm"], self.config.layer_norm_epsilon)
            cross_keys, cross_values = cache["cross"][i]
            x = x + self.attend(h, w, "cross_", cross_keys, cross_values, zero)

            h = rmsnorm(x, w["ff_norm"], self.config.layer_norm_epsilon)
            x = x + self.feed_forward(h, w)
        cache["length"] = position + 1
        return rmsnorm(x, self.final_norm, self.config.layer_norm_epsilon)

    def block_weights(self, store, block):
        name = f"decoder.block.{block}"
        w = self.attention_weights(store, f"{name}.layer.0", "", "SelfAttention")
        w.update(self.attention_weights(store, f"{name}.layer.1", "cross_", "EncDecAttention"))
        w["attention_norm"] = xp.asarray(store[f"{name}.layer.0.layer_norm.weight"])
        w["cross_norm"] = xp.asarray(store[f"{name}.layer.1.layer_norm.weight"])
        w["ff_norm"] = xp.asarray(store[f"{name}.layer.2.layer_norm.weight"])
        w.update(self.feed_forward_weights(store, f"{name}.layer.2.mlp",
                                           self.config.sparse_decoder_layer(block)))
        return w


class Model:
    def __init__(self, path, config_path=None, router="dispatch", expert_capacity=None):
        config_path = config_path or path.replace(".safetensors", "") + "/config.json"
        self.config = load_config(config_path)
        store = load_file(path)
        self.encoder = Encoder(store, self.config, router, expert_capacity)
        self.decoder = Decoder(store, self.config, router, expert_capacity)
        self.lm_head_t = transposed(store[SHARED])
        # The embedding is tied to the classifier, and the reference scales by
        # d_model ** -0.5 exactly when it is.
        self.scale = np.float32(self.config.d_model ** -0.5)

    def encode(self, ids):
        return self.encoder.forward(ids)

    def decode(self, token_id, cache):
        x = self.decoder.decode(token_id, cache)
        return (x * self.scale) @ self.lm_head_t

    def generate(self, ids, max_new_tokens, stop_at_eos=True):
        states = self.encode(ids)
        cache = self.decoder.new_cache(states, max_new_tokens + 1)
        tokens = [self.config.decoder_start_token_id]
        for _ in range(max_new_tokens):
            logits = self.decode(tokens[-1], cache)
            token = int(xp.argmax(logits[0]))
            tokens.append(token)
            if stop_at_eos and token == self.config.eos_token_id:
                break
        return tokens


def synchronize():
    if GPU:
        xp.cuda.Stream.null.synchronize()
