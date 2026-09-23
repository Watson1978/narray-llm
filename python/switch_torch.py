"""Switch Transformer base-8 in PyTorch, over the same safetensors.

Only the backend differs from switch.py. transformers' own implementation is
deliberately not used: it would make this a comparison of recipes rather than
of backends, and every other table in this repository holds the algorithm
fixed. Where PyTorch has a native op for what the Ruby side spells out, it is
used, the way llama2_torch.py uses F.rms_norm and F.silu.

  python/.venv/bin/python python/bench_switch.py --impl torch
  GPU=1 python/.venv/bin/python python/bench_switch.py --impl torch --router dense
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from switch import ROUTERS, SHARED, buckets, load_config

GPU = os.environ.get("GPU", "").lower() in ("1", "on", "true")
DEVICE = torch.device("cuda" if GPU else "cpu")


def rmsnorm(x, weight, eps):
    # T5's layer norm is rms_norm with no bias, which is what F.rms_norm is.
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


class Stack:
    def __init__(self, store, config, prefix, router, expert_capacity, device):
        if router not in ROUTERS:
            raise ValueError(f"unknown router {router}")
        self.config = config
        self.router = router
        self.expert_capacity = expert_capacity
        self.device = device
        self.before = torch.triu(torch.ones((config.num_experts, config.num_experts),
                                            device=device), 1)
        self.embedding = store[SHARED].to(device)
        self.final_norm = store[f"{prefix}.final_layer_norm.weight"].to(device)
        self.bias_table = store[
            f"{prefix}.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].to(device)

    def bias_from(self, query_length, key_length, bidirectional, offset=0):
        index = torch.from_numpy(
            buckets(query_length, key_length, self.config, bidirectional, offset)).to(self.device)
        rows = self.bias_table[index]
        return rows.reshape(query_length, key_length, self.config.num_heads) \
                   .permute(2, 0, 1).contiguous()

    def attend(self, h, w, prefix, keys, values, bias):
        q = self.heads_major(h @ w[f"{prefix}q_t"])
        parts = []
        for head in range(self.config.num_heads):
            scores = q[head] @ keys[head].T + bias[head]
            parts.append(torch.softmax(scores, dim=-1) @ values[head])
        return torch.hstack(parts).contiguous() @ w[f"{prefix}o_t"]

    def heads_major(self, x):
        return x.reshape(x.shape[0], self.config.num_heads, self.config.d_kv) \
                .permute(1, 0, 2).contiguous()

    def project_kv(self, x, w, prefix):
        return x @ w[f"{prefix}k_t"], x @ w[f"{prefix}v_t"]

    def project_kv_heads(self, x, w, prefix):
        return [self.heads_major(a) for a in self.project_kv(x, w, prefix)]

    def feed_forward(self, h, w):
        if "experts" not in w:
            return self.dense(h, w)
        probs = torch.softmax(h @ w["router_t"], dim=-1)
        if self.router == "dense":
            return self.mixture_dense(h, w, probs)
        return self.mixture_dispatch(h, w, probs)

    def dense(self, h, w):
        return F.relu(h @ w["wi_t"]) @ w["wo_t"]

    def mixture_dispatch(self, h, w, probs):
        gate = probs.max(dim=1, keepdim=True).values
        # The one readback per sparse layer.
        chosen = probs.argmax(dim=1).tolist()
        out = torch.zeros_like(h)
        seats = [[] for _ in range(self.config.num_experts)]
        for token, expert in enumerate(chosen):
            bucket = seats[expert]
            if self.expert_capacity is None or len(bucket) < self.expert_capacity:
                bucket.append(token)
        for expert, rows in enumerate(seats):
            if not rows:
                continue
            index = torch.tensor(rows, device=h.device)
            out[index] = self.dense(h[index], w["experts"][expert]) * gate[index]
        return out

    def mixture_dense(self, h, w, probs):
        top = probs.max(dim=1, keepdim=True).values
        selected = self.first_maximum(probs, top)
        within = self.capacity_mask(selected)
        weights = selected * within * top
        out = torch.zeros_like(h)
        for expert in range(self.config.num_experts):
            out = out + self.dense(h, w["experts"][expert]) * weights[:, expert:expert + 1]
        return out

    def first_maximum(self, probs, top):
        """1 on the expert holding the row maximum, and on the first one when
        several tie. argmax takes the first, so a mask that kept every tie
        would send a token to more than one expert.

        The running count comes from a matmul with a strictly lower triangular
        block of ones, matching the Ruby side, where cumsum synchronizes."""
        hit = 1.0 - torch.ceil(torch.clamp((probs - top).abs(), 0.0, 1.0))
        return hit * (1.0 - torch.ceil(torch.clamp(hit @ self.before, 0.0, 1.0)))

    def capacity_mask(self, selected):
        if self.expert_capacity is None:
            return torch.ones_like(selected)
        return 1.0 - torch.ceil(
            torch.clamp(selected.cumsum(dim=0) - self.expert_capacity, 0.0, 1.0))

    def attention_weights(self, store, name, prefix, kind):
        return {f"{prefix}{p}_t": store[f"{name}.{kind}.{p}.weight"].T.contiguous().to(self.device)
                for p in ("q", "k", "v", "o")}

    def feed_forward_weights(self, store, prefix, sparse):
        def t(name):
            return store[f"{prefix}.{name}"].T.contiguous().to(self.device)

        if not sparse:
            return {"wi_t": t("wi.weight"), "wo_t": t("wo.weight")}
        return {"router_t": t("router.classifier.weight"),
                "experts": [{"wi_t": t(f"experts.expert_{e}.wi.weight"),
                             "wo_t": t(f"experts.expert_{e}.wo.weight")}
                            for e in range(self.config.num_experts)]}


class Encoder(Stack):
    def __init__(self, store, config, router, expert_capacity, device):
        super().__init__(store, config, "encoder", router, expert_capacity, device)
        self.blocks = [self.block_weights(store, b) for b in range(config.num_layers)]

    def forward(self, ids):
        x = self.embedding[torch.tensor(ids, device=self.device)].contiguous()
        bias = self.bias_from(len(ids), len(ids), True)
        eps = self.config.layer_norm_epsilon
        for w in self.blocks:
            h = rmsnorm(x, w["attention_norm"], eps)
            keys, values = self.project_kv_heads(h, w, "")
            x = x + self.attend(h, w, "", keys, values, bias)
            h = rmsnorm(x, w["ff_norm"], eps)
            x = x + self.feed_forward(h, w)
        return rmsnorm(x, self.final_norm, eps)

    def block_weights(self, store, block):
        name = f"encoder.block.{block}"
        w = self.attention_weights(store, f"{name}.layer.0", "", "SelfAttention")
        w["attention_norm"] = store[f"{name}.layer.0.layer_norm.weight"].to(self.device)
        w["ff_norm"] = store[f"{name}.layer.1.layer_norm.weight"].to(self.device)
        w.update(self.feed_forward_weights(store, f"{name}.layer.1.mlp",
                                           self.config.sparse_encoder_layer(block)))
        return w


class Decoder(Stack):
    def __init__(self, store, config, router, expert_capacity, device):
        super().__init__(store, config, "decoder", router, expert_capacity, device)
        self.blocks = [self.block_weights(store, b) for b in range(config.num_decoder_layers)]

    def new_cache(self, encoder_states, max_seq_len):
        cross = [self.project_kv_heads(encoder_states, w, "cross_") for w in self.blocks]
        shape = (max_seq_len, self.config.inner_dim)
        return {"cross": cross,
                "keys": [torch.zeros(shape, device=self.device) for _ in self.blocks],
                "values": [torch.zeros(shape, device=self.device) for _ in self.blocks],
                "length": 0}

    def decode(self, token_id, cache):
        position = cache["length"]
        eps = self.config.layer_norm_epsilon
        x = self.embedding[torch.tensor([token_id], device=self.device)].contiguous()
        bias = self.bias_from(1, position + 1, False, position)
        zero = torch.zeros((self.config.num_heads, 1, 1), device=self.device)
        for i, w in enumerate(self.blocks):
            h = rmsnorm(x, w["attention_norm"], eps)
            keys, values = self.project_kv(h, w, "")
            cache["keys"][i][position] = keys[0]
            cache["values"][i][position] = values[0]
            span = slice(0, position + 1)
            x = x + self.attend(h, w, "", self.heads_major(cache["keys"][i][span]),
                                self.heads_major(cache["values"][i][span]), bias)

            h = rmsnorm(x, w["cross_norm"], eps)
            cross_keys, cross_values = cache["cross"][i]
            x = x + self.attend(h, w, "cross_", cross_keys, cross_values, zero)

            h = rmsnorm(x, w["ff_norm"], eps)
            x = x + self.feed_forward(h, w)
        cache["length"] = position + 1
        return rmsnorm(x, self.final_norm, eps)

    def block_weights(self, store, block):
        name = f"decoder.block.{block}"
        w = self.attention_weights(store, f"{name}.layer.0", "", "SelfAttention")
        w.update(self.attention_weights(store, f"{name}.layer.1", "cross_", "EncDecAttention"))
        w["attention_norm"] = store[f"{name}.layer.0.layer_norm.weight"].to(self.device)
        w["cross_norm"] = store[f"{name}.layer.1.layer_norm.weight"].to(self.device)
        w["ff_norm"] = store[f"{name}.layer.2.layer_norm.weight"].to(self.device)
        w.update(self.feed_forward_weights(store, f"{name}.layer.2.mlp",
                                           self.config.sparse_decoder_layer(block)))
        return w


class Model:
    def __init__(self, path, config_path=None, router="dispatch", expert_capacity=None,
                 device=DEVICE):
        config_path = config_path or path.replace(".safetensors", "") + "/config.json"
        self.config = load_config(config_path)
        store = load_file(path)
        self.encoder = Encoder(store, self.config, router, expert_capacity, device)
        self.decoder = Decoder(store, self.config, router, expert_capacity, device)
        self.lm_head_t = store[SHARED].T.contiguous().to(device)
        self.scale = np.float32(self.config.d_model ** -0.5).item()

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
            token = int(self.decode(tokens[-1], cache)[0].argmax())
            tokens.append(token)
            if stop_at_eos and token == self.config.eos_token_id:
                break
        return tokens


def synchronize():
    if GPU:
        torch.cuda.synchronize()
