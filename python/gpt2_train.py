"""Backward and AdamW for the numpy/cupy port, mirroring the Ruby side.

One function per *_backward in llm.c's train_gpt2.c, answering gradients rather
than accumulating into buffers, exactly as lib/narray_llm/backward.rb does. The
point of the port is to run the same operations, so nothing here reaches for an
autograd or a fused kernel.

The debug state reader is here too: llm.c's gpt2_*_debug_state.bin carries the
inputs, the targets and the gradient for every parameter, which is the gate both
sides are checked against.
"""

from __future__ import annotations

import struct

import numpy as np

import gpt2
from gpt2 import DT, contiguous, tensor_shapes, xp

GELU_SCALING_FACTOR = float(np.sqrt(2.0 / np.pi))
LAYERNORM_EPS = 1e-5
DEBUG_MAGIC = 20240327
DEBUG_VERSION = 2
HEADER_BYTES = 256 * 4


def one_hot(ids, num_classes):
    """1.0 where the id matches the column, built with arithmetic so no index
    array reaches the device (AGENTS.md)."""
    rows = np.asarray(ids, dtype=np.int64).reshape(-1)
    column = xp.arange(num_classes, dtype=DT).reshape(1, num_classes)
    identifiers = xp.asarray(rows.astype(np.float32), dtype=DT).reshape(rows.size, 1)
    return 1.0 - xp.ceil(xp.clip(xp.abs(identifiers - column), 0.0, 1.0))


def residual_backward(dout):
    return dout, dout


def gelu_backward(dout, inp):
    cube = 0.044715 * inp * inp * inp
    arg = GELU_SCALING_FACTOR * (inp + cube)
    tanh_out = xp.tanh(arg)
    sech2 = 1.0 - tanh_out * tanh_out
    local = 0.5 * (1.0 + tanh_out) + inp * 0.5 * sech2 * GELU_SCALING_FACTOR * (
        1.0 + 3.0 * 0.044715 * inp * inp)
    return local * dout


def layernorm_backward(dout, inp, weight, eps=LAYERNORM_EPS):
    mean = inp.mean(axis=1, keepdims=True)
    centered = inp - mean
    variance = (centered * centered).mean(axis=1, keepdims=True)
    rstd = 1.0 / xp.sqrt(variance + eps)
    norm = centered * rstd

    dnorm = weight * dout
    dnorm_mean = dnorm.mean(axis=1, keepdims=True)
    dnorm_norm_mean = (dnorm * norm).mean(axis=1, keepdims=True)
    dinp = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd
    return dinp, (norm * dout).sum(axis=0), dout.sum(axis=0)


def matmul_backward(dout, inp, weight_t, bias=True):
    """weight_t is [C, OC] where llm.c's weight is [OC, C]."""
    dinp = dout @ contiguous(weight_t.T)
    dweight_t = contiguous(inp.T) @ dout
    return dinp, dweight_t, (dout.sum(axis=0) if bias else None)


def crossentropy_softmax_backward(probs, targets, dloss=None):
    rows = probs.shape[0]
    scale = dloss if dloss is not None else 1.0 / rows
    return (probs - one_hot(targets, probs.shape[1])) * scale


def encoder_backward(dout, ids, vocab_size, batch_size, seq_len):
    channels = dout.shape[1]
    hot = one_hot(ids, vocab_size)
    dwte = contiguous(hot.T) @ dout
    dwpe = dout.reshape(batch_size, seq_len, channels).sum(axis=0)
    return dwte, dwpe


def _to_heads(flat, batch_size, seq_len, num_heads, head_size):
    part = flat.reshape(batch_size, seq_len, num_heads, head_size).transpose(0, 2, 1, 3)
    return contiguous(part).reshape(batch_size * num_heads, seq_len, head_size)


def _split_heads(qkv, batch_size, seq_len, num_heads, head_size):
    channels = qkv.shape[1] // 3
    packed = qkv.reshape(batch_size, seq_len, 3 * channels)
    return [_to_heads(contiguous(packed[:, :, b * channels:(b + 1) * channels])
                      .reshape(batch_size * seq_len, channels),
                      batch_size, seq_len, num_heads, head_size) for b in range(3)]


def _pack_heads(parts, batch_size, seq_len, num_heads, head_size):
    channels = num_heads * head_size
    out = xp.zeros((batch_size * seq_len, 3 * channels), dtype=DT)
    for block, part in enumerate(parts):
        flat = part.reshape(batch_size, num_heads, seq_len, head_size).transpose(0, 2, 1, 3)
        out[:, block * channels:(block + 1) * channels] = \
            contiguous(flat).reshape(batch_size * seq_len, channels)
    return out


def attention_backward(dout, qkv, weights, batch_size, seq_len, num_heads):
    channels = qkv.shape[1] // 3
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))
    queries, keys, values = _split_heads(qkv, batch_size, seq_len, num_heads, head_size)
    dout_heads = _to_heads(dout, batch_size, seq_len, num_heads, head_size)

    dweights = dout_heads @ contiguous(values.transpose(0, 2, 1))
    dvalues = contiguous(weights.transpose(0, 2, 1)) @ dout_heads
    row_sum = (weights * dweights).sum(axis=2, keepdims=True)
    dscores = weights * (dweights - row_sum) * scale
    dqueries = dscores @ keys
    dkeys = contiguous(dscores.transpose(0, 2, 1)) @ queries
    return _pack_heads([dqueries, dkeys, dvalues], batch_size, seq_len, num_heads, head_size)


def attention_with_weights(qkv, batch_size, seq_len, num_heads, mask):
    """gpt2.attention_batched, answering the softmax output beside the result."""
    channels = qkv.shape[1] // 3
    head_size = channels // num_heads
    scale = 1.0 / float(np.sqrt(head_size))
    queries, keys, values = _split_heads(qkv, batch_size, seq_len, num_heads, head_size)
    scores = queries @ contiguous(keys.transpose(0, 2, 1)) * scale + mask
    weights = gpt2.softmax_rows(scores)
    out = (weights @ values).reshape(batch_size, num_heads, seq_len, head_size)
    return contiguous(out.transpose(0, 2, 1, 3)).reshape(batch_size * seq_len, channels), weights


class DebugState:
    """llm.c's gpt2_*_debug_state.bin. The gradients are read one tensor at a
    time so that 475 MB never lands in one piece."""

    def __init__(self, path, config):
        self.path = path
        self.config = config
        with open(path, "rb") as f:
            header = struct.unpack("<256i", f.read(HEADER_BYTES))
            if header[0] != DEBUG_MAGIC or header[1] != DEBUG_VERSION:
                raise ValueError(f"{path}: bad header")
            self.batch_size, self.seq_len = header[2], header[3]
            rows = self.batch_size * self.seq_len
            self.inputs = np.frombuffer(f.read(4 * rows), dtype="<i4").reshape(
                self.batch_size, self.seq_len)
            self.targets = np.frombuffer(f.read(4 * rows), dtype="<i4").reshape(
                self.batch_size, self.seq_len)
            f.seek(4 * rows * config.vocab_size, 1)  # expected_logits (V, not the padded width)
            self.loss = struct.unpack("<f", f.read(4))[0]
            self.grads_offset = f.tell()

    def grad(self, name):
        shapes = tensor_shapes(self.config)
        offset = self.grads_offset
        for other, shape in shapes.items():
            if other == name:
                break
            offset += 4 * int(np.prod(shape))
        shape = shapes[name]
        with open(self.path, "rb") as f:
            f.seek(offset)
            raw = f.read(4 * int(np.prod(shape)))
        return xp.asarray(np.frombuffer(raw, dtype="<f4").reshape(shape), dtype=DT)


class Trainer:
    """forward_train / backward / AdamW for gpt2.Model, mirroring
    lib/narray_llm/models/gpt2/training.rb and lib/narray_llm/adam_w.rb.

    The weights are held transposed here as they are on the Ruby side, so the
    gradients come back in llm.c's orientation and four matrices a layer turn
    around on the way into the update.
    """

    DEFAULTS = dict(learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

    def __init__(self, model, **options):
        self.model = model
        self.settings = {**self.DEFAULTS, **options}
        self.moments = {}
        self.steps = 0

    def forward(self, tokens, targets):
        model, config = self.model, self.model.config
        ids = np.asarray(tokens, dtype=np.int32)
        batch_size, seq_len = ids.shape
        acts = {"ids": ids.reshape(-1), "targets": np.asarray(targets).reshape(-1),
                "batch_size": batch_size, "seq_len": seq_len, "layers": []}

        x = model._embed(ids, batch_size, seq_len)
        mask = model._causal_mask(seq_len)
        for weights in model.layers:
            x, saved = self._block(x, weights, batch_size, seq_len, mask)
            acts["layers"].append(saved)

        acts["final"] = x
        acts["lnf"] = gpt2.layernorm(x, model.lnfw, model.lnfb)
        acts["probs"] = gpt2.softmax_rows(acts["lnf"] @ model.wte_t)
        hot = one_hot(acts["targets"], config.vocab_size)
        loss = float(-xp.log((acts["probs"] * hot).sum(axis=1)).mean())
        return loss, acts

    def _block(self, x, w, batch_size, seq_len, mask):
        saved = {"inp": x}
        saved["ln1"] = gpt2.layernorm(x, w["ln1w"], w["ln1b"])
        saved["qkv"] = gpt2.linear(saved["ln1"], w["qkvw_t"], w["qkvb"])
        saved["atty"], saved["att"] = attention_with_weights(
            saved["qkv"], batch_size, seq_len, self.model.config.num_heads, mask)
        saved["res2"] = x + gpt2.linear(saved["atty"], w["attprojw_t"], w["attprojb"])
        saved["ln2"] = gpt2.layernorm(saved["res2"], w["ln2w"], w["ln2b"])
        saved["fch"] = gpt2.linear(saved["ln2"], w["fcw_t"], w["fcb"])
        saved["gelu"] = gpt2.gelu(saved["fch"])
        out = saved["res2"] + gpt2.linear(saved["gelu"], w["fcprojw_t"], w["fcprojb"])
        return out, saved

    def backward(self, acts):
        model, config = self.model, self.model.config
        grads = self._zero_grads()
        dlogits = crossentropy_softmax_backward(acts["probs"], acts["targets"])
        dlnf, dwte_t, _ = matmul_backward(dlogits, acts["lnf"], model.wte_t, bias=False)
        grads["wte"] = contiguous(dwte_t.T)
        dx, grads["lnfw"], grads["lnfb"] = layernorm_backward(dlnf, acts["final"], model.lnfw)

        for layer in range(len(model.layers) - 1, -1, -1):
            dx = self._block_backward(dx, acts["layers"][layer], model.layers[layer], grads,
                                      layer, acts["batch_size"], acts["seq_len"])

        dwte, dwpe = encoder_backward(dx, acts["ids"], config.vocab_size,
                                      acts["batch_size"], acts["seq_len"])
        grads["wte"] = grads["wte"] + dwte
        grads["wpe"][:acts["seq_len"]] = grads["wpe"][:acts["seq_len"]] + dwpe
        return grads

    def _block_backward(self, dout, saved, w, grads, layer, batch_size, seq_len):
        dgelu, dfcprojw_t, dfcprojb = matmul_backward(dout, saved["gelu"], w["fcprojw_t"])
        grads["fcprojw"][layer] += contiguous(dfcprojw_t.T)
        grads["fcprojb"][layer] += dfcprojb

        dfch = gelu_backward(dgelu, saved["fch"])
        dln2, dfcw_t, dfcb = matmul_backward(dfch, saved["ln2"], w["fcw_t"])
        grads["fcw"][layer] += contiguous(dfcw_t.T)
        grads["fcb"][layer] += dfcb

        dres2, dln2w, dln2b = layernorm_backward(dln2, saved["res2"], w["ln2w"])
        grads["ln2w"][layer] += dln2w
        grads["ln2b"][layer] += dln2b
        dres2 = dres2 + dout

        datty, dattprojw_t, dattprojb = matmul_backward(dres2, saved["atty"], w["attprojw_t"])
        grads["attprojw"][layer] += contiguous(dattprojw_t.T)
        grads["attprojb"][layer] += dattprojb

        dqkv = attention_backward(datty, saved["qkv"], saved["att"],
                                  batch_size, seq_len, self.model.config.num_heads)
        dln1, dqkvw_t, dqkvb = matmul_backward(dqkv, saved["ln1"], w["qkvw_t"])
        grads["qkvw"][layer] += contiguous(dqkvw_t.T)
        grads["qkvb"][layer] += dqkvb

        dinp, dln1w, dln1b = layernorm_backward(dln1, saved["inp"], w["ln1w"])
        grads["ln1w"][layer] += dln1w
        grads["ln1b"][layer] += dln1b
        return dres2 + dinp

    def _zero_grads(self):
        config = self.model.config
        grads = {}
        for name, shape in tensor_shapes(config).items():
            if name == "wte":
                shape = (config.vocab_size, shape[1])
            grads[name] = xp.zeros(shape, dtype=DT)
        return grads

    def each_parameter(self, grads):
        model = self.model
        yield ("wte",), grads["wte"], model.wte
        yield ("wpe",), grads["wpe"], model.wpe
        for layer, w in enumerate(model.layers):
            for name in ("ln1w", "ln1b", "qkvb", "attprojb", "ln2w", "ln2b", "fcb", "fcprojb"):
                yield (name, layer), grads[name][layer], w[name]
            for name, stored in (("qkvw", "qkvw_t"), ("attprojw", "attprojw_t"),
                                 ("fcw", "fcw_t"), ("fcprojw", "fcprojw_t")):
                yield (name, layer), contiguous(grads[name][layer].T), w[stored]
        yield ("lnfw",), grads["lnfw"], model.lnfw
        yield ("lnfb",), grads["lnfb"], model.lnfb

    def step(self, grads):
        """gpt2_update in train_gpt2.c. Updates in place: the token table is
        held in both orientations and rebinding would strand one of them."""
        self.steps += 1
        s = self.settings
        bias1 = 1.0 - s["beta1"] ** self.steps
        bias2 = 1.0 - s["beta2"] ** self.steps

        for key, grad, param in self.each_parameter(grads):
            if key not in self.moments:
                self.moments[key] = [xp.zeros(param.shape, dtype=DT),
                                     xp.zeros(param.shape, dtype=DT)]
            first, second = self.moments[key]
            first[...] = s["beta1"] * first + (1.0 - s["beta1"]) * grad
            second[...] = s["beta2"] * second + (1.0 - s["beta2"]) * grad * grad
            corrected = (first / bias1) / (xp.sqrt(second / bias2) + s["eps"])
            param[...] = param - s["learning_rate"] * (corrected + s["weight_decay"] * param)
        self.model.wte_t[...] = self.model.wte.T
        return self.steps

    def bytes(self):
        return sum(a.nbytes for pair in self.moments.values() for a in pair)
