# frozen_string_literal: true

module NArrayLLM
  # The backward pass, one method per *_backward in llm.c's train_gpt2.c.
  #
  # llm.c accumulates into caller-owned buffers because it is C. These answer
  # the gradients instead and let the model add them up, which keeps each one
  # testable on its own against a finite difference.
  #
  # Nothing here names XF. Every array is built from the class of an argument,
  # so the same code runs in fp32 for the model and in fp64 for the tests that
  # check it against a numerical derivative.
  module Backward
    module_function

    # train_gpt2.c residual_backward. Both inputs take the whole gradient.
    def residual(dout)
      [dout, dout]
    end

    # train_gpt2.c gelu_backward. llm.c writes the derivative with 1/cosh^2;
    # 1 - tanh^2 is the same number and cannot overflow, which matters because
    # cosh of a large argument is Inf and Inf's reciprocal is 0 only by luck.
    def gelu(dout, inp)
      cube = 0.044715 * inp * inp * inp
      arg = Ops::GELU_SCALING_FACTOR * (inp + cube)
      tanh_out = XM::NMath.tanh(arg)
      sech2 = 1.0 - (tanh_out * tanh_out)
      local = (0.5 * (1.0 + tanh_out)) +
              (inp * 0.5 * sech2 * Ops::GELU_SCALING_FACTOR * (1.0 + (3.0 * 0.044715 * inp * inp)))
      local * dout
    end

    # train_gpt2.c layernorm_backward. inp and dout are [N, C], weight is [C].
    # Answers [dinp, dweight, dbias].
    #
    # llm.c caches mean and rstd from the forward pass; they are recomputed here
    # instead, which costs two reductions and keeps the forward unchanged.
    def layernorm(dout, inp, weight, eps: Ops::LAYERNORM_EPS)
      mean = inp.mean(axis: 1, keepdims: true)
      centered = inp - mean
      variance = (centered * centered).mean(axis: 1, keepdims: true)
      rstd = 1.0 / XM::NMath.sqrt(variance + eps)
      norm = centered * rstd

      dnorm = weight * dout
      dnorm_mean = dnorm.mean(axis: 1, keepdims: true)
      dnorm_norm_mean = (dnorm * norm).mean(axis: 1, keepdims: true)

      dinp = (dnorm - dnorm_mean - (norm * dnorm_norm_mean)) * rstd
      [dinp, (norm * dout).sum(axis: 0), dout.sum(axis: 0)]
    end

    # train_gpt2.c matmul_backward, in this repository's orientation: weight_t
    # is [C, OC] where llm.c's weight is [OC, C]. inp is [N, C], dout is [N, OC].
    # Answers [dinp, dweight_t, dbias].
    def matmul(dout, inp, weight_t, bias: true)
      dinp = dout.dot(Ops.contiguous(weight_t.transpose, weight_t.class))
      dweight_t = Ops.contiguous(inp.transpose, inp.class).dot(dout)
      [dinp, dweight_t, bias ? dout.sum(axis: 0) : nil]
    end

    # train_gpt2.c crossentropy_softmax_backward. probs is [N, V] and targets is
    # a Ruby Array of N ids. dloss is 1/N for llm.c's mean loss.
    #
    # The one-hot is built arithmetically so no index array reaches the device
    # (AGENTS.md).
    def crossentropy_softmax(probs, targets, dloss: nil)
      rows = probs.shape[0]
      (probs - one_hot(probs.class, targets, probs.shape[1])) * (dloss || (1.0 / rows))
    end

    # train_gpt2.c encoder_backward. dout is [B*T, C] and ids is a Ruby Array of
    # B*T token ids. Answers [dwte, dwpe], shaped [vocab_size, C] and [T, C].
    #
    # llm.c scatters into dwte by token id. A scatter needs an index array, so
    # the same sum is taken as one matmul against the one-hot instead.
    def encoder(dout, ids, vocab_size:, batch_size:, seq_len:)
      channels = dout.shape[1]
      hot = one_hot(dout.class, ids, vocab_size)
      dwte = Ops.contiguous(hot.transpose, hot.class).dot(dout)
      dwpe = dout.reshape(batch_size, seq_len, channels).sum(axis: 0)
      [dwte, dwpe]
    end

    # train_gpt2.c attention_backward. qkv is [B*T, 3C] and weights is the saved
    # softmax output, [B*NH, T, T]. Answers dqkv, [B*T, 3C].
    #
    # weights is already zero above the diagonal, so the causal mask needs no
    # separate handling: every product below carries that zero through.
    def attention(dout, qkv, weights, batch_size:, seq_len:, num_heads:)
      channels = qkv.shape[1] / 3
      head_size = channels / num_heads
      scale = 1.0 / Math.sqrt(head_size)
      queries, keys, values = split_heads(qkv, batch_size, seq_len, num_heads, head_size)
      dout_heads = to_heads(dout, batch_size, seq_len, num_heads, head_size)

      dweights = Ops.batched_dot(dout_heads, Ops.contiguous(values.transpose(0, 2, 1), values.class))
      dvalues = Ops.batched_dot(Ops.contiguous(weights.transpose(0, 2, 1), weights.class), dout_heads)

      row_sum = (weights * dweights).sum(axis: 2, keepdims: true)
      dscores = weights * (dweights - row_sum) * scale

      dqueries = Ops.batched_dot(dscores, keys)
      dkeys = Ops.batched_dot(Ops.contiguous(dscores.transpose(0, 2, 1), dscores.class), queries)

      pack_heads([dqueries, dkeys, dvalues], qkv.class,
                 batch_size, seq_len, num_heads, head_size)
    end

    # 1.0 where the id matches the column, built with arithmetic rather than a
    # comparison so no Bit array is formed (AGENTS.md).
    def one_hot(klass, ids, num_classes)
      rows = Array(ids).flatten
      column = klass.new(1, num_classes).seq
      identifiers = klass.cast(rows).reshape(rows.size, 1)
      1.0 - (identifiers - column).abs.clip(0.0, 1.0).ceil
    end

    # [B*T, 3C] -> three [B*NH, T, hs], matching Ops.attention's split.
    def split_heads(qkv, batch_size, seq_len, num_heads, head_size)
      channels = qkv.shape[1] / 3
      packed = qkv.reshape(batch_size, seq_len, 3 * channels)
      (0..2).map do |block|
        slice = packed[true, true, (block * channels)...((block + 1) * channels)]
        to_heads(Ops.contiguous(slice, qkv.class).reshape!(batch_size * seq_len, channels),
                 batch_size, seq_len, num_heads, head_size)
      end
    end

    # [B*T, C] -> [B*NH, T, hs]
    def to_heads(flat, batch_size, seq_len, num_heads, head_size)
      Ops.contiguous(flat.reshape(batch_size, seq_len, num_heads, head_size)
                         .transpose(0, 2, 1, 3), flat.class)
         .reshape!(batch_size * num_heads, seq_len, head_size)
    end

    # three [B*NH, T, hs] -> [B*T, 3C], laid out Q then K then V.
    def pack_heads(parts, klass, batch_size, seq_len, num_heads, head_size)
      channels = num_heads * head_size
      out = klass.zeros(batch_size * seq_len, 3 * channels)
      parts.each_with_index do |part, block|
        flat = Ops.contiguous(part.reshape(batch_size, num_heads, seq_len, head_size)
                                  .transpose(0, 2, 1, 3), part.class)
                  .reshape!(batch_size * seq_len, channels)
        out[true, (block * channels)...((block + 1) * channels)] = flat
      end
      out
    end
  end
end
