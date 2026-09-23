# frozen_string_literal: true

module NArrayLLM
  # The individual layers of the forward pass, each matching one function in
  # llm.c's train_gpt2.c. No Bit arrays, no fancy indexing, no mask assignment.
  module Ops
    # train_gpt2.c:86 layernorm_forward
    LAYERNORM_EPS = 1e-5

    # train_gpt2.c:407. The tanh approximation, which is what GPT-2 uses.
    # Not the erf formulation.
    GELU_SCALING_FACTOR = Math.sqrt(2.0 / Math::PI)

    # sqrt(2/pi) * (x + 0.044715 x^3) factored as (C1 x^2 + C0) x: same value,
    # one kernel fewer.
    GELU_CUBIC_FACTOR = GELU_SCALING_FACTOR * 0.044715

    # Large enough that exp() underflows to exactly 0, small enough that adding
    # it to a logit cannot produce Inf. fp16 needs its own: -1.0e9 is already
    # -Infinity there, and Inf - Inf in softmax is NaN.
    MASK_VALUE = NArrayLLM.fp16? ? -1.0e4 : -1.0e9

    # run.c:182. Llama uses the same 1e-5 as GPT-2's layernorm, but the two are
    # separate constants because nothing ties them together.
    RMSNORM_EPS = 1e-5

    # run.c:268. The RoPE base.
    ROPE_THETA = 10_000.0

    # The last argument whose exp() is finite: 88.7 in fp32, 11.09 in fp16
    # (exp(11.0) is 59872 and exp(11.2) is already Inf). Clipping here keeps Inf
    # out of silu and softmax (AGENTS.md).
    EXP_LIMIT = NArrayLLM.fp16? ? 11.0 : 88.0

    # Numo's dot handles a stack of matrices, but Numo::Linalg.dot -- which
    # takes over once numo-linalg is loaded -- only accepts 2-D. Decided once
    # here rather than rescued per call.
    BATCHED_DOT_NATIVE = begin
      XF.zeros(2, 2, 2).dot(XF.zeros(2, 2, 2))
      true
    rescue StandardError
      false
    end

    # Cumo ships these fused; Numo has none of them yet, and a cumo older than
    # the one that added a given kernel has only some. Each is asked for on its
    # own so the ones that are there still get used. The math pair is asked of
    # SFloat::Math, not NMath: NMath answers respond_to? only since cumo #463,
    # and before that it says false and silently turns the fused path off.
    FUSED_LAYERNORM = XF.method_defined?(:layer_norm)
    FUSED_RMSNORM = XF.method_defined?(:rms_norm)
    FUSED_SOFTMAX = XF.method_defined?(:softmax)
    FUSED_GELU = XF::Math.respond_to?(:gelu_tanh)
    FUSED_GELU_ERF = XF::Math.respond_to?(:gelu)
    FUSED_SILU = XF::Math.respond_to?(:silu)
    FUSED_QUANTIZE = XF.method_defined?(:quantize_symmetric)
    GEMM_WITH_C = XF.method_defined?(:gemm)
    # Cumo lets a one-row column slice through reshape!; Numo calls the same
    # slice non-contiguous and refuses. Probed rather than assumed, so the
    # caller can take a view where that is free and a copy where it is not.
    RESHAPABLE_ROW_SLICE = begin
      XF.zeros(1, 4)[true, 0...2].reshape!(1, 2, 1)
      true
    rescue StandardError
      false
    end

    # DTYPE=fp16 has no safe fallback for the two norms: both square the row, and
    # a GPT-2 outlier squared reaches 142x the fp16 ceiling.
    FP16_NEEDS_FUSED = "DTYPE=fp16 needs a backend whose %<m>s folds in fp32; this " \
                       "one has no fused %<m>s and the fp16 square overflows"

    module_function

    # [n, i, j] x [n, j, k] -> [n, i, k]
    def batched_dot(a, b)
      return a.dot(b) if BATCHED_DOT_NATIVE

      out = a.class.zeros(a.shape[0], a.shape[1], b.shape[2])
      a.shape[0].times do |i|
        out[i, true, true] = contiguous(a[i, true, true], a.class)
                             .dot(contiguous(b[i, true, true], b.class))
      end
      out
    end

    # new rather than zeros: store overwrites every element, so the fill kernel
    # zeros costs is wasted work.
    def contiguous(a, into = XF)
      out = into.new(*a.shape)
      out.store(a)
      out
    end

    # out = x . weight_t + bias, where weight_t is the [C, OC] transpose of
    # llm.c's [OC, C] weight (train_gpt2.c:184 matmul_forward).
    def linear(x, weight_t, bias = nil)
      y = x.dot(weight_t)
      bias.nil? ? y : y + bias
    end

    # One column range of an array, in whatever form the backend can reshape in
    # place afterwards. Past one row the slice is strided whatever the backend,
    # and reshape! refuses that, so it is copied.
    def row_slice(array, span)
      view = array[true, span]
      RESHAPABLE_ROW_SLICE && array.shape[0] == 1 ? view : contiguous(view)
    end

    M_SQRT1_2 = 1.0 / Math.sqrt(2.0)

    Q_MAX = 127.0

    # runq.c:145 quantize. Splits the row into groups of group_size, scales each
    # by its own maximum absolute value, and rounds to the int8 range. The
    # values are kept in XF rather than Int8: they are integers either way, and
    # every use of them is a product with a float.
    def quantize_groups(x, group_size)
      rows = x.reshape(x.size / group_size, group_size)
      # The fused kernel answers in Int8, and Int8#mulsum(Int8) accumulates in
      # Int8, so the cast back is what keeps the group sums from wrapping.
      if FUSED_QUANTIZE
        q, scale = rows.quantize_symmetric
        return [XF.cast(q), scale]
      end

      scale = rows.abs.max(axis: 1, keepdims: true) / Q_MAX
      # A group of exact zeros would divide by zero. Dividing it by one instead
      # gives the zeros it should quantize to.
      positive = scale.clip(0.0, 1.0).ceil
      [(rows / (scale + (1.0 - positive))).round, scale.flatten]
    end

    # runq.c:317 matmul, with both sides quantized. weight.q is
    # [out, groups, group_size] int8 carrying one scale per group.
    #
    # runq.c accumulates each group in int32 before scaling it. Folding the
    # products in fp32 gives the same integer: group_size * 127 * 127 is at most
    # 1,032,256, well inside the 2 ** 24 that fp32 represents exactly, so no
    # partial sum is ever rounded and the order of the additions cannot matter.
    def qmatmul(weight, xq, xs)
      ival = weight.q.mulsum(xq, axis: 2)
      (ival * weight.scales * xs).sum(axis: 1).reshape!(1, weight.shape[0])
    end

    # Same product, with the bias handed to the GEMM as C so that beta folds it
    # in instead of a second kernel. cuBLAS reads C and writes the result over
    # it, so bias_c has to be a buffer the caller is willing to lose.
    def linear_into(x, weight_t, bias_c)
      x.gemm(weight_t, bias_c.inplace, beta: 1)
    end

    # x . weight_t + residual. Where the backend takes a C, beta folds the
    # residual in and the addition costs no kernel of its own. The residual is
    # written over, so the caller has to be done with it.
    def linear_add(x, weight_t, residual)
      return linear_into(x, weight_t, residual) if GEMM_WITH_C

      x.dot(weight_t) + residual
    end

    # x: [N, C]. Normalizes each row, then scales and shifts by weight/bias [C].
    def layernorm(x, weight, bias, eps: LAYERNORM_EPS)
      return x.layer_norm(weight, bias, eps: eps) if FUSED_LAYERNORM
      raise NArrayLLM::Error, format(FP16_NEEDS_FUSED, m: "layer_norm") if NArrayLLM.fp16?

      mean = x.mean(axis: 1, keepdims: true)
      centered = x - mean
      variance = (centered * centered).mean(axis: 1, keepdims: true)
      # Dividing by the deviation costs one kernel fewer than forming 1/x first.
      centered / XM::NMath.sqrt(variance + eps) * weight + bias
    end

    # gelu_tanh, not gelu: the latter is the erf formulation, which is a
    # different function from the one GPT-2 was trained with.
    def gelu(x)
      return XM::NMath.gelu_tanh(x) if FUSED_GELU

      inner = (x * x * GELU_CUBIC_FACTOR + GELU_SCALING_FACTOR) * x
      0.5 * x * (1.0 + XM::NMath.tanh(inner))
    end

    # The erf formulation, which is what torch's nn.functional.gelu computes
    # and what Whisper was trained with. Not the same function as gelu above:
    # the two differ by about 1e-3 at their widest.
    #
    # XF::Math rather than XM::NMath, which dispatches through method_missing.
    def gelu_erf(x)
      return XF::Math.gelu(x) if FUSED_GELU_ERF

      0.5 * x * (1.0 + XF::Math.erf(x * M_SQRT1_2))
    end

    # run.c:178 rmsnorm. Unlike layernorm this does not centre the row, so a row
    # with a large mean keeps it; the scale is the root mean square, not the
    # standard deviation.
    def rmsnorm(x, weight, eps: RMSNORM_EPS)
      return x.rms_norm(weight, eps: eps) if FUSED_RMSNORM
      raise NArrayLLM::Error, format(FP16_NEEDS_FUSED, m: "rms_norm") if NArrayLLM.fp16?

      # The reciprocal first, then two multiplications, because that is what
      # llama2.c does (run.c:282). Dividing each element instead rounds once
      # more and the result is a bit away from the reference.
      ms = (x * x).mean(axis: -1, keepdims: true)
      rstd = 1.0 / XM::NMath.sqrt(ms + eps)
      weight * (rstd * x)
    end

    # x * sigmoid(x) (run.c:337). exp overflows to Inf in fp32 past ~88, and
    # the result there is 0 either way, so the argument is clipped first.
    def silu(x)
      return XM::NMath.silu(x) if FUSED_SILU

      # Reciprocal then multiply, as run.c:457 does. Dividing instead is a bit
      # away from the reference.
      x * (1.0 / (1.0 + XM::NMath.exp((-x).clip(-EXP_LIMIT, EXP_LIMIT))))
    end

    # cos / sin for every position and rotation pair, both [max_pos, head_size/2].
    # run.c:267-269 uses freq = 1 / 10000^(head_dim / head_size) with head_dim
    # stepping by 2, so pair j has exponent -2j / head_size.
    # Laid out per element rather than per pair, so that rope can rotate a whole
    # head in one expression: cos is repeated across each pair and sin carries
    # the sign the even half needs.
    def rope_tables(max_pos, head_size)
      half = head_size / 2
      exponent = XF.new(1, half).seq * (-2.0 * Math.log(ROPE_THETA) / head_size)
      angle = XF.new(max_pos, 1).seq * XM::NMath.exp(exponent)
      cos = XM::NMath.cos(angle)
      sin = XM::NMath.sin(angle)

      cos_t = XF.new(max_pos, half, 2).allocate
      cos_t[true, true, 0] = cos
      cos_t[true, true, 1] = cos
      sin_t = XF.new(max_pos, half, 2).allocate
      sin_t[true, true, 0] = -sin
      sin_t[true, true, 1] = sin
      [cos_t.reshape!(max_pos, head_size), sin_t.reshape!(max_pos, head_size)]
    end

    # Widens [t, num_kv_heads * head_size] to [t, num_kv_heads * kv_mul * head_size]
    # by repeating each key/value head kv_mul times in a row. run.c:295 reads head
    # h / kv_mul for query head h, which is exactly this order.
    #
    # Broadcasting into a zeroed array rather than an index array: an NArray index
    # blocks on a device-to-host copy in Cumo (AGENTS.md).
    def repeat_kv_heads(x, num_kv_heads:, kv_mul:)
      return x if kv_mul == 1

      t = x.shape[0]
      head_size = x.shape[1] / num_kv_heads
      x_shape = x.shape
      begin
        (XF.zeros(t, num_kv_heads, kv_mul, head_size) +
          x.reshape!(t, num_kv_heads, 1, head_size))
          .reshape!(t, num_kv_heads * kv_mul * head_size)
      ensure
        x.reshape!(*x_shape)
      end
    end

    # Rotates adjacent pairs within each head. x is [t, num_heads * head_size];
    # cos_t / sin_t are the rows of rope_tables for those t positions.
    #
    # The pairs are taken as a [.., half, 2] view and put back with concatenate
    # rather than by assigning into a strided slice (AGENTS.md).
    #
    # Every shape change is reshape!, not reshape: reshape copies the whole
    # array even when it is contiguous, which cost this one call six copies out
    # of the fifteen kernels it launched (docs/results/llama2-110m.md). The
    # three arguments are put back the way they came, and nothing here writes
    # to them, so the caller cannot tell.
    # x * cos + swap(x) * sin, where swap exchanges the two elements of every
    # pair. Splitting the halves and concatenating them back costs nine kernels
    # a call against five, and the [1, 0] subscript is a Ruby Array, so it does
    # not synchronize the way an NArray index would (AGENTS.md).
    def rope(x, cos_t, sin_t, num_heads:)
      t = x.shape[0]
      head_size = x.shape[1] / num_heads
      half = head_size / 2
      x_shape = x.shape
      table_shape = cos_t.shape

      begin
        x.reshape!(t, num_heads, half, 2)
        cos_t.reshape!(t, 1, half, 2)
        sin_t.reshape!(t, 1, half, 2)
        (x * cos_t + x[true, true, true, [1, 0]] * sin_t).reshape!(t, num_heads * head_size)
      ensure
        x.reshape!(*x_shape)
        cos_t.reshape!(*table_shape)
        sin_t.reshape!(*table_shape)
      end
    end

    # Softmax along the last axis, with the max subtracted first so exp() never
    # overflows (AGENTS.md: fp32 exp of a large value is Inf, and Inf - Inf is NaN).
    def softmax_rows(x)
      return x.softmax if FUSED_SOFTMAX

      shifted = x - x.max(axis: -1, keepdims: true)
      e = XM::NMath.exp(shifted)
      e / e.sum(axis: -1, keepdims: true)
    end

    # Everything below builds its 0/1 flags arithmetically. A comparison would
    # answer a Bit array, and the Bit path is where Cumo incompatibilities and
    # slowdowns live (AGENTS.md). 1 where x >= threshold, 0 below it.
    def at_least(x, threshold)
      1.0 - (threshold - x).clip(0.0, 1.0).ceil
    end

    # 1 for the k largest of values, 0 for the rest. values is [V].
    #
    # The threshold is a value, not a position, so nothing has to carry the
    # permutation back: no scatter, no fancy index. The cost is that ties at the
    # boundary all survive, so k is a floor and not a ceiling.
    #
    # sorted: lets a caller that already holds values ascending hand it over.
    # One sort is 8 launches here, so sharing it is worth the argument.
    def top_k_keep(values, k, sorted: nil)
      size = values.shape[0]
      return XF.ones(size) if k >= size
      raise Error, "top_k must be positive, got #{k}" unless k.positive?

      # An Integer subscript reads nothing back to the host (AGENTS.md).
      at_least(values, (sorted || values.sort)[size - k])
    end

    # 1 for the smallest set of probabilities whose total reaches p, 0 for the
    # rest. probs is [V] and already sums to 1.
    #
    # The running total is taken over the descending order, and a token is kept
    # when the total *before* it is still under p, which is what keeps the token
    # that crosses p. As in top_k_keep the answer comes back as a value.
    def top_p_keep(probs, p, sorted: nil)
      return XF.ones(probs.shape[0]) if p >= 1.0
      raise Error, "top_p must be positive, got #{p}" unless p.positive?

      # reverse answers a view and launches nothing, and both cumsum and the
      # element-wise work below take that view as it is.
      descending = (sorted || probs.sort).reverse
      before = descending.cumsum - descending
      kept = 1.0 - at_least(before, p)
      # 2.0 is past any probability, so the minimum runs over the kept ones.
      smallest = ((descending * kept) + ((1.0 - kept) * 2.0)).min
      at_least(probs, smallest)
    end

    # The first position whose running total passes u, as an Integer. probs is
    # [V] and sums to 1, and u is a Float in [0, 1).
    #
    # Counting the positions that fall short is the same answer as searching for
    # the first one that does not, and it is one reduction instead of a scan.
    # That reduction is the single readback this costs.
    def sample_index(probs, u)
      size = probs.shape[0]
      below = (u - probs.cumsum).clip(0.0, 1.0).ceil.sum
      index = NArrayLLM.scalar(below).to_i
      # Rounding can leave the last total a hair under u.
      index >= size ? size - 1 : index
    end

    # [t, t], 0 on and below the diagonal and MASK_VALUE strictly above it.
    # Built with clip+ceil instead of a Bit comparison.
    def causal_mask(t)
      row = XF.new(t, 1).seq
      col = XF.new(1, t).seq
      MASK_VALUE * ((XF.zeros(t, t) + col) - row).clip(0.0, 1.0).ceil
    end

    # Rows of table picked by token id. The subscript is a Ruby Array, which
    # has no device data to read back and so stays asynchronous; an NArray
    # subscript is what blocks on a device-to-host copy (AGENTS.md). The result
    # is an index-backed view, so it is materialized before anything reshapes
    # or multiplies it.
    def gather_rows(table, token_ids)
      contiguous(table[Array(token_ids).flatten, true])
    end

    # One-hot rows for the token ids, built arithmetically so no index array
    # ever reaches the device (AGENTS.md: NArray gather synchronizes in Cumo).
    # Max pooling over [N, H, W, C]. One strided view per position in the
    # window, folded together with an elementwise maximum, so nothing is
    # gathered and no index array reaches the device.
    #
    # The padding is filled with MASK_VALUE rather than zero: a window that
    # fell entirely outside would otherwise answer zero, and the value has to
    # sit below anything the activation can produce.
    def max_pool2d(x, kernel:, stride:, padding: 0, layout: :nhwc)
      height, width = spatial(x, layout)
      out_h = pool_size(height, kernel, stride, padding)
      out_w = pool_size(width, kernel, stride, padding)
      source = padding.zero? ? x : pad_with(x, padding, MASK_VALUE, layout)

      best = nil
      kernel.times do |i|
        kernel.times do |j|
          rows = (i..(i + ((out_h - 1) * stride))).step(stride)
          cols = (j..(j + ((out_w - 1) * stride))).step(stride)
          view = window2d(source, rows, cols, layout)
          best = best.nil? ? view : XF.maximum(best, view)
        end
      end
      best.is_a?(XF) && best.contiguous? ? best : contiguous(best)
    end

    # [N, H, W, C] or [N, C, H, W] -> [N, C], the average over every position.
    def global_average_pool(x, layout: :nhwc)
      x.mean(axis: layout == :nhwc ? [1, 2] : [2, 3])
    end

    def spatial(x, layout)
      layout == :nhwc ? x.shape[1..2] : x.shape[2..3]
    end

    def window2d(source, rows, cols, layout)
      layout == :nhwc ? source[true, rows, cols, true] : source[true, true, rows, cols]
    end

    def pool_size(size, kernel, stride, padding)
      ((size + (2 * padding) - kernel) / stride) + 1
    end

    def pad_with(x, padding, value, layout = :nhwc)
      shape = x.shape.dup
      axes = layout == :nhwc ? [1, 2] : [2, 3]
      axes.each { |axis| shape[axis] += 2 * padding }
      padded = XF.new(*shape)
      padded.allocate
      first, second = axes
      inside = ->(axis) { padding...(padding + x.shape[axis]) }
      span = ->(ranges) { shape.each_index.map { |axis| ranges.fetch(axis, true) } }
      [0...padding, inside.(first).end...shape[first]].each do |edge|
        padded[*span.({ first => edge })].fill(value)
      end
      [0...padding, inside.(second).end...shape[second]].each do |edge|
        padded[*span.({ first => inside.(first), second => edge })].fill(value)
      end
      padded[*span.({ first => inside.(first), second => inside.(second) })] = x
      padded
    end

    def one_hot(token_ids, num_classes)
      # Built in fp32 whatever the compute dtype is: a token id past 2048 does not
      # survive fp16, so the comparison has to happen before the cast.
      ids = XM::SFloat.cast(Array(token_ids).flatten)
      n = ids.size
      col = XM::SFloat.new(1, num_classes).seq
      # |col - id| is 0 at the token and >= 1 everywhere else.
      hot = 1.0 - (col - ids.reshape!(n, 1)).abs.clip(0.0, 1.0).ceil
      NArrayLLM.reduced_precision? ? XF.cast(hot) : hot
    end

    # qkv: [B*T, 3C] laid out per position as [Q(C) | K(C) | V(C)]
    # (train_gpt2.c:271 attention_forward). Returns [B*T, C].
    #
    # Every head is one batch entry of a single 3-D dot rather than its own
    # [T, hs] x [hs, T] call, so the launch count stops scaling with B * NH.
    def attention(qkv, batch_size:, seq_len:, num_heads:, mask:, prof: Profiler::NULL)
      attention_with_weights(qkv, batch_size: batch_size, seq_len: seq_len,
                             num_heads: num_heads, mask: mask, prof: prof).first
    end

    # The same forward, answering the softmax output beside the result. Training
    # needs it: Backward.attention takes it as the one intermediate it cannot
    # recompute cheaply.
    def attention_with_weights(qkv, batch_size:, seq_len:, num_heads:, mask:, prof: Profiler::NULL)
      channels = qkv.shape[1] / 3
      head_size = channels / num_heads
      scale = 1.0 / Math.sqrt(head_size)
      qkv_shape = qkv.shape

      begin
        # reshape! wherever the receiver is this expression's own temporary,
        # because reshape copies even a contiguous array
        # (docs/results/llama2-110m.md). qkv belongs to the caller, so its shape
        # is put back below; nothing here writes to it.
        packed = qkv.reshape!(batch_size, seq_len, 3 * channels)

        queries, keys, values = prof.section(:attention) do
          (0..2).map do |block|
            # [B, T, C] -> [B, T, NH, hs] -> [B, NH, T, hs] -> [B*NH, T, hs].
            # The slice is non-contiguous, so it is made contiguous first and
            # then reshaped in place, rather than reshaped (a copy) and copied
            # again by the transpose below.
            contiguous(contiguous(packed[true, true,
                                         (block * channels)...((block + 1) * channels)], qkv.class)
                         .reshape!(batch_size, seq_len, num_heads, head_size)
                         .transpose(0, 2, 1, 3), qkv.class)
              .reshape!(batch_size * num_heads, seq_len, head_size)
          end
        end

        scores = prof.section(:attention) do
          batched_dot(queries, contiguous(keys.transpose(0, 2, 1), keys.class)) * scale + mask
        end
        weights = prof.section(:softmax) { softmax_rows(scores) }

        out = prof.section(:attention) do
          contiguous(batched_dot(weights, values)
                            .reshape!(batch_size, num_heads, seq_len, head_size)
                            .transpose(0, 2, 1, 3), qkv.class)
            .reshape!(batch_size * seq_len, channels)
        end
        [out, weights]
      ensure
        qkv.reshape!(*qkv_shape)
      end
    end

    # Attention for one new query against the cached keys and values.
    # q: [1, C]; keys, values: [t, C]. Returns [1, C].
    #
    # No causal mask here: everything already in the cache is at or before the
    # current position, so there is nothing to hide and no [t, t] matrix to build.
    #
    # All heads are done at once. Per head the score is a [1, hs] x [hs, t]
    # product, which is far too small to keep a GPU busy, so a loop over heads
    # costs one kernel launch per head per step and nothing else. Broadcasting q
    # over the cached rows and summing inside each head's channel block is the
    # same arithmetic in two kernels for all heads.
    def decode_attention(q, keys, values, num_heads:, num_kv_heads: num_heads, prof: Profiler::NULL)
      if keys.ndim == 3
        if num_kv_heads != num_heads
          raise Error, 'batched decode attention does not do grouped query attention yet'
        end

        return decode_attention_batched(q, keys, values, num_heads: num_heads, prof: prof)
      end

      if num_kv_heads != num_heads
        return decode_attention_grouped(q, keys, values, num_heads: num_heads,
                                        num_kv_heads: num_kv_heads, prof: prof)
      end

      channels = q.shape[1]
      head_size = channels / num_heads
      length = keys.shape[0]
      scale = 1.0 / Math.sqrt(head_size)

      # reshape! rather than reshape wherever the receiver is this expression's
      # own temporary or a throwaway cache view: reshape copies the whole array
      # even when it is contiguous (docs/results/llama2-110m.md). values is put
      # back so that a caller holding the view still sees the shape it asked for.
      # weights.transpose is not contiguous, and reshape! refuses that outright,
      # so that one stays a copy.
      values_shape = values.shape
      keys_shape = keys.shape
      q_shape = q.shape
      scores = prof.section(:attention) do
        keys.reshape!(length, num_heads, head_size)
            .mulsum(q.reshape!(1, num_heads, head_size), axis: 2).transpose * scale
      ensure
        keys.reshape!(*keys_shape)
        q.reshape!(*q_shape)
      end
      weights = prof.section(:softmax) { softmax_rows(scores) }
      prof.section(:attention) do
        (values.reshape!(length, num_heads, head_size) *
          weights.transpose.reshape(length, num_heads, 1)).sum(axis: 0).reshape!(1, channels)
      ensure
        values.reshape!(*values_shape)
      end
    end

    # The same step for a batch. q is [B, C] and the cache views are [t, B, C],
    # so the reduction axis stays 0 and what changes is that the head split now
    # sits behind a batch axis. softmax still needs t last, hence the transpose.
    def decode_attention_batched(q, keys, values, num_heads:, prof: Profiler::NULL)
      length, batch, channels = keys.shape
      head_size = channels / num_heads
      scale = 1.0 / Math.sqrt(head_size)

      q_shape = q.shape
      keys_shape = keys.shape
      values_shape = values.shape
      scores = prof.section(:attention) do
        keys.reshape!(length, batch, num_heads, head_size)
            .mulsum(q.reshape!(1, batch, num_heads, head_size), axis: 3)
            .transpose(1, 2, 0) * scale
      ensure
        keys.reshape!(*keys_shape)
        q.reshape!(*q_shape)
      end
      weights = prof.section(:softmax) { softmax_rows(scores) }
      prof.section(:attention) do
        (values.reshape!(length, batch, num_heads, head_size) *
          weights.transpose(2, 0, 1).reshape(length, batch, num_heads, 1))
          .sum(axis: 0).reshape!(batch, channels)
      ensure
        values.reshape!(*values_shape)
      end
    end

    # The same step when kv_mul query heads share one key/value head. q is
    # [1, num_heads * head_size] while keys and values are [t, num_kv_heads *
    # head_size], so the shared axis is broadcast rather than the keys being
    # widened into a [t, num_heads * head_size] copy per layer per step.
    #
    # Splitting the query as [kv, kv_mul, hs] puts head kv * kv_mul + mul where
    # run.c:295 expects it, which is the same grouping repeat_kv_heads makes.
    def decode_attention_grouped(q, keys, values, num_heads:, num_kv_heads:, prof: Profiler::NULL)
      channels = q.shape[1]
      head_size = channels / num_heads
      kv_mul = num_heads / num_kv_heads
      length = keys.shape[0]
      scale = 1.0 / Math.sqrt(head_size)

      # Same reasoning as above. q belongs to the caller and keys / values are
      # cache views, so all three get their shapes put back.
      q_shape = q.shape
      keys_shape = keys.shape
      values_shape = values.shape

      scores = prof.section(:attention) do
        (keys.reshape!(length, num_kv_heads, 1, head_size) *
          q.reshape!(1, num_kv_heads, kv_mul, head_size))
          .sum(axis: 3).reshape!(length, num_heads).transpose * scale
      ensure
        keys.reshape!(*keys_shape)
        q.reshape!(*q_shape)
      end
      weights = prof.section(:softmax) { softmax_rows(scores) }
      prof.section(:attention) do
        (values.reshape!(length, num_kv_heads, 1, head_size) *
          weights.transpose.reshape(length, num_kv_heads, kv_mul, 1))
          .sum(axis: 0).reshape!(1, channels)
      ensure
        values.reshape!(*values_shape)
      end
    end

    # One pre-LayerNorm transformer block (train_gpt2.c:820-843 in gpt2_forward).
    # x: [B*T, C] in and out. Pass a Hash as trace: to capture the intermediates.
    def transformer_block(x, w, batch_size:, seq_len:, num_heads:, mask:,
                          trace: nil, prefix: '', prof: Profiler::NULL, kv_sink: nil)
      ln1 = prof.section(:layernorm) { layernorm(x, w[:ln1w], w[:ln1b]) }
      qkv = prof.section(:gemm) { linear(ln1, w[:qkvw_t], w[:qkvb]) }
      if kv_sink
        channels = qkv.shape[1] / 3
        prof.section(:cache) do
          kv_sink.call(qkv[true, channels...(2 * channels)],
                       qkv[true, (2 * channels)...(3 * channels)])
        end
      end
      attn = attention(qkv, batch_size: batch_size, seq_len: seq_len,
                       num_heads: num_heads, mask: mask, prof: prof)
      attproj = prof.section(:gemm) { linear(attn, w[:attprojw_t], w[:attprojb]) }
      residual2 = prof.section(:residual) { x + attproj }

      ln2 = prof.section(:layernorm) { layernorm(residual2, w[:ln2w], w[:ln2b]) }
      fch = prof.section(:gemm) { linear(ln2, w[:fcw_t], w[:fcb]) }
      fch_gelu = prof.section(:gelu) { gelu(fch) }
      fcproj = prof.section(:gemm) { linear(fch_gelu, w[:fcprojw_t], w[:fcprojb]) }
      residual3 = prof.section(:residual) { residual2 + fcproj }

      if trace
        trace["#{prefix}ln1"] = ln1
        trace["#{prefix}qkv"] = qkv
        trace["#{prefix}atty"] = attn
        trace["#{prefix}attproj"] = attproj
        trace["#{prefix}residual2"] = residual2
        trace["#{prefix}ln2"] = ln2
        trace["#{prefix}fch"] = fch
        trace["#{prefix}fch_gelu"] = fch_gelu
        trace["#{prefix}fcproj"] = fcproj
        trace["#{prefix}residual3"] = residual3
      end

      residual3
    end
  end
end
