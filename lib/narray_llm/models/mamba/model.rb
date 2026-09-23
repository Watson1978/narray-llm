# frozen_string_literal: true

module NArrayLLM
  module Mamba
    # Mamba forward pass, following forward() and forward_layer() in
    # kroggen/mamba.c (learning branch).
    #
    # There is no attention and no KV cache. Each layer carries two
    # recurrences instead, and both are rewritten on every token, so a prompt
    # is run one token at a time exactly as generation is.
    class Model
      attr_reader :config

      def self.load(path)
        new(Checkpoint.load(path))
      end

      # A reaches -2.836e+08 in mamba-130m, so 1535 of its 589824 entries turn
      # into -Inf in fp16, and 0 * -Inf is the NaN that makes every token come
      # out 0. softplus overflows too: its argument reaches 22.26 where fp16's
      # exp is already Inf at 11.09. bf16 keeps fp32's exponent and runs.
      NEEDS_EXPONENT_RANGE = "Mamba needs fp32 or bf16, not %<dtype>s: A runs " \
                             "to -2.8e+08 and softplus takes arguments past 22."

      def initialize(checkpoint)
        @config = checkpoint.config
        raise Error, format(NEEDS_EXPONENT_RANGE, dtype: NArrayLLM.dtype_name) if NArrayLLM.fp16?

        prepare(checkpoint)
      end

      def new_state
        State.new(num_layers: @config.num_layers, d_inner: @config.d_inner,
                  d_conv: @config.d_conv, d_state: @config.d_state)
      end

      # The shared Generator says "cache" for the thing a model carries between
      # tokens. Here it is the two recurrences, which are not a cache at all,
      # but the name is what lets that Generator be reused unchanged.
      alias new_cache new_state

      # One token at a time, which is what mamba.c does with a prompt too.
      def prefill(tokens, cache:, prof: Profiler::NULL)
        cache.reset
        logits = nil
        Array(tokens).flatten.each { |id| logits = decode(id, cache: cache, prof: prof) }
        logits
      end

      # Answers [1, vocab_size]. mamba.c computes the rounded width because the
      # table is stored that way, then samples over vocab_size (mamba.c:479 and
      # its sampler); the tail belongs to no token.
      #
      # position is unused: the state carries it, unlike a KV cache that has to
      # be told where to write. It is accepted so the shared Generator's call
      # matches.
      def decode(token_id, _position = nil, cache:, prof: Profiler::NULL)
        unless token_id.is_a?(Integer) && token_id >= 0 && token_id < @config.rounded_vocab_size
          raise Error, "invalid token id #{token_id.inspect}"
        end

        x = prof.section(:embed) do
          Ops.contiguous(@embedding[token_id, true]).reshape!(1, @config.dim)
        end
        @layers.each_with_index do |weights, layer|
          x = prof.section(:block) { block(x, weights, layer, cache, prof) }
        end
        cache.advance
        x = prof.section(:final_norm) { Ops.rmsnorm(x, @final_norm, eps: RMSNORM_EPS) }
        logits = prof.section(:unembed) do
          @lm_head.dot(x.reshape!(@config.dim, 1)).reshape!(1, @config.rounded_vocab_size)
        end
        Ops.row_slice(logits, 0...@config.vocab_size)
      end

      # A shared classifier is counted once. The table is kept in the stored
      # [out, in] order that mamba.c reads in place, so nothing is copied.
      def parameter_bytes
        tensors = [@embedding, @final_norm, @lm_head] + @layers.flat_map(&:values)
        XF::ELEMENT_BYTE_SIZE * tensors.uniq.sum(&:size)
      end

      private

      # mamba.c:377 forward_layer, then the residual mamba.c:481 folds in.
      def block(x, w, layer, state, prof)
        h = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:norm], eps: RMSNORM_EPS) }
        xz = prof.section(:gemm) { h.dot(w[:in_proj_t]) }
        inner = @config.d_inner
        # Selecting the row with an integer answers a contiguous view on both
        # backends, so the halves cost nothing. Taking it with true instead
        # gives Numo something it calls non-contiguous, which needs a copy.
        xc = xz[0, 0...inner]
        z = xz[0, inner...(2 * inner)]

        conv = prof.section(:conv) { shift_in(state.conv(layer), xc) }
        state.conv = [layer, conv]
        # mulsum rather than * then sum: one kernel instead of two, and no
        # [1536, 4] product in between. It folds in a different order, so the
        # token sequence is checked on both backends.
        xc = prof.section(:conv) do
          Ops.silu(conv.mulsum(w[:conv1d_weight], axis: 1) + w[:conv1d_bias])
        end

        # reshape! rather than reshape: reshape copies, which is a kernel, and
        # xc is ours to bend. It goes row, column, row and ends as it started.
        x_db = prof.section(:gemm) { xc.reshape!(1, inner).dot(w[:x_proj_t]) }
        dt, b, c = split_x_db(x_db)

        dt = prof.section(:gemm) { dt.dot(w[:dt_proj_t]) + w[:dt_proj_bias] }
        dt = prof.section(:softplus) { softplus(dt).reshape!(inner, 1) }

        xc.reshape!(inner, 1)
        ssm = prof.section(:ssm) do
          MATH.exp(dt * w[:a]).then { |da| state.ssm(layer) * da + xc * (dt * b) }
        end
        state.ssm = [layer, ssm]

        y = prof.section(:ssm) { ssm.mulsum(c, axis: 1) + w[:d] * xc.reshape!(inner) }
        y = prof.section(:silu) { y * Ops.silu(z) }
        # x is the residual and is handed to the GEMM as C, which overwrites
        # it. Nothing reads it after this: the caller rebinds x to what comes
        # back, and rmsnorm above answered a new array rather than a view.
        prof.section(:gemm) { Ops.linear_add(y.reshape!(1, inner), w[:out_proj_t], x) }
      end

      # mamba.c:269 shift_matrix_left then mamba.c:278 update_last_column. The
      # copy goes into a fresh buffer rather than over itself, which the
      # in-place spelling would make an overlapping read.
      #
      # allocate rather than zeros: every one of the d_conv columns is written
      # below, so filling with zeros first is a kernel whose result never
      # survives. XF.new alone answers unallocated and the writes would fail.
      def shift_in(conv, xc)
        width = @config.d_conv
        moved = XF.new(@config.d_inner, width).allocate
        moved[true, 0...(width - 1)] = conv[true, 1...width]
        moved[true, width - 1] = xc
        moved
      end

      # dt, B and C come out of one row (mamba.c:425), as views.
      def split_x_db(x_db)
        rank = @config.dt_rank
        state = @config.d_state
        [x_db[0, 0...rank].reshape!(1, rank),
         x_db[0, rank...(rank + state)],
         x_db[0, (rank + state)...(rank + 2 * state)]]
      end

      # mamba.c:222, which guards nothing: past ~88 its expf is Inf and so is
      # the answer. Clipping and adding the excess back was three kernels spent
      # on a case that does not arise here (the largest argument over 64 tokens
      # of mamba-130m is 26.5) and that the reference does not handle either.
      #
      # log1p would be a kernel cheaper still, but it is a different number:
      # the reference rounds 1.0 + exp(x) before taking the log.
      # The fused kernel folds in float and rounds once, so it keeps a value
      # where the operators overflow. It takes the sum first, as the reference
      # does, so the answer is the same wherever the operators have one.
      FUSED_SOFTPLUS = XF::Math.respond_to?(:softplus)

      def softplus(x)
        return MATH.softplus(x) if FUSED_SOFTPLUS

        MATH.log(1.0 + MATH.exp(x))
      end

      # mamba.c:234 groups the multiplications x * weight * ss where llama2.c
      # writes weight * (ss * x), so Ops.rmsnorm is a different association.
      # Taken anyway: the token sequence is unchanged on both backends, and the
      # logits move to 4.272e-04 (Cumo, the same) and 3.738e-04 (Numo, closer),
      # against a 5e-3 bound. On Cumo it is one kernel instead of seven.
      RMSNORM_EPS = 1e-5
      # XF::Math rather than XM::NMath: NMath dispatches through
      # method_missing, which cumo measured as an overhead on the same work.
      MATH = XF::Math

      def prepare(checkpoint)
        @embedding = Ops.contiguous(checkpoint[:embedding])
        @final_norm = Ops.contiguous(checkpoint[:final_norm])
        # [out, in] as stored, so decode multiplies from the left. Transposing
        # instead would hold a shared classifier twice (147 MiB for mamba-130m).
        @lm_head = @config.shared_classifier ? @embedding : Ops.contiguous(checkpoint[:lm_head])
        @layers = Array.new(@config.num_layers) { |l| layer_weights(checkpoint, l) }
      end

      # Stored [out, in]; every matmul here wants [in, out].
      def layer_weights(checkpoint, layer)
        {
          norm: Ops.contiguous(checkpoint[:norm][layer, true]),
          in_proj_t: Ops.contiguous(checkpoint[:in_proj][layer, true, true].transpose),
          conv1d_weight: Ops.contiguous(checkpoint[:conv1d_weight][layer, true, true]),
          conv1d_bias: Ops.contiguous(checkpoint[:conv1d_bias][layer, true]),
          x_proj_t: Ops.contiguous(checkpoint[:x_proj][layer, true, true].transpose),
          dt_proj_t: Ops.contiguous(checkpoint[:dt_proj_weight][layer, true, true].transpose),
          dt_proj_bias: Ops.contiguous(checkpoint[:dt_proj_bias][layer, true]),
          a: Ops.contiguous(checkpoint[:a][layer, true, true]),
          d: Ops.contiguous(checkpoint[:d][layer, true]),
          out_proj_t: Ops.contiguous(checkpoint[:out_proj][layer, true, true].transpose)
        }
      end
    end
  end
end
