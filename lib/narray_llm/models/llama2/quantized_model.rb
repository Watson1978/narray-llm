# frozen_string_literal: true

module NArrayLLM
  module Llama2
    # Llama 2 forward pass over an int8 checkpoint, following forward() in
    # llama2.c's runq.c.
    #
    # Only the matrices are quantized. The RMSNorm weights stay fp32, and the
    # embedding table is dequantized once at load because runq.c does the same
    # (runq.c:204) and then reads rows out of it.
    #
    # There is no recompute-the-whole-sequence path: runq.c has none, and the
    # quantization is defined per row, so a prompt is run one token at a time.
    class QuantizedModel
      attr_reader :config, :group_size

      def self.load(path)
        new(QuantizedCheckpoint.load(path))
      end

      # The group sums have to stay exact integers. fp32 holds them, the two
      # 16-bit types do not: group_size * 127 * 127 reaches 1,032,256, which is
      # past fp16's 65,504 ceiling and past the 256 that bf16's 8 mantissa bits
      # represent exactly.
      REDUCED_PRECISION = "int8 needs fp32: a group sum reaches %<max>d, which " \
                          "%<dtype>s cannot hold exactly. Drop DTYPE."

      def initialize(checkpoint)
        @config = checkpoint.config
        @group_size = checkpoint.group_size
        if NArrayLLM.reduced_precision?
          raise Error, format(REDUCED_PRECISION, max: @group_size * 127 * 127,
                                                 dtype: NArrayLLM.dtype_name)
        end

        prepare(checkpoint)
      end

      def new_cache
        KVCache.new(num_layers: @config.num_layers, max_seq_len: @config.max_seq_len,
                    channels: @config.kv_dim)
      end

      # One token at a time, which is what runq.c does with a prompt too.
      def prefill(tokens, cache:, prof: Profiler::NULL)
        cache.reset
        ids = Array(tokens).flatten
        logits = nil
        ids.each_with_index { |id, position| logits = decode(id, position, cache: cache, prof: prof) }
        logits
      end

      def decode(token_id, position, cache:, prof: Profiler::NULL)
        if position >= @config.max_seq_len
          raise Error, "position #{position} exceeds max_seq_len #{@config.max_seq_len}"
        end
        unless token_id.is_a?(Integer) && token_id >= 0 && token_id < @config.vocab_size
          raise Error, "invalid token id #{token_id.inspect}"
        end

        x = prof.section(:embed) { Ops.contiguous(@token_embedding[token_id, true]).reshape!(1, @config.dim) }
        cos_t, sin_t = rope_slice(position)
        @layers.each_with_index do |weights, layer|
          x = prof.section(:block) { block(x, weights, layer, cos_t, sin_t, cache, prof) }
        end
        x = prof.section(:final_norm) { Ops.rmsnorm(x, @rms_final) }
        prof.section(:unembed) { qlinear(@wcls, x, prof) }
      end

      def parameter_bytes
        matrices = [@wcls] + @layers.flat_map(&:values).grep(QuantizedCheckpoint::Quantized)
        plain = [@token_embedding, @rms_final] +
                @layers.flat_map { |w| [w[:rms_att_weight], w[:rms_ffn_weight]] }
        matrices.uniq.sum { |w| w.q.size + XF::ELEMENT_BYTE_SIZE * w.scales.size } +
          XF::ELEMENT_BYTE_SIZE * plain.sum(&:size)
      end

      private

      def block(x, w, layer, cos_t, sin_t, cache, prof)
        h = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_att_weight]) }
        hq = prof.section(:quantize) { Ops.quantize_groups(h, @group_size) }
        q = prof.section(:gemm) { Ops.qmatmul(w[:wq], *hq) }
        k = prof.section(:gemm) { Ops.qmatmul(w[:wk], *hq) }
        v = prof.section(:gemm) { Ops.qmatmul(w[:wv], *hq) }

        q = prof.section(:rope) { Ops.rope(q, cos_t, sin_t, num_heads: @config.num_heads) }
        k = prof.section(:rope) { Ops.rope(k, cos_t, sin_t, num_heads: @config.num_kv_heads) }
        prof.section(:cache) { cache.append(layer, k, v) }

        keys, values = cache.view(layer)
        attn = Ops.decode_attention(q, keys, values, num_heads: @config.num_heads,
                                    num_kv_heads: @config.num_kv_heads, prof: prof)
        attproj = prof.section(:gemm) { qlinear(w[:wo], attn, prof) }
        x = prof.section(:residual) { x + attproj }

        h2 = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_ffn_weight]) }
        h2q = prof.section(:quantize) { Ops.quantize_groups(h2, @group_size) }
        w1h = prof.section(:gemm) { Ops.qmatmul(w[:w1], *h2q) }
        w3h = prof.section(:gemm) { Ops.qmatmul(w[:w3], *h2q) }
        swiglu = prof.section(:silu) { Ops.silu(w1h) * w3h }
        ffn = prof.section(:gemm) { qlinear(w[:w2], swiglu, prof) }
        prof.section(:residual) { x + ffn }
      end

      # runq.c quantizes once per distinct activation and shares it across the
      # matmuls that read it: q/k/v off one (runq.c:367), w1/w3 off another
      # (runq.c:450). This is for the three that have no partner.
      def qlinear(weight, x, prof)
        xq, xs = prof.section(:quantize) { Ops.quantize_groups(x, @group_size) }
        Ops.qmatmul(weight, xq, xs)
      end

      def rope_slice(position)
        @rope_cos ||= nil
        @rope_cos, @rope_sin = Ops.rope_tables(@config.max_seq_len, @config.head_size) if @rope_cos.nil?
        span = position...(position + 1)
        [Ops.contiguous(@rope_cos[span, true]), Ops.contiguous(@rope_sin[span, true])]
      end

      def prepare(checkpoint)
        @token_embedding = Ops.contiguous(dequantize(checkpoint[:q_tokens]))
        @rms_final = Ops.contiguous(checkpoint[:rms_final_weight])
        @wcls = to_device(checkpoint[:wcls])
        @layers = Array.new(@config.num_layers) { |l| layer_weights(checkpoint, l) }
      end

      def layer_weights(checkpoint, layer)
        {
          rms_att_weight: Ops.contiguous(checkpoint[:rms_att_weight][layer, true]),
          rms_ffn_weight: Ops.contiguous(checkpoint[:rms_ffn_weight][layer, true]),
          wq: to_device(checkpoint[:wq][layer]),
          wk: to_device(checkpoint[:wk][layer]),
          wv: to_device(checkpoint[:wv][layer]),
          wo: to_device(checkpoint[:wo][layer]),
          w1: to_device(checkpoint[:w1][layer]),
          w2: to_device(checkpoint[:w2][layer]),
          w3: to_device(checkpoint[:w3][layer])
        }
      end

      # The int8 block stays int8: the products are taken against a float
      # activation, which promotes inside the kernel.
      def to_device(weight)
        QuantizedCheckpoint::Quantized.new(weight.q, Ops.contiguous(weight.scales), weight.shape)
      end

      def dequantize(weight)
        groups = weight.scales.shape[1]
        (XF.cast(weight.q) * weight.scales.reshape(weight.shape[0], groups, 1))
          .reshape(*weight.shape)
      end
    end
  end
end
