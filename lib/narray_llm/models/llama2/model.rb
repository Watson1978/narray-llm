# frozen_string_literal: true

module NArrayLLM
  module Llama2
    # Llama 2 forward pass, following forward() in llama2.c's run.c.
    #
    # There is no batch dimension: run.c has none, and the reference this is
    # checked against is per-position. Activations are [t, dim] throughout.
    class Model
      attr_reader :config

      def self.load(path)
        new(Checkpoint.load(path))
      end

      def initialize(checkpoint)
        @config = checkpoint.config
        prepare(checkpoint)
      end

      # tokens: an Array of ids. Returns logits [t, vocab_size], or [1, vocab_size]
      # with last_only.
      def forward(tokens, trace: nil, prof: Profiler::NULL, last_only: false, cache: nil)
        ids = Array(tokens).flatten
        t = ids.size
        if t > @config.max_seq_len
          raise Error, "sequence length #{t} exceeds max_seq_len #{@config.max_seq_len}"
        end

        x = prof.section(:embed) { embed(ids) }
        trace&.[]=('embed', x)
        cos_t, sin_t = rope_slice(0, t)
        mask = causal_mask(t)

        @layers.each_with_index do |weights, layer|
          sink = cache && ->(k, v) { cache.append(layer, k, v) }
          x = prof.section(:block) do
            block(x, weights, layer, cos_t, sin_t, mask, trace: trace, prof: prof, kv_sink: sink)
          end
        end

        x = prof.section(:final_norm) { Ops.rmsnorm(x, @rms_final) }
        trace&.[]=('rms_final', x)
        x = Ops.contiguous(x[(t - 1)...t, true]) if last_only
        logits = prof.section(:unembed) { x.dot(@wcls_t) }
        trace&.[]=('logits', logits)
        logits
      end

      def parameter_bytes
        tensors = [@token_embedding, @wcls_t, @rms_final] + @layers.flat_map(&:values)
        XF::ELEMENT_BYTE_SIZE * tensors.sum(&:size)
      end

      # K and V are kv_dim wide, not dim: with grouped-query attention there are
      # fewer key/value heads than query heads, so the cache is kv_mul times
      # smaller than GPT-2's would be at the same dim.
      def new_cache
        KVCache.new(num_layers: @config.num_layers, max_seq_len: @config.max_seq_len,
                    channels: @config.kv_dim)
      end

      # Phase 1 of cached generation: run the prompt through the ordinary forward
      # pass, filling every layer's K and V. Returns the last position's logits.
      def prefill(tokens, cache:, prof: Profiler::NULL)
        cache.reset
        forward(tokens, last_only: true, prof: prof, cache: cache)
      end

      # Phase 2: one new token against the cache. token_id is a Ruby Integer that
      # argmax already read back, so indexing the embedding with it costs no
      # device readback (AGENTS.md).
      def decode(token_id, position, cache:, prof: Profiler::NULL)
        if position >= @config.max_seq_len
          raise Error, "position #{position} exceeds max_seq_len #{@config.max_seq_len}"
        end

        x = prof.section(:embed) { decode_embed(token_id) }
        cos_t, sin_t = rope_slice(position, 1)
        @layers.each_with_index do |weights, layer|
          x = prof.section(:block) { decode_block(x, weights, layer, cos_t, sin_t, cache, prof) }
        end
        x = prof.section(:final_norm) { Ops.rmsnorm(x, @rms_final) }
        prof.section(:unembed) { x.dot(@wcls_t) }
      end

      private

      def decode_embed(token_id)
        unless token_id.is_a?(Integer) && token_id >= 0 && token_id < @config.vocab_size
          raise Error, "invalid token id #{token_id.inspect}"
        end

        @token_embedding[token_id, true].reshape!(1, @config.dim)
      end

      def decode_block(x, w, layer, cos_t, sin_t, cache, prof)
        h = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_att_weight]) }
        qkv = prof.section(:gemm) { h.dot(w[:wqkv_t]) }
        q, k, v = qkv_spans.map { |span| Ops.row_slice(qkv, span) }

        q = prof.section(:rope) { Ops.rope(q, cos_t, sin_t, num_heads: @config.num_heads) }
        k = prof.section(:rope) { Ops.rope(k, cos_t, sin_t, num_heads: @config.num_kv_heads) }
        prof.section(:cache) { cache.append(layer, k, v) }

        keys, values = cache.view(layer)
        attn = Ops.decode_attention(q, keys, values, num_heads: @config.num_heads,
                                    num_kv_heads: @config.num_kv_heads, prof: prof)
        attproj = prof.section(:gemm) { attn.dot(w[:wo_t]) }
        x = prof.section(:residual) { x + attproj }

        h2 = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_ffn_weight]) }
        gates = prof.section(:gemm) { h2.dot(w[:w13_t]) }
        w1h, w3h = gate_spans.map { |span| Ops.row_slice(gates, span) }
        swiglu = prof.section(:silu) { Ops.silu(w1h) * w3h }
        ffn = prof.section(:gemm) { swiglu.dot(w[:w2_t]) }
        prof.section(:residual) { x + ffn }
      end

      # run.c:245-330. Pre-norm, so the residual carries the unnormalized stream.
      def block(x, w, layer, cos_t, sin_t, mask, trace: nil, prof: Profiler::NULL, kv_sink: nil)
        t = x.shape[0]
        prefix = "L#{layer}/"

        h = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_att_weight]) }
        trace&.[]=("#{prefix}rms_att", h)

        # With more than one row a column slice is not contiguous, and rope and
        # heads_first both reshape! their input, so this path copies what decode
        # can use as a view.
        qkv = prof.section(:gemm) { h.dot(w[:wqkv_t]) }
        q, k, v = qkv_spans.map { |span| Ops.contiguous(qkv[true, span]) }
        if trace
          trace["#{prefix}q_pre_rope"] = q
          trace["#{prefix}k_pre_rope"] = k
          trace["#{prefix}v"] = v
        end

        q = prof.section(:rope) { Ops.rope(q, cos_t, sin_t, num_heads: @config.num_heads) }
        k = prof.section(:rope) { Ops.rope(k, cos_t, sin_t, num_heads: @config.num_kv_heads) }
        if trace
          trace["#{prefix}q"] = q
          trace["#{prefix}k"] = k
        end
        # After RoPE, not before: run.c rotates s->k in place and s->k already
        # points into the cache row (run.c:259, :279), so what is stored is the
        # rotated key. Caching the pre-RoPE key would rotate it again by the
        # wrong position on every later step.
        prof.section(:cache) { kv_sink.call(k, v) } if kv_sink

        attn = attention(q, k, v, mask, trace: trace, prefix: prefix, prof: prof)
        trace&.[]=("#{prefix}attn_out", attn)

        attproj = prof.section(:gemm) { attn.dot(w[:wo_t]) }
        trace&.[]=("#{prefix}attproj", attproj)
        x = prof.section(:residual) { x + attproj }
        trace&.[]=("#{prefix}res_att", x)

        h2 = prof.section(:rmsnorm) { Ops.rmsnorm(x, w[:rms_ffn_weight]) }
        trace&.[]=("#{prefix}rms_ffn", h2)

        gates = prof.section(:gemm) { h2.dot(w[:w13_t]) }
        w1h, w3h = gate_spans.map { |span| Ops.contiguous(gates[true, span]) }
        if trace
          trace["#{prefix}w1h"] = w1h
          trace["#{prefix}w3h"] = w3h
        end

        swiglu = prof.section(:silu) { Ops.silu(w1h) * w3h }
        trace&.[]=("#{prefix}swiglu", swiglu)
        ffn = prof.section(:gemm) { swiglu.dot(w[:w2_t]) }
        trace&.[]=("#{prefix}ffn_out", ffn)
        out = prof.section(:residual) { x + ffn }
        trace&.[]=("#{prefix}res_ffn", out)
        out
      end

      # q is [t, num_heads * head_size]; k and v are [t, num_kv_heads * head_size].
      def attention(q, k, v, mask, trace: nil, prefix: '', prof: Profiler::NULL)
        t = q.shape[0]
        heads = @config.num_heads
        hs = @config.head_size
        scale = 1.0 / Math.sqrt(hs)

        queries = prof.section(:attention) { heads_first(q, heads, hs, t) }
        keys = prof.section(:attention) { heads_first(expand_kv(k), heads, hs, t) }
        values = prof.section(:attention) { heads_first(expand_kv(v), heads, hs, t) }

        scores = prof.section(:attention) do
          Ops.batched_dot(queries, Ops.contiguous(keys.transpose(0, 2, 1))) * scale + mask
        end
        weights = prof.section(:softmax) { Ops.softmax_rows(scores) }
        trace&.[]=("#{prefix}att", weights)

        prof.section(:attention) do
          Ops.contiguous(Ops.batched_dot(weights, values).transpose(1, 0, 2)).reshape!(t, heads * hs)
        end
      end

      # [t, c] -> [num_heads, t, head_size]. The transpose leaves a strided view,
      # so it is copied once here rather than per dot() call (AGENTS.md).
      def heads_first(x, heads, head_size, t)
        x_shape = x.shape
        Ops.contiguous(x.reshape!(t, heads, head_size).transpose(1, 0, 2))
      ensure
        x.reshape!(*x_shape)
      end

      def expand_kv(x)
        Ops.repeat_kv_heads(x, num_kv_heads: @config.num_kv_heads, kv_mul: @config.kv_mul)
      end

      # Rows [from, from + count) of the tables, built once at max_seq_len so a
      # generation loop does not rebuild them every step.
      def rope_slice(from, count)
        @rope_cos ||= nil
        @rope_cos, @rope_sin = Ops.rope_tables(@config.max_seq_len, @config.head_size) if @rope_cos.nil?
        span = from...(from + count)
        [Ops.contiguous(@rope_cos[span, true]), Ops.contiguous(@rope_sin[span, true])]
      end

      def causal_mask(t)
        @full_mask ||= Ops.causal_mask(@config.max_seq_len)
        Ops.contiguous(@full_mask[0...t, 0...t])
      end

      def embed(ids)
        Ops.gather_rows(@token_embedding, ids)
      end

      def prepare(checkpoint)
        @token_embedding = Ops.contiguous(checkpoint[:token_embedding_table])
        @wcls_t = Ops.contiguous(checkpoint[:wcls].transpose)
        @rms_final = Ops.contiguous(checkpoint[:rms_final_weight])
        @layers = Array.new(@config.num_layers) { |l| layer_weights(checkpoint, l) }
      end

      # Stored [out, in]; every GEMM here wants [in, out]. Q, K and V share an
      # input and so do the two FFN gates, so they are concatenated into one
      # matrix each: seven products a layer become four. A column slice of the
      # result is contiguous while there is one row, which is the decode case;
      # the prefill path copies (see block).
      def layer_weights(checkpoint, layer)
        {
          rms_att_weight: Ops.contiguous(checkpoint[:rms_att_weight][layer, true]),
          wqkv_t: side_by_side(%i[wq wk wv].map { |k| checkpoint[k][layer, true, true].transpose }),
          wo_t: Ops.contiguous(checkpoint[:wo][layer, true, true].transpose),
          rms_ffn_weight: Ops.contiguous(checkpoint[:rms_ffn_weight][layer, true]),
          w13_t: side_by_side(%i[w1 w3].map { |k| checkpoint[k][layer, true, true].transpose }),
          w2_t: Ops.contiguous(checkpoint[:w2][layer, true, true].transpose)
        }
      end

      def side_by_side(parts)
        rows = parts.first.shape[0]
        out = XF.zeros(rows, parts.sum { |m| m.shape[1] })
        at = 0
        parts.each do |m|
          out[true, at...(at + m.shape[1])] = m
          at += m.shape[1]
        end
        out
      end

      def qkv_spans
        dim = @config.dim
        kv = @config.kv_dim
        [0...dim, dim...(dim + kv), (dim + kv)...(dim + 2 * kv)]
      end

      def gate_spans
        hidden = @config.hidden_dim
        [0...hidden, hidden...(2 * hidden)]
      end
    end
  end
end
