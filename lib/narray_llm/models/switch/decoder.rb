# frozen_string_literal: true

module NArrayLLM
  module Switch
    # What the decoder carries between tokens: its own growing K and V, and
    # the encoder's K and V, which are projected once per cross-attention
    # layer and then never change.
    class Cache
      attr_reader :self_attention, :cross, :encoder_length

      def initialize(self_attention, cross, encoder_length)
        @self_attention = self_attention
        @cross = cross
        @encoder_length = encoder_length
      end

      def length
        @self_attention.length
      end

      def reset
        @self_attention.reset
        self
      end
    end

    # The decoder stack. One directional, so the relative bias folds
    # everything ahead onto zero, and every block also attends to the encoder
    # with no position bias at all.
    class Decoder < Stack
      # Answers [1, d_model] for one token. The causal mask is implicit: the
      # cache holds only what came before, so the row of scores covers exactly
      # the allowed keys.
      def decode(token_id, cache:, prof: Profiler::NULL, trace: nil)
        unless token_id.is_a?(Integer) && token_id >= 0 && token_id < @config.vocab_size
          raise Error, "invalid token id #{token_id.inspect}"
        end

        position = cache.length
        x = prof.section(:embed) { Ops.contiguous(@embedding[[token_id], true]) }
        bias = prof.section(:bias) { relative_bias(position + 1) }
        @blocks.each_with_index do |w, i|
          x = prof.section(:block) { block(x, w, bias, cache, i, prof, trace) }
        end
        prof.section(:final_norm) { norm(x, @final_norm) }
      end

      # encoder_states: [tokens, d_model] from the encoder.
      def new_cache(encoder_states, max_seq_len:)
        # Laid out head major once, so a decode step copies nothing for the
        # cross attention at all.
        cross = @blocks.map { |w| project_kv_heads(encoder_states, w, 'cross_') }
        Cache.new(
          KVCache.new(num_layers: @config.num_decoder_layers, max_seq_len: max_seq_len,
                      channels: @config.inner_dim),
          cross, encoder_states.shape[0]
        )
      end

      # One query row at `length - 1`, attending over everything up to it.
      def relative_bias(length)
        bias_from(@bias_table, 1, length, bidirectional: false, offset: length - 1)
      end

      private

      def block(x, w, bias, cache, index, prof, trace)
        h = norm(x, w[:attention_norm])
        keys, values = project_kv(h, w, '')
        cache.self_attention.append(index, keys, values)
        past = cache.self_attention.view(index).map { |a| heads_major(a) }
        x += prof.section(:attention) { attend(h, w, '', past[0], past[1], bias) }

        h = norm(x, w[:cross_norm])
        cross_keys, cross_values = cache.cross[index]
        x += prof.section(:cross) { attend(h, w, 'cross_', cross_keys, cross_values, @zero_bias) }

        h = norm(x, w[:ff_norm])
        x + prof.section(:ff) { feed_forward(h, w, prof, trace, index) }
      end

      def prepare(checkpoint)
        c = @config
        @embedding = Ops.contiguous(checkpoint[Checkpoint::SHARED])
        @final_norm = Ops.contiguous(checkpoint['decoder.final_layer_norm.weight'])
        @bias_table = Ops.contiguous(
          checkpoint['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
        )
        @blocks = Array.new(c.num_decoder_layers) { |block| block_weights(checkpoint, block) }
      end

      def block_weights(checkpoint, block)
        name = "decoder.block.#{block}"
        weights = attention_weights(checkpoint, "#{name}.layer.0", '', 'SelfAttention')
        weights.merge!(attention_weights(checkpoint, "#{name}.layer.1", 'cross_', 'EncDecAttention'))
        weights[:attention_norm] = Ops.contiguous(checkpoint["#{name}.layer.0.layer_norm.weight"])
        weights[:cross_norm] = Ops.contiguous(checkpoint["#{name}.layer.1.layer_norm.weight"])
        weights[:ff_norm] = Ops.contiguous(checkpoint["#{name}.layer.2.layer_norm.weight"])
        weights.merge(feed_forward_weights(checkpoint, "#{name}.layer.2.mlp",
                                           @config.sparse_decoder_layer?(block)))
      end
    end
  end
end
