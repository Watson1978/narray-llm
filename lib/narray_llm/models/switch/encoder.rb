# frozen_string_literal: true

module NArrayLLM
  module Switch
    # The encoder stack. Bidirectional, so every token sees every other one
    # and the relative bias keeps the sign of the distance.
    class Encoder < Stack
      # ids: an Array of token ids. Answers [tokens, d_model].
      #
      # trace, when given a Hash, collects each sparse layer's routing as a
      # [tokens, experts] one hot with the capacity already applied, under the
      # same names the reference dump uses. A token that went nowhere is an
      # all zero row. Only the tests ask for it.
      def forward(ids, prof: Profiler::NULL, trace: nil)
        tokens = Array(ids).flatten.map(&:to_i)
        x = prof.section(:embed) { Ops.contiguous(@embedding[tokens, true]) }
        bias = prof.section(:bias) { relative_bias(tokens.size) }
        @blocks.each_with_index { |w, i| x = prof.section(:block) { block(x, w, bias, prof, trace, i) } }
        prof.section(:final_norm) { norm(x, @final_norm) }
      end

      def relative_bias(length)
        bias_from(@bias_table, length, length, bidirectional: true)
      end

      private

      def block(x, w, bias, prof, trace, index)
        h = norm(x, w[:attention_norm])
        keys, values = project_kv_heads(h, w, '')
        x += prof.section(:attention) { attend(h, w, '', keys, values, bias) }
        h = norm(x, w[:ff_norm])
        x + prof.section(:ff) { feed_forward(h, w, prof, trace, index) }
      end

      def prepare(checkpoint)
        c = @config
        @embedding = Ops.contiguous(checkpoint[Checkpoint::SHARED])
        @final_norm = Ops.contiguous(checkpoint['encoder.final_layer_norm.weight'])
        @bias_table = Ops.contiguous(
          checkpoint['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
        )
        @blocks = Array.new(c.num_layers) { |block| block_weights(checkpoint, block) }
      end

      def block_weights(checkpoint, block)
        name = "encoder.block.#{block}"
        weights = attention_weights(checkpoint, "#{name}.layer.0", '', 'SelfAttention')
        weights[:attention_norm] = Ops.contiguous(checkpoint["#{name}.layer.0.layer_norm.weight"])
        weights[:ff_norm] = Ops.contiguous(checkpoint["#{name}.layer.1.layer_norm.weight"])
        weights.merge(feed_forward_weights(checkpoint, "#{name}.layer.1.mlp",
                                           @config.sparse_encoder_layer?(block)))
      end
    end
  end
end
