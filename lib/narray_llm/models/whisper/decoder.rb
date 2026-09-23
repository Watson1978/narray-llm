# frozen_string_literal: true

module NArrayLLM
  module Whisper
    # What the decoder carries between tokens: its own growing K and V, and
    # the encoder's, which are projected once per cross-attention layer and
    # then never change.
    class Cache
      attr_reader :self_attention, :cross

      def initialize(self_attention, cross)
        @self_attention = self_attention
        @cross = cross
      end

      def length
        @self_attention.length
      end

      def reset
        @self_attention.reset
        self
      end
    end

    # The decoder: learned positions, then four pre-norm layers that each
    # attend to themselves, then to the encoder, then feed forward.
    class Decoder
      attr_reader :config

      def initialize(checkpoint)
        @config = checkpoint.config
        prepare(checkpoint)
      end

      # Answers [1, d_model] for one token. The causal mask is implicit: the
      # cache holds only what came before, so the row of scores covers exactly
      # the allowed keys.
      def decode(token_id, cache:, prof: Profiler::NULL)
        c = @config
        unless token_id.is_a?(Integer) && token_id >= 0 && token_id < c.vocab_size
          raise Error, "invalid token id #{token_id.inspect}"
        end

        position = cache.length
        raise Error, "position #{position} past #{c.max_target_positions}" if position >= c.max_target_positions

        x = prof.section(:embed) do
          Ops.contiguous(@embedding[[token_id], true]) + @positions[position...(position + 1), true]
        end
        @layers.each_with_index { |w, i| x = prof.section(:layer) { layer(x, w, cache, i, prof) } }
        prof.section(:final_norm) { norm(x, @final_norm) }
      end

      # encoder_states: [positions, d_model] from the encoder.
      def new_cache(encoder_states)
        cross = @layers.map do |w|
          [heads_major(linear(encoder_states, w[:cross_k])),
           heads_major(linear(encoder_states, w[:cross_v]))]
        end
        Cache.new(
          KVCache.new(num_layers: @config.decoder_layers,
                      max_seq_len: @config.max_target_positions,
                      channels: @config.d_model),
          cross
        )
      end

      private

      def layer(x, w, cache, index, prof)
        h = norm(x, w[:attention_norm])
        keys = linear(h, w[:k])
        values = linear(h, w[:v])
        cache.self_attention.append(index, keys, values)
        past = cache.self_attention.view(index)
        x += prof.section(:attention) do
          attend(h, w, '', heads_major(past[0]), heads_major(past[1]))
        end

        h = norm(x, w[:cross_norm])
        cross_keys, cross_values = cache.cross[index]
        x += prof.section(:cross) { attend(h, w, 'cross_', cross_keys, cross_values) }

        h = norm(x, w[:ff_norm])
        x + prof.section(:ff) { linear(Ops.gelu_erf(linear(h, w[:fc1])), w[:fc2]) }
      end

      def norm(x, weight)
        Ops.layernorm(x, weight[:weight], weight[:bias], eps: Encoder::LAYER_NORM_EPS)
      end

      def linear(x, weight)
        y = x.dot(weight[:weight])
        weight[:bias].nil? ? y : y + weight[:bias]
      end

      # keys and values come in head major as [heads, keys, head_dim].
      def attend(h, w, prefix, keys, values)
        q = heads_major(linear(h, w[:"#{prefix}q"]) * @scaling)
        parts = Array.new(@config.decoder_attention_heads) do |head|
          scores = q[head, true, true].dot(keys[head, true, true].transpose)
          Ops.softmax_rows(scores).dot(values[head, true, true])
        end
        linear(Ops.contiguous(XF.hstack(parts)), w[:"#{prefix}out"])
      end

      def heads_major(x)
        rows = x.shape[0]
        Ops.contiguous(x.reshape!(rows, @config.decoder_attention_heads, @config.head_dim)
                        .transpose(1, 0, 2))
      end

      def prepare(checkpoint)
        c = @config
        @scaling = c.head_dim**-0.5
        @embedding = Ops.contiguous(checkpoint[Checkpoint::EMBED])
        @positions = Ops.contiguous(checkpoint['model.decoder.embed_positions.weight'])
        @final_norm = norm_weights(checkpoint, 'model.decoder.layer_norm')
        @layers = Array.new(c.decoder_layers) { |i| layer_weights(checkpoint, i) }
      end

      def layer_weights(checkpoint, index)
        at = "model.decoder.layers.#{index}"
        weights = { attention_norm: norm_weights(checkpoint, "#{at}.self_attn_layer_norm"),
                    cross_norm: norm_weights(checkpoint, "#{at}.encoder_attn_layer_norm"),
                    ff_norm: norm_weights(checkpoint, "#{at}.final_layer_norm"),
                    fc1: linear_weights(checkpoint, "#{at}.fc1"),
                    fc2: linear_weights(checkpoint, "#{at}.fc2") }
        { 'self_attn' => '', 'encoder_attn' => 'cross_' }.each do |kind, prefix|
          %w[q k v out].each do |part|
            weights[:"#{prefix}#{part}"] = linear_weights(checkpoint, "#{at}.#{kind}.#{part}_proj")
          end
        end
        weights
      end

      def linear_weights(checkpoint, name)
        bias = checkpoint.include?("#{name}.bias") ? Ops.contiguous(checkpoint["#{name}.bias"]) : nil
        { weight: Ops.contiguous(checkpoint["#{name}.weight"].transpose), bias: bias }
      end

      def norm_weights(checkpoint, name)
        { weight: Ops.contiguous(checkpoint["#{name}.weight"]),
          bias: Ops.contiguous(checkpoint["#{name}.bias"]) }
      end
    end
  end
end
