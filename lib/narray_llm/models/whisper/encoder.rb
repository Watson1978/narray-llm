# frozen_string_literal: true

module NArrayLLM
  module Whisper
    # The encoder: two convolutions over the mel bins, a fixed sinusoidal
    # position table, then four pre-norm transformer layers.
    #
    # Whisper's flavour differs from the other models here in three places.
    # The norm is LayerNorm with a bias, not RMSNorm. The activation is the
    # erf gelu, not the tanh one GPT-2 uses. And the key projection carries no
    # bias while the other three do.
    class Encoder
      LAYER_NORM_EPS = 1.0e-5

      attr_reader :config, :spelling

      def initialize(checkpoint, spelling: :shift)
        @config = checkpoint.config
        @spelling = spelling
        prepare(checkpoint)
      end

      # mel: [num_mel_bins, frames] the way the feature extractor answers it.
      # Answers [max_source_positions, d_model].
      def forward(mel, prof: Profiler::NULL)
        c = @config
        unless mel.shape == [c.num_mel_bins, c.mel_frames]
          raise Error, "expected mel #{[c.num_mel_bins, c.mel_frames].inspect}, got #{mel.shape.inspect}"
        end

        x = prof.section(:transpose) { Ops.contiguous(mel.transpose) }
        x = prof.section(:conv) { Ops.gelu_erf(@conv1.call(x, prof: prof)) }
        x = prof.section(:conv) { Ops.gelu_erf(@conv2.call(x, prof: prof)) }
        x += @positions
        @layers.each { |w| x = prof.section(:layer) { layer(x, w, prof) } }
        prof.section(:final_norm) { norm(x, @final_norm) }
      end

      private

      def layer(x, w, prof)
        h = norm(x, w[:attention_norm])
        x += prof.section(:attention) { attend(h, w) }
        h = norm(x, w[:ff_norm])
        x + prof.section(:ff) { Ops.gelu_erf(linear(h, w[:fc1])).dot(w[:fc2][:weight]) + w[:fc2][:bias] }
      end

      def norm(x, weight)
        Ops.layernorm(x, weight[:weight], weight[:bias], eps: LAYER_NORM_EPS)
      end

      def linear(x, weight)
        y = x.dot(weight[:weight])
        weight[:bias].nil? ? y : y + weight[:bias]
      end

      # The query is scaled before the heads split, which is where
      # WhisperAttention puts it. Nothing masks the encoder: every frame sees
      # every other one.
      def attend(h, w)
        q = heads_major(linear(h, w[:q]) * @scaling)
        k = heads_major(linear(h, w[:k]))
        v = heads_major(linear(h, w[:v]))
        parts = Array.new(@config.encoder_attention_heads) do |head|
          scores = q[head, true, true].dot(k[head, true, true].transpose)
          Ops.softmax_rows(scores).dot(v[head, true, true])
        end
        linear(Ops.contiguous(XF.hstack(parts)), w[:out])
      end

      def heads_major(x)
        rows = x.shape[0]
        Ops.contiguous(x.reshape!(rows, @config.encoder_attention_heads, @config.head_dim)
                        .transpose(1, 0, 2))
      end

      def prepare(checkpoint)
        c = @config
        @scaling = c.head_dim**-0.5
        @conv1 = Conv1d.new(checkpoint['model.encoder.conv1.weight'],
                            checkpoint['model.encoder.conv1.bias'],
                            stride: 1, padding: 1, spelling: @spelling)
        @conv2 = Conv1d.new(checkpoint['model.encoder.conv2.weight'],
                            checkpoint['model.encoder.conv2.bias'],
                            stride: 2, padding: 1, spelling: @spelling)
        @positions = Ops.contiguous(checkpoint['model.encoder.embed_positions.weight'])
        @final_norm = norm_weights(checkpoint, 'model.encoder.layer_norm')
        @layers = Array.new(c.encoder_layers) { |i| layer_weights(checkpoint, i) }
      end

      def layer_weights(checkpoint, index)
        at = "model.encoder.layers.#{index}"
        { attention_norm: norm_weights(checkpoint, "#{at}.self_attn_layer_norm"),
          ff_norm: norm_weights(checkpoint, "#{at}.final_layer_norm"),
          q: linear_weights(checkpoint, "#{at}.self_attn.q_proj"),
          k: linear_weights(checkpoint, "#{at}.self_attn.k_proj"),
          v: linear_weights(checkpoint, "#{at}.self_attn.v_proj"),
          out: linear_weights(checkpoint, "#{at}.self_attn.out_proj"),
          fc1: linear_weights(checkpoint, "#{at}.fc1"),
          fc2: linear_weights(checkpoint, "#{at}.fc2") }
      end

      # Stored [out, in]; every matmul here wants [in, out]. k_proj has no
      # bias and the other three do.
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
