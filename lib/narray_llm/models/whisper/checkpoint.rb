# frozen_string_literal: true

module NArrayLLM
  # Whisper 固有。共有するものは lib/narray_llm/ の直下にある。
  module Whisper
    Config = Struct.new(:d_model, :encoder_layers, :decoder_layers,
                        :encoder_attention_heads, :decoder_attention_heads,
                        :encoder_ffn_dim, :decoder_ffn_dim, :num_mel_bins,
                        :max_source_positions, :max_target_positions, :vocab_size,
                        :activation_function, :scale_embedding, :max_length,
                        :decoder_start_token_id, :eos_token_id, :pad_token_id, :bos_token_id,
                        :forced_decoder_ids, :suppress_tokens, :begin_suppress_tokens,
                        keyword_init: true) do
      def head_dim
        d_model / encoder_attention_heads
      end

      # The two convolutions turn num_mel_bins x frames into d_model x
      # max_source_positions, the second one with stride 2.
      def mel_frames
        max_source_positions * 2
      end

      # Whisper leaves the key projection without a bias and gives the other
      # three one (modeling_whisper.py: `bias=False` on k_proj only).
      def projection_has_bias?(part)
        part.to_s != 'k_proj'
      end
    end

    # Reader for the safetensors openai/whisper-tiny publishes. Unlike
    # switch-base-8 there is no conversion step: the file is F32 already and
    # lib/narray_llm/safetensors.rb reads it directly.
    # Format and provenance: docs/checkpoint-format-whisper.md
    class Checkpoint
      EMBED = 'model.decoder.embed_tokens.weight'
      # The output projection is the embedding read the other way; the
      # checkpoint has no proj_out of its own.
      TIED = %w[proj_out.weight].freeze
      PROJECTIONS = %w[q_proj k_proj v_proj out_proj].freeze

      attr_reader :path, :config, :store

      def self.load(path, config_path: nil)
        new(path, config_path: config_path)
      end

      def initialize(path, config_path: nil)
        @path = path
        @config = Checkpoint.read_config(config_path || Checkpoint.config_beside(path))
        @store = Safetensors.new(path)
        validate
      end

      def close
        @store.close
      end

      def [](name)
        @store[TIED.include?(name) ? EMBED : name]
      end

      def include?(name)
        TIED.include?(name) || @store.include?(name)
      end

      def names
        @store.names
      end

      def num_parameters
        @store.names.sum { |n| @store.shape(n).inject(1, :*) }
      end

      def self.config_beside(path)
        File.join(File.dirname(path), 'config.json')
      end

      def self.read_config(path)
        raise FormatError, "#{path}: not found" unless File.exist?(path)

        raw = JSON.parse(File.read(path))
        Config.new(
          d_model: raw.fetch('d_model'), encoder_layers: raw.fetch('encoder_layers'),
          decoder_layers: raw.fetch('decoder_layers'),
          encoder_attention_heads: raw.fetch('encoder_attention_heads'),
          decoder_attention_heads: raw.fetch('decoder_attention_heads'),
          encoder_ffn_dim: raw.fetch('encoder_ffn_dim'), decoder_ffn_dim: raw.fetch('decoder_ffn_dim'),
          num_mel_bins: raw.fetch('num_mel_bins'),
          max_source_positions: raw.fetch('max_source_positions'),
          max_target_positions: raw.fetch('max_target_positions'),
          vocab_size: raw.fetch('vocab_size'),
          activation_function: raw.fetch('activation_function'),
          scale_embedding: raw.fetch('scale_embedding'), max_length: raw.fetch('max_length'),
          decoder_start_token_id: raw.fetch('decoder_start_token_id'),
          eos_token_id: raw.fetch('eos_token_id'), pad_token_id: raw.fetch('pad_token_id'),
          bos_token_id: raw.fetch('bos_token_id'),
          forced_decoder_ids: raw.fetch('forced_decoder_ids'),
          suppress_tokens: raw.fetch('suppress_tokens'),
          begin_suppress_tokens: raw.fetch('begin_suppress_tokens')
        )
      end

      # Every name the model will ask for, with the shape the config says it
      # has. Building it from the config rather than from the file is what
      # makes the check mean something.
      def expected_shapes
        c = @config
        shapes = {
          EMBED => [c.vocab_size, c.d_model],
          'model.decoder.embed_positions.weight' => [c.max_target_positions, c.d_model],
          'model.encoder.embed_positions.weight' => [c.max_source_positions, c.d_model],
          'model.encoder.conv1.weight' => [c.d_model, c.num_mel_bins, 3],
          'model.encoder.conv1.bias' => [c.d_model],
          'model.encoder.conv2.weight' => [c.d_model, c.d_model, 3],
          'model.encoder.conv2.bias' => [c.d_model]
        }
        %w[encoder decoder].each do |stack|
          norm(shapes, "model.#{stack}.layer_norm", c.d_model)
          layers = stack == 'encoder' ? c.encoder_layers : c.decoder_layers
          layers.times { |i| layer_shapes(shapes, stack, i) }
        end
        shapes
      end

      private

      def layer_shapes(shapes, stack, index)
        c = @config
        at = "model.#{stack}.layers.#{index}"
        ffn = stack == 'encoder' ? c.encoder_ffn_dim : c.decoder_ffn_dim
        attentions = stack == 'encoder' ? %w[self_attn] : %w[self_attn encoder_attn]
        attentions.each do |kind|
          PROJECTIONS.each do |part|
            shapes["#{at}.#{kind}.#{part}.weight"] = [c.d_model, c.d_model]
            shapes["#{at}.#{kind}.#{part}.bias"] = [c.d_model] if c.projection_has_bias?(part)
          end
          norm(shapes, "#{at}.#{kind}_layer_norm", c.d_model)
        end
        shapes["#{at}.fc1.weight"] = [ffn, c.d_model]
        shapes["#{at}.fc1.bias"] = [ffn]
        shapes["#{at}.fc2.weight"] = [c.d_model, ffn]
        shapes["#{at}.fc2.bias"] = [c.d_model]
        norm(shapes, "#{at}.final_layer_norm", c.d_model)
      end

      def norm(shapes, prefix, width)
        shapes["#{prefix}.weight"] = [width]
        shapes["#{prefix}.bias"] = [width]
      end

      def validate
        want = expected_shapes
        missing = want.keys - @store.names
        raise FormatError, "#{@path}: missing #{missing.size} tensors, first #{missing.first}" unless missing.empty?

        extra = @store.names - want.keys
        raise FormatError, "#{@path}: unexpected tensors, first #{extra.first}" unless extra.empty?

        want.each do |name, shape|
          got = @store.shape(name)
          raise FormatError, "#{@path}: #{name} is #{got.inspect}, expected #{shape.inspect}" unless got == shape
        end
      end
    end
  end
end
