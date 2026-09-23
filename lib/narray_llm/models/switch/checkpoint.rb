# frozen_string_literal: true

module NArrayLLM
  # Switch Transformer 固有。共有するものは lib/narray_llm/ の直下にある。
  module Switch
    Config = Struct.new(:num_layers, :num_decoder_layers, :d_model, :d_ff, :d_kv,
                        :num_heads, :num_experts, :expert_capacity,
                        :encoder_sparse_step, :decoder_sparse_step,
                        :relative_attention_num_buckets, :relative_attention_max_distance,
                        :layer_norm_epsilon, :vocab_size, :pad_token_id, :eos_token_id,
                        :decoder_start_token_id, :dense_act_fn, :is_gated_act,
                        :router_bias, :router_jitter_noise, keyword_init: true) do
      # Sparse layers are the ones the step lands on, counting from 1, which is
      # blocks 1, 3, 5, ... for a step of 2 (modeling_switch_transformers.py).
      def sparse_encoder_layer?(block)
        (block % encoder_sparse_step) == 1
      end

      def sparse_decoder_layer?(block)
        (block % decoder_sparse_step) == 1
      end

      def inner_dim
        num_heads * d_kv
      end

      # T5 shares one relative bias across a stack, held on its first block.
      def relative_bias_block
        0
      end
    end

    # Reader for the safetensors this repository's export_switch.py writes from
    # google/switch-base-8. Format and provenance: docs/checkpoint-format-switch.md
    class Checkpoint
      SHARED = 'shared.weight'
      # Four names point at one storage in the published pickle, so the export
      # writes it once. Anything asking for the other three gets this.
      TIED = %w[encoder.embed_tokens.weight decoder.embed_tokens.weight lm_head.weight].freeze

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
        @store[TIED.include?(name) ? SHARED : name]
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
        dir = path.sub(/\.safetensors\z/, '')
        File.join(dir, 'config.json')
      end

      def self.read_config(path)
        raise FormatError, "#{path}: not found" unless File.exist?(path)

        raw = JSON.parse(File.read(path))
        Config.new(
          num_layers: raw.fetch('num_layers'), num_decoder_layers: raw.fetch('num_decoder_layers'),
          d_model: raw.fetch('d_model'), d_ff: raw.fetch('d_ff'), d_kv: raw.fetch('d_kv'),
          num_heads: raw.fetch('num_heads'), num_experts: raw.fetch('num_experts'),
          expert_capacity: raw.fetch('expert_capacity'),
          encoder_sparse_step: raw.fetch('encoder_sparse_step'),
          decoder_sparse_step: raw.fetch('decoder_sparse_step'),
          relative_attention_num_buckets: raw.fetch('relative_attention_num_buckets'),
          relative_attention_max_distance: raw.fetch('relative_attention_max_distance'),
          layer_norm_epsilon: raw.fetch('layer_norm_epsilon'), vocab_size: raw.fetch('vocab_size'),
          pad_token_id: raw.fetch('pad_token_id'), eos_token_id: raw.fetch('eos_token_id'),
          decoder_start_token_id: raw.fetch('decoder_start_token_id'),
          dense_act_fn: raw.fetch('dense_act_fn'), is_gated_act: raw.fetch('is_gated_act'),
          router_bias: raw.fetch('router_bias'), router_jitter_noise: raw.fetch('router_jitter_noise')
        )
      end

      # Every name the model will ask for, with the shape the config says it
      # has. Building it from the config rather than from the file is what
      # makes the check mean something.
      def expected_shapes
        c = @config
        shapes = { SHARED => [c.vocab_size, c.d_model] }
        %w[encoder decoder].each do |stack|
          layers = stack == 'encoder' ? c.num_layers : c.num_decoder_layers
          shapes["#{stack}.final_layer_norm.weight"] = [c.d_model]
          layers.times { |b| stack_block_shapes(shapes, stack, b) }
        end
        shapes
      end

      private

      def stack_block_shapes(shapes, stack, block)
        c = @config
        attentions = stack == 'encoder' ? %w[SelfAttention] : %w[SelfAttention EncDecAttention]
        attentions.each_with_index do |kind, i|
          %w[q k v o].each { |p| shapes["#{stack}.block.#{block}.layer.#{i}.#{kind}.#{p}.weight"] = [c.inner_dim, c.d_model] }
          shapes["#{stack}.block.#{block}.layer.#{i}.layer_norm.weight"] = [c.d_model]
        end
        if block == c.relative_bias_block
          shapes["#{stack}.block.#{block}.layer.0.SelfAttention.relative_attention_bias.weight"] =
            [c.relative_attention_num_buckets, c.num_heads]
        end
        ff = attentions.size
        shapes["#{stack}.block.#{block}.layer.#{ff}.layer_norm.weight"] = [c.d_model]
        sparse = stack == 'encoder' ? c.sparse_encoder_layer?(block) : c.sparse_decoder_layer?(block)
        prefix = "#{stack}.block.#{block}.layer.#{ff}.mlp"
        if sparse
          shapes["#{prefix}.router.classifier.weight"] = [c.num_experts, c.d_model]
          c.num_experts.times do |e|
            shapes["#{prefix}.experts.expert_#{e}.wi.weight"] = [c.d_ff, c.d_model]
            shapes["#{prefix}.experts.expert_#{e}.wo.weight"] = [c.d_model, c.d_ff]
          end
        else
          shapes["#{prefix}.wi.weight"] = [c.d_ff, c.d_model]
          shapes["#{prefix}.wo.weight"] = [c.d_model, c.d_ff]
        end
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
