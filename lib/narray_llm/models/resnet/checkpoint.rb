# frozen_string_literal: true

module NArrayLLM
  # ResNet 固有。共有するものは lib/narray_llm/ の直下にある。
  module ResNet
    Config = Struct.new(:embedding_size, :hidden_sizes, :depths, :layer_type,
                        :hidden_act, :downsample_in_first_stage, :num_labels,
                        keyword_init: true) do
      def stages
        depths.size
      end

      # The stem halves twice, once in the 7x7 convolution and once in the
      # pooler, and every stage after the first halves again.
      def stride_for(stage)
        stage.zero? && !downsample_in_first_stage ? 1 : 2
      end

      # A stage takes the width of the one before it, and the first takes the
      # stem's.
      def in_size(stage)
        stage.zero? ? embedding_size : hidden_sizes[stage - 1]
      end

      # The shortcut is a 1x1 convolution only where the shape changes.
      def shortcut?(stage)
        in_size(stage) != hidden_sizes[stage] || stride_for(stage) != 1
      end
    end

    # Reader for the safetensors microsoft/resnet-18 publishes. The file is F32
    # already and lib/narray_llm/safetensors.rb reads it directly.
    class Checkpoint
      STEM = 'resnet.embedder.embedder'
      CLASSIFIER = 'classifier.1'
      # BatchNorm at inference is an affine, so the four buffers travel with
      # the convolution they belong to.
      NORM_PARTS = %w[weight bias running_mean running_var].freeze
      BUFFERS = %w[running_mean running_var num_batches_tracked].freeze

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
        @store[name]
      end

      def include?(name)
        @store.include?(name)
      end

      def names
        @store.names
      end

      # The running statistics are buffers, not parameters. Counting them is
      # what makes this disagree with what torch reports.
      def num_parameters
        @store.names.reject { |n| BUFFERS.any? { |b| n.end_with?(".#{b}") } }
              .sum { |n| @store.shape(n).inject(1, :*) }
      end

      # Every convolution in the order the forward pass reaches them, as
      # [prefix, [out, in, kh, kw], stride, padding]. The prefix names the pair:
      # "#{prefix}.convolution.weight" and "#{prefix}.normalization.*".
      def convolutions
        found = [[STEM, [@config.embedding_size, 3, 7, 7], 2, 3]]
        @config.depths.each_with_index do |depth, stage|
          width = @config.hidden_sizes[stage]
          depth.times do |layer|
            base = "resnet.encoder.stages.#{stage}.layers.#{layer}"
            first = layer.zero?
            stride = first ? @config.stride_for(stage) : 1
            input = first ? @config.in_size(stage) : width
            found << ["#{base}.shortcut", [width, input, 1, 1], stride, 0] if first && @config.shortcut?(stage)
            found << ["#{base}.layer.0", [width, input, 3, 3], stride, 1]
            found << ["#{base}.layer.1", [width, width, 3, 3], 1, 1]
          end
        end
        found
      end

      def self.config_beside(path)
        File.join(File.dirname(path), 'config.json')
      end

      def self.read_config(path)
        raise FormatError, "#{path}: not found" unless File.exist?(path)

        raw = JSON.parse(File.read(path))
        Config.new(
          embedding_size: raw.fetch('embedding_size'), hidden_sizes: raw.fetch('hidden_sizes'),
          depths: raw.fetch('depths'), layer_type: raw.fetch('layer_type'),
          hidden_act: raw.fetch('hidden_act'),
          downsample_in_first_stage: raw.fetch('downsample_in_first_stage'),
          num_labels: raw.fetch('id2label').size
        )
      end

      # Every name the model will ask for, with the shape the config says it
      # has. Building it from the config rather than from the file is what
      # makes the check mean something.
      def expected_shapes
        shapes = {}
        convolutions.each do |prefix, weight, _stride, _padding|
          shapes["#{prefix}.convolution.weight"] = weight
          NORM_PARTS.each { |part| shapes["#{prefix}.normalization.#{part}"] = [weight[0]] }
          shapes["#{prefix}.normalization.num_batches_tracked"] = []
        end
        shapes["#{CLASSIFIER}.weight"] = [@config.num_labels, @config.hidden_sizes.last]
        shapes["#{CLASSIFIER}.bias"] = [@config.num_labels]
        shapes
      end

      private

      def validate
        raise FormatError, "#{@path}: layer_type #{@config.layer_type}" unless @config.layer_type == 'basic'

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
