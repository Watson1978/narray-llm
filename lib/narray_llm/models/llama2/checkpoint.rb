# frozen_string_literal: true

module NArrayLLM
  # Llama 2 固有。共有するものは lib/narray_llm/ の直下にある。
  module Llama2
    Config = Struct.new(:dim, :hidden_dim, :num_layers, :num_heads, :num_kv_heads,
                        :vocab_size, :max_seq_len, :shared_classifier,
                        keyword_init: true) do
      def head_size
        dim / num_heads
      end

      def kv_dim
        dim * num_kv_heads / num_heads
      end

      # How many query heads share one key/value head. 1 is plain MHA.
      def kv_mul
        num_heads / num_kv_heads
      end

      def grouped_query?
        num_kv_heads != num_heads
      end
    end

    # Reader for llama2.c's legacy (v0) checkpoint format.
    # Format and provenance: docs/checkpoint-format-llama2.md
    class Checkpoint
      HEADER_INTS = 7
      HEADER_BYTES = HEADER_INTS * 4

      TENSOR_NAMES = %i[
        token_embedding_table rms_att_weight wq wk wv wo
        rms_ffn_weight w1 w2 w3 rms_final_weight wcls
      ].freeze

      attr_reader :path, :header, :config, :shapes, :params, :byte_offsets

      def self.load(path)
        new(path)
      end

      # Row-major, out_features first: these are torch nn.Linear weights written
      # straight out by export.py, and run.c's matmul reads w as (d, n).
      def self.tensor_shapes(config)
        d = config.dim
        l = config.num_layers
        q = config.num_heads * config.head_size
        kv = config.kv_dim
        h = config.hidden_dim
        {
          token_embedding_table: [config.vocab_size, d],
          rms_att_weight: [l, d],
          wq: [l, q, d],
          wk: [l, kv, d],
          wv: [l, kv, d],
          wo: [l, d, q],
          rms_ffn_weight: [l, d],
          w1: [l, h, d],
          w2: [l, d, h],
          w3: [l, h, d],
          rms_final_weight: [d],
          wcls: [config.vocab_size, d]
        }
      end

      # run.c skips what used to be freq_cis_real and freq_cis_imag, each
      # seq_len * head_size / 2 floats, between rms_final_weight and wcls.
      def self.freq_cis_elements(config)
        2 * (config.max_seq_len * config.head_size / 2)
      end

      def self.num_parameters(config)
        shapes = tensor_shapes(config)
        shapes.delete(:wcls) if config.shared_classifier
        shapes.sum { |_, shape| shape.inject(:*) }
      end

      def initialize(path)
        @path = path
        File.open(path, 'rb') do |io|
          @header = Checkpoint.read_header(io, path)
          @config = Checkpoint.build_config(@header)
          @shapes = Checkpoint.tensor_shapes(@config)
          @byte_offsets = {}
          @params = read_params(io)
          Checkpoint.assert_eof(io, path)
        end
      end

      def [](name)
        @params.fetch(name) { raise Error, "no such tensor: #{name.inspect}" }
      end

      def byte_offset(name)
        @byte_offsets.fetch(name) { raise Error, "no such tensor: #{name.inspect}" }
      end

      # The classifier is not stored when it is shared, so counting its elements
      # would not match the file.
      def num_parameters
        Checkpoint.num_parameters(@config)
      end

      def expected_file_size
        HEADER_BYTES + 4 * (num_parameters + Checkpoint.freq_cis_elements(@config))
      end

      def self.read_header(io, path)
        BinaryIO.read_exactly(io, HEADER_BYTES, path, 'header').unpack('l<*')
      end

      # A negative vocab_size is llama2.c's flag for a classifier that is not
      # shared with the embedding (run.c:150 "bit yikes").
      def self.build_config(header)
        dim, hidden_dim, num_layers, num_heads, num_kv_heads, vocab_size, max_seq_len = header
        config = Config.new(dim: dim, hidden_dim: hidden_dim, num_layers: num_layers,
                            num_heads: num_heads, num_kv_heads: num_kv_heads,
                            vocab_size: vocab_size.abs, max_seq_len: max_seq_len,
                            shared_classifier: vocab_size.positive?)
        validate(config)
        config
      end

      def self.validate(config)
        %i[dim hidden_dim num_layers num_heads num_kv_heads vocab_size max_seq_len].each do |field|
          value = config[field]
          raise FormatError, "#{field} must be positive, got #{value}" unless value.positive?
        end
        unless (config.dim % config.num_heads).zero?
          raise FormatError, "dim #{config.dim} is not a multiple of num_heads #{config.num_heads}"
        end
        unless (config.num_heads % config.num_kv_heads).zero?
          raise FormatError,
                "num_heads #{config.num_heads} is not a multiple of num_kv_heads #{config.num_kv_heads}"
        end
      end

      def self.assert_eof(io, path)
        extra = io.read(1)
        return if extra.nil?

        raise FormatError, "#{path}: #{File.size(path) - io.pos + 1} trailing bytes after the parameters"
      end

      private

      def read_params(io)
        acc = {}
        TENSOR_NAMES.each do |name|
          if name == :wcls
            # Read rather than seek: seeking past the end succeeds, which would
            # let a file truncated inside freq_cis look complete.
            BinaryIO.read_exactly(io, 4 * Checkpoint.freq_cis_elements(@config),
                                  @path, 'skipped freq_cis')
            if @config.shared_classifier
              @byte_offsets[name] = @byte_offsets[:token_embedding_table]
              acc[name] = acc[:token_embedding_table]
              next
            end
          end
          acc[name] = read_tensor(io, name)
        end
        acc
      end

      def read_tensor(io, name)
        shape = @shapes.fetch(name)
        @byte_offsets[name] = io.pos
        raw = BinaryIO.read_exactly(io, 4 * shape.inject(:*), @path, name.to_s)
        XM::SFloat.from_binary(raw, shape)
      end
    end
  end
end
