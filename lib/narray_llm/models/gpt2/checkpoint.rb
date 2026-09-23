# frozen_string_literal: true

module NArrayLLM
  # GPT-2 固有。共有するものは lib/narray_llm/ の直下にある。
  module GPT2
    Config = Struct.new(:max_seq_len, :vocab_size, :num_layers, :num_heads,
                        :channels, :padded_vocab_size, keyword_init: true)

    # Reader for llm.c's gpt2_*.bin weight files.
    # Format and provenance: docs/checkpoint-format-gpt2.md
    class Checkpoint
      MAGIC = 20_240_326
      VERSION = 3
      HEADER_INTS = 256
      HEADER_BYTES = HEADER_INTS * 4

      TENSOR_NAMES = %i[
        wte wpe ln1w ln1b qkvw qkvb attprojw attprojb
        ln2w ln2b fcw fcb fcprojw fcprojb lnfw lnfb
      ].freeze

      attr_reader :path, :header, :config, :shapes, :params, :byte_offsets

      def self.load(path)
        new(path)
      end

      def self.tensor_shapes(config)
        c = config.channels
        l = config.num_layers
        {
          wte: [config.padded_vocab_size, c],
          wpe: [config.max_seq_len, c],
          ln1w: [l, c],
          ln1b: [l, c],
          qkvw: [l, 3 * c, c],
          qkvb: [l, 3 * c],
          attprojw: [l, c, c],
          attprojb: [l, c],
          ln2w: [l, c],
          ln2b: [l, c],
          fcw: [l, 4 * c, c],
          fcb: [l, 4 * c],
          fcprojw: [l, c, 4 * c],
          fcprojb: [l, c],
          lnfw: [c],
          lnfb: [c]
        }
      end

      def self.num_parameters(config)
        tensor_shapes(config).sum { |_, shape| shape.inject(:*) }
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

      def num_parameters
        @shapes.sum { |_, shape| shape.inject(:*) }
      end

      def expected_file_size
        HEADER_BYTES + 4 * num_parameters
      end

      def self.read_header(io, path)
        BinaryIO.read_header(io, path, MAGIC, VERSION, ints: HEADER_INTS)
      end

      def self.build_config(header)
        Config.new(max_seq_len: header[2], vocab_size: header[3], num_layers: header[4],
                   num_heads: header[5], channels: header[6], padded_vocab_size: header[7])
      end

      def self.assert_eof(io, path)
        extra = io.read(1)
        return if extra.nil?

        raise FormatError, "#{path}: #{File.size(path) - io.pos + 1} trailing bytes after the parameters"
      end

      private

      def read_params(io)
        @shapes.each_with_object({}) do |(name, shape), acc|
          @byte_offsets[name] = io.pos
          count = shape.inject(:*)
          raw = BinaryIO.read_exactly(io, 4 * count, @path, name.to_s)
          acc[name] = XM::SFloat.from_binary(raw, shape)
        end
      end
    end
  end
end
