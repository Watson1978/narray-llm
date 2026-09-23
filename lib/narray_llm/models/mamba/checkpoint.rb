# frozen_string_literal: true

module NArrayLLM
  # Mamba 固有。共有するものは lib/narray_llm/ の直下にある。
  module Mamba
    Config = Struct.new(:num_layers, :vocab_size, :dim, :d_inner, :dt_rank,
                        :d_state, :d_conv, :shared_classifier,
                        keyword_init: true) do
      # mamba.c:187 rounds the vocabulary up to a multiple of 8, and the
      # embedding and classifier are stored at that size, not at vocab_size.
      def rounded_vocab_size
        remainder = vocab_size % 8
        remainder.zero? ? vocab_size : vocab_size + (8 - remainder)
      end

      # x_proj answers dt, B and C in one row (mamba.c:425).
      def x_proj_out
        dt_rank + 2 * d_state
      end

      # Mamba has no positional limit: the two recurrences are the same size
      # whatever the length, and mamba.c allocates nothing per position. The
      # field exists because the shared Generator checks it.
      def max_seq_len
        Float::INFINITY
      end
    end

    # Reader for kroggen/mamba.c's version 1 checkpoint, the one its export.py
    # writes. Format and provenance: docs/checkpoint-format-mamba.md
    class Checkpoint
      MAGIC = 0x4d616d62 # "Mamb"
      VERSION = 1
      HEADER_BYTES = 256
      HEADER_INTS = 8

      TENSOR_NAMES = %i[
        embedding in_proj conv1d_weight conv1d_bias x_proj dt_proj_weight
        dt_proj_bias a d out_proj norm final_norm lm_head
      ].freeze

      attr_reader :path, :header, :config, :shapes, :params, :byte_offsets

      def self.load(path)
        new(path)
      end

      # Written out by export.py in this order, each tensor holding every layer.
      # A is stored as -exp(A_log), so it is read as it stands.
      def self.tensor_shapes(config)
        l = config.num_layers
        d = config.dim
        di = config.d_inner
        {
          embedding: [config.rounded_vocab_size, d],
          in_proj: [l, 2 * di, d],
          conv1d_weight: [l, di, config.d_conv],
          conv1d_bias: [l, di],
          x_proj: [l, config.x_proj_out, di],
          dt_proj_weight: [l, di, config.dt_rank],
          dt_proj_bias: [l, di],
          a: [l, di, config.d_state],
          d: [l, di],
          out_proj: [l, d, di],
          norm: [l, d],
          final_norm: [d],
          lm_head: [config.rounded_vocab_size, d]
        }
      end

      def self.num_parameters(config)
        shapes = tensor_shapes(config)
        shapes.delete(:lm_head) if config.shared_classifier
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

      def num_parameters
        Checkpoint.num_parameters(@config)
      end

      def expected_file_size
        HEADER_BYTES + 4 * num_parameters
      end

      def self.read_header(io, path)
        head = BinaryIO.read_exactly(io, HEADER_BYTES, path, 'header')
        magic, version = head.unpack('L<l<')
        unless magic == MAGIC
          raise FormatError, format('%<path>s: bad magic 0x%<magic>08x, expected Mamb',
                                    path: path, magic: magic)
        end
        raise FormatError, "#{path}: version #{version}, expected #{VERSION}" unless version == VERSION

        head[8, 4 * HEADER_INTS].unpack('l<*')
      end

      def self.build_config(header)
        num_layers, vocab_size, dim, d_inner, dt_rank, d_state, d_conv, shared = header
        config = Config.new(num_layers: num_layers, vocab_size: vocab_size, dim: dim,
                            d_inner: d_inner, dt_rank: dt_rank, d_state: d_state,
                            d_conv: d_conv, shared_classifier: shared == 1)
        validate(config)
        config
      end

      def self.validate(config)
        %i[num_layers vocab_size dim d_inner dt_rank d_state d_conv].each do |field|
          value = config[field]
          raise FormatError, "#{field} must be positive, got #{value}" unless value.positive?
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
          if name == :lm_head && @config.shared_classifier
            @byte_offsets[name] = @byte_offsets[:embedding]
            acc[name] = acc[:embedding]
            next
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
