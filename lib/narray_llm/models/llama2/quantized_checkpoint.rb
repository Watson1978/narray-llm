# frozen_string_literal: true

module NArrayLLM
  module Llama2
    # Reader for llama2.c's version 2 (Q8_0) checkpoint, the one runq.c reads.
    #
    # The 1-D RMSNorm weights stay fp32. Every matrix is int8 with one fp32
    # scale per group of group_size elements, laid out as the whole int8 block
    # followed by the whole scale block (runq.c:173 init_quantized_tensors).
    class QuantizedCheckpoint
      MAGIC = 0x616b3432 # "ak42"
      VERSION = 2
      HEADER_BYTES = 256

      # q holds the int8 values shaped [out, in / group_size, group_size] and
      # scales the fp32 factor for each of those groups, [out, in / group_size].
      Quantized = Struct.new(:q, :scales, :shape)

      FP32_NAMES = %i[rms_att_weight rms_ffn_weight rms_final_weight].freeze
      QUANTIZED_NAMES = %i[q_tokens wq wk wv wo w1 w2 w3 wcls].freeze

      attr_reader :path, :config, :group_size, :params

      def self.load(path)
        new(path)
      end

      def initialize(path)
        @path = path
        File.open(path, 'rb') do |io|
          read_header(io)
          @params = {}
          read_fp32(io)
          read_quantized(io)
          assert_eof(io)
        end
      end

      def [](name)
        @params.fetch(name) { raise Error, "no such tensor: #{name.inspect}" }
      end

      private

      def read_header(io)
        head = BinaryIO.read_exactly(io, HEADER_BYTES, @path, 'header')
        magic, version = head.unpack('L<l<')
        raise Error, "#{@path}: bad magic 0x#{format('%08x', magic)}, expected ak42" if magic != MAGIC
        raise Error, "#{@path}: version #{version}, expected #{VERSION}" if version != VERSION

        dim, hidden_dim, num_layers, num_heads, num_kv_heads, vocab, seq = head[8, 28].unpack('l<7')
        shared = head[36].unpack1('C')
        @group_size = head[37, 4].unpack1('l<')
        @config = Config.new(dim: dim, hidden_dim: hidden_dim, num_layers: num_layers,
                             num_heads: num_heads, num_kv_heads: num_kv_heads,
                             vocab_size: vocab, max_seq_len: seq,
                             shared_classifier: shared == 1)
      end

      def read_fp32(io)
        c = @config
        { rms_att_weight: [c.num_layers, c.dim],
          rms_ffn_weight: [c.num_layers, c.dim],
          rms_final_weight: [c.dim] }.each do |name, shape|
          count = shape.inject(:*)
          raw = BinaryIO.read_exactly(io, 4 * count, @path, name.to_s)
          @params[name] = XM::SFloat.from_binary(raw, shape)
        end
      end

      # Every matrix is stored [out, in]; runq.c's matmul reads it as (d, n).
      def quantized_shapes
        c = @config
        q = c.num_heads * c.head_size
        {
          q_tokens: [[c.vocab_size, c.dim]],
          wq: Array.new(c.num_layers) { [q, c.dim] },
          wk: Array.new(c.num_layers) { [c.kv_dim, c.dim] },
          wv: Array.new(c.num_layers) { [c.kv_dim, c.dim] },
          wo: Array.new(c.num_layers) { [c.dim, q] },
          w1: Array.new(c.num_layers) { [c.hidden_dim, c.dim] },
          w2: Array.new(c.num_layers) { [c.dim, c.hidden_dim] },
          w3: Array.new(c.num_layers) { [c.hidden_dim, c.dim] }
        }
      end

      def read_quantized(io)
        quantized_shapes.each do |name, shapes|
          tensors = shapes.map { |shape| read_one(io, shape, name) }
          @params[name] = name == :q_tokens ? tensors.first : tensors
        end
        # The classifier is the embedding table itself unless it was stored.
        @params[:wcls] = if @config.shared_classifier
                           @params[:q_tokens]
                         else
                           read_one(io, [@config.vocab_size, @config.dim], :wcls)
                         end
      end

      def read_one(io, shape, name)
        out, inner = shape
        if (inner % @group_size).zero?
          groups_per_row = inner / @group_size
        else
          raise Error, "#{name}: row of #{inner} is not a multiple of group size #{@group_size}"
        end

        q = BinaryIO.read_exactly(io, out * inner, @path, "#{name} (int8)")
        s = BinaryIO.read_exactly(io, 4 * out * groups_per_row, @path, "#{name} (scales)")
        Quantized.new(XM::Int8.from_binary(q, [out, groups_per_row, @group_size]),
                      XM::SFloat.from_binary(s, [out, groups_per_row]),
                      shape)
      end

      def assert_eof(io)
        return if io.read(1).nil?

        raise Error, "#{@path}: trailing bytes after the weights"
      end
    end
  end
end
