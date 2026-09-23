# frozen_string_literal: true

module NArrayLLM
  # GPT-2 固有。共有するものは lib/narray_llm/ の直下にある。
  module GPT2
    # Reader for llm.c's gpt2_*_debug_state.bin (reference inputs and outputs).
    # Format and provenance: docs/checkpoint-format-gpt2.md
    class DebugState
      MAGIC = 20_240_327
      VERSION = 2
      HEADER_INTS = 256
      HEADER_BYTES = HEADER_INTS * 4

      attr_reader :path, :header, :config, :batch_size, :seq_len,
                  :inputs, :targets, :logits, :loss, :grads_byte_offset

      def self.load(path, config:)
        new(path, config: config)
      end

      def initialize(path, config:)
        @path = path
        @config = config
        File.open(path, 'rb') do |io|
          @header = BinaryIO.read_header(io, path, MAGIC, VERSION, ints: HEADER_INTS)
          @batch_size = @header[2]
          @seq_len = @header[3]
          read_sections(io)
        end
      end

      def expected_file_size
        bt = @batch_size * @seq_len
        HEADER_BYTES + 4 * bt + 4 * bt + 4 * bt * @config.vocab_size + 4 +
          4 * Checkpoint.num_parameters(@config)
      end

      private

      def read_sections(io)
        bt = @batch_size * @seq_len
        shape = [@batch_size, @seq_len]
        # Token ids stay on the host: they are only ever used to build the one-hot
        # matrix, and an NArray index on the device would force a sync.
        @inputs = HM::Int32.from_binary(BinaryIO.read_exactly(io, 4 * bt, @path, 'x'), shape)
        @targets = HM::Int32.from_binary(BinaryIO.read_exactly(io, 4 * bt, @path, 'y'), shape)

        count = bt * @config.vocab_size
        logits_raw = BinaryIO.read_exactly(io, 4 * count, @path, 'expected_logits')
        @logits = XM::SFloat.from_binary(logits_raw, [@batch_size, @seq_len, @config.vocab_size])

        @loss = BinaryIO.read_exactly(io, 4, @path, 'expected_loss').unpack1('e')

        # expected_grads has the same layout as the parameters. Inference does not
        # need it, so record where it starts and stop; grad reads it one tensor
        # at a time so that 475 MB never lands on the host in one piece.
        @grads_byte_offset = io.pos
        unless File.size(@path) == expected_file_size
          raise FormatError, "#{@path}: size #{File.size(@path)}, expected #{expected_file_size}"
        end
      end

      public

      # The reference gradient for one parameter tensor, in the layout
      # Checkpoint.tensor_shapes gives. test_gpt2.c:151-166 checks these against
      # the backward pass with an absolute tolerance of 2e-2.
      def grad(name)
        shapes = Checkpoint.tensor_shapes(@config)
        shape = shapes.fetch(name) { raise FormatError, "#{@path}: no such tensor: #{name}" }
        offset = @grads_byte_offset
        shapes.each do |other, other_shape|
          break if other == name

          offset += 4 * other_shape.inject(:*)
        end

        File.open(@path, 'rb') do |io|
          io.seek(offset)
          raw = BinaryIO.read_exactly(io, 4 * shape.inject(:*), @path, "grad #{name}")
          XM::SFloat.from_binary(raw, shape)
        end
      end

      def grad_names
        Checkpoint::TENSOR_NAMES
      end

      private
    end
  end
end
