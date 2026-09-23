# frozen_string_literal: true

module NArrayLLM
  module Llama2
    # Reader for the reference activations script/llama2_dump.c writes.
    # llama2.c ships no equivalent of llm.c's debug_state, so this file is
    # generated locally; see script/llama2_dump.rb.
    #
    # Layout: a header, the token fed at each position, then one self-describing
    # record per tensor (name, position, layer, count, floats).
    class DebugState
      MAGIC = 20_260_916
      VERSION = 1
      NAME_BYTES = 24
      RECORD_HEADER_BYTES = NAME_BYTES + 12

      attr_reader :path, :config, :tokens, :steps

      def self.load(path)
        new(path)
      end

      def initialize(path)
        @path = path
        @records = {}
        File.open(path, 'rb') do |io|
          read_header(io)
          read_records(io)
        end
      end

      # Tensors that belong to a block carry a layer; the rest pass -1.
      def [](name, pos, layer = -1)
        @records.fetch([name.to_s, pos, layer]) do
          raise Error, "no such record: #{name} pos=#{pos} layer=#{layer}"
        end
      end

      def key?(name, pos, layer = -1)
        @records.key?([name.to_s, pos, layer])
      end

      def names
        @records.keys.map(&:first).uniq
      end

      def logits(pos)
        self['logits', pos]
      end

      private

      def read_header(io)
        ints = BinaryIO.read_exactly(io, 40, @path, 'header').unpack('l<10')
        raise FormatError, "#{@path}: bad magic #{ints[0]}, expected #{MAGIC}" unless ints[0] == MAGIC
        raise FormatError, "#{@path}: unsupported version #{ints[1]}" unless ints[1] == VERSION

        @config = Config.new(dim: ints[2], hidden_dim: ints[3], num_layers: ints[4],
                             num_heads: ints[5], num_kv_heads: ints[6],
                             vocab_size: ints[7], max_seq_len: ints[8],
                             shared_classifier: true)
        @steps = ints[9]
        raw = BinaryIO.read_exactly(io, 4 * @steps, @path, 'tokens')
        @tokens = raw.unpack('l<*')
      end

      def read_records(io)
        until io.eof?
          head = BinaryIO.read_exactly(io, RECORD_HEADER_BYTES, @path, 'record header')
          name = head[0, NAME_BYTES].unpack1('Z*')
          pos, layer, count = head[NAME_BYTES..].unpack('l<3')
          raw = BinaryIO.read_exactly(io, 4 * count, @path, "#{name} at pos #{pos}")
          @records[[name, pos, layer]] = XM::SFloat.from_binary(raw, [count])
        end
      end
    end
  end
end
