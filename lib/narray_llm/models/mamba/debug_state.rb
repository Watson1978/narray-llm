# frozen_string_literal: true

module NArrayLLM
  module Mamba
    # Reader for the reference activations script/mamba_dump.c writes.
    # mamba.c ships no debug dump and its CLI cannot be pointed at a small
    # checkpoint, so this file is generated locally.
    #
    # Layout: a header, the token fed at each position, then one
    # self-describing record per tensor (name, position, layer, count, floats).
    class DebugState
      MAGIC = 20_260_919
      VERSION = 1
      NAME_BYTES = 24
      RECORD_HEADER_BYTES = NAME_BYTES + 12
      HEADER_INTS = 11

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
        ints = BinaryIO.read_exactly(io, 4 * HEADER_INTS, @path, 'header').unpack("l<#{HEADER_INTS}")
        raise FormatError, "#{@path}: bad magic #{ints[0]}, expected #{MAGIC}" unless ints[0] == MAGIC
        raise FormatError, "#{@path}: unsupported version #{ints[1]}" unless ints[1] == VERSION

        @config = Config.new(num_layers: ints[2], vocab_size: ints[3], dim: ints[4],
                             d_inner: ints[5], dt_rank: ints[6], d_state: ints[7],
                             d_conv: ints[8], shared_classifier: ints[9] == 1)
        @steps = ints[10]
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
