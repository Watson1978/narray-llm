# frozen_string_literal: true

module NArrayLLM
  # Reader for the safetensors layout: an 8 byte little-endian header length,
  # a JSON header, then the raw tensors back to back, row major and little
  # endian with no gaps and no overlaps.
  #
  # ankane/safetensors-ruby is not used. Its deserialize answers every tensor
  # at once as Ruby Strings, so the whole file lands on the host before any of
  # it reaches an NArray, and its mmap path (safe_open) builds Numo or Torch
  # arrays only. Reading one tensor at a time is what this needs, and the
  # format is small enough to read directly (docs/idea.md).
  class Safetensors
    HEADER_LENGTH_BYTES = 8
    # Only what this repository stores. Half precision is deliberately absent:
    # Numo has none, so a file meant for both backends cannot carry it.
    DTYPES = { 'F32' => [4, :sfloat], 'I64' => [8, :int64], 'I32' => [4, :int32] }.freeze

    attr_reader :path, :metadata

    def self.open(path, &block)
      new(path, &block)
    end

    def initialize(path)
      @path = path
      @io = File.open(path, 'rb')
      read_header
      return unless block_given?

      begin
        yield self
      ensure
        close
      end
    end

    def close
      @io.close unless @io.closed?
    end

    def names
      @entries.keys
    end

    def include?(name)
      @entries.key?(name)
    end

    def shape(name)
      entry(name).fetch('shape')
    end

    def dtype(name)
      entry(name).fetch('dtype')
    end

    # Answers the tensor as an XF array. Only the bytes of this one tensor are
    # read, so a 2.4 GB file never lands on the host in one piece.
    def [](name)
      info = entry(name)
      shape = info.fetch('shape')
      width, kind = DTYPES.fetch(info.fetch('dtype')) do
        raise FormatError, "#{@path}: #{name} has unsupported dtype #{info['dtype']}"
      end
      from, to = info.fetch('data_offsets')
      expected = width * shape.inject(1, :*)
      unless to - from == expected
        raise FormatError, "#{@path}: #{name} spans #{to - from} bytes, " \
                           "but #{info['dtype']} #{shape.inspect} needs #{expected}"
      end

      @io.seek(@body + from)
      raw = BinaryIO.read_exactly(@io, expected, @path, name)
      array_class(kind).from_binary(raw, shape.empty? ? [1] : shape)
    end

    private

    def entry(name)
      @entries.fetch(name) { raise FormatError, "#{@path}: no such tensor: #{name}" }
    end

    # The file's own dtype, not the forward pass's. Landing F32 bytes in XF
    # would reinterpret four byte floats as a two byte type under DTYPE=fp16
    # or bf16, which is not a loss of precision but a different number, and
    # from_binary would take the first half of the buffer and answer without
    # complaining. Ops.contiguous casts on the way in.
    def array_class(kind)
      case kind
      when :sfloat then XM::SFloat
      when :int64 then XM::Int64
      when :int32 then XM::Int32
      end
    end

    def read_header
      size = BinaryIO.read_exactly(@io, HEADER_LENGTH_BYTES, @path, 'header length').unpack1('Q<')
      if size <= 0 || size > 100_000_000
        raise FormatError, "#{@path}: header length #{size} is out of range"
      end

      json = BinaryIO.read_exactly(@io, size, @path, 'header')
      @body = HEADER_LENGTH_BYTES + size
      parsed = begin
        JSON.parse(json)
      rescue JSON::ParserError => e
        raise FormatError, "#{@path}: header is not JSON: #{e.message}"
      end
      @metadata = parsed.delete('__metadata__') || {}
      @entries = parsed
      validate_layout
    end

    # The format promises no gaps and no overlaps. Checking it here is cheap
    # and catches a truncated or spliced file before any tensor is read.
    def validate_layout
      spans = @entries.map { |name, info| [info.fetch('data_offsets'), name] }.sort_by { |(from, _), _| from }
      at = 0
      spans.each do |(from, to), name|
        raise FormatError, "#{@path}: #{name} starts at #{from}, expected #{at}" unless from == at

        at = to
      end
      total = File.size(@path) - @body
      raise FormatError, "#{@path}: tensors end at #{at} but the body is #{total} bytes" unless at == total
    end
  end
end
