# frozen_string_literal: true

module NArrayLLM
  # GPT-2 固有。共有するものは lib/narray_llm/ の直下にある。
  module GPT2
    # Decoder for llm.c's gpt2_tokenizer.bin. Decoding only; GPT-2's BPE encoder
    # is out of scope (docs/plans/PLAN-gpt2.md). Format and provenance: docs/tokenizer-format-gpt2.md
    class Tokenizer
      MAGIC = 20_240_328
      SUPPORTED_VERSIONS = [1, 2].freeze
      HEADER_INTS = 256
      HEADER_BYTES = HEADER_INTS * 4

      # tokenizer.h:59-63: version 1 predates the EOT field, so llm.c falls back to
      # 50256 and asserts the vocabulary is the standard GPT-2 one. Version 2 and
      # later carry the id in header[3], which is where the shipped file has it.
      VERSION1_EOT_TOKEN = 50_256
      VERSION1_VOCAB_SIZE = 50_257

      attr_reader :path, :header, :version, :vocab_size, :eot_token

      def self.load(path)
        new(path)
      end

      def initialize(path)
        @path = path
        File.open(path, 'rb') do |io|
          read_header(io)
          read_token_table(io)
        end
      end

      # Raw bytes for one token id, as ASCII-8BIT. Token bytes can contain 0x00,
      # so they are kept by length rather than NUL-terminated the way llm.c does.
      def [](id)
        unless id.is_a?(Integer) && id >= 0 && id < @vocab_size
          raise Error, "invalid token id #{id.inspect} (vocab_size=#{@vocab_size})"
        end

        @tokens[id]
      end

      def decode_bytes(ids)
        normalize(ids).map { |id| self[id] }.join
      end

      # GPT-2 is byte-level BPE, so a single character can be split across tokens
      # and any prefix of the byte stream may be invalid UTF-8. Concatenate first,
      # then reinterpret, then replace whatever is still malformed.
      def decode(ids)
        decode_bytes(ids).force_encoding(Encoding::UTF_8).scrub
      end

      def eot?(id)
        id == @eot_token
      end

      def size
        @vocab_size
      end

      private

      def read_header(io)
        raw = BinaryIO.read_exactly(io, HEADER_BYTES, @path, 'header')
        # uint32 here, unlike the int32 headers of the checkpoint files.
        @header = raw.unpack('L<*')
        raise FormatError, "#{@path}: bad magic #{@header[0]}, expected #{MAGIC}" unless @header[0] == MAGIC

        @version = @header[1]
        unless SUPPORTED_VERSIONS.include?(@version)
          raise FormatError, "#{@path}: unsupported tokenizer version #{@version}"
        end

        @vocab_size = @header[2]
        @eot_token = read_eot_token
      end

      def read_eot_token
        return @header[3] if @version >= 2

        unless @vocab_size == VERSION1_VOCAB_SIZE
          raise FormatError, "#{@path}: version 1 without the standard GPT-2 vocabulary " \
                             "(vocab_size=#{@vocab_size})"
        end

        VERSION1_EOT_TOKEN
      end

      def read_token_table(io)
        blob = io.read
        offset = 0
        @tokens = Array.new(@vocab_size) do |id|
          length = blob.getbyte(offset)
          raise FormatError, "#{@path}: truncated token table at id #{id}" if length.nil?
          raise FormatError, "#{@path}: zero-length token at id #{id}" if length.zero?

          offset += 1
          bytes = blob.byteslice(offset, length)
          if bytes.nil? || bytes.bytesize != length
            raise FormatError, "#{@path}: truncated token #{id} (wanted #{length} bytes)"
          end

          offset += length
          bytes
        end

        return if offset == blob.bytesize

        raise FormatError, "#{@path}: #{blob.bytesize - offset} trailing bytes after #{@vocab_size} tokens"
      end

      def normalize(ids)
        return [ids] if ids.is_a?(Integer)

        ids.to_a.flatten
      end
    end
  end
end
