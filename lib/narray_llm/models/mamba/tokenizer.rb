# frozen_string_literal: true

module NArrayLLM
  module Mamba
    # Decoder for the table mamba.c's tokenizer.py writes: GPT-NeoX, already
    # decoded to UTF-8 bytes, with no scores. Decoding only, like the other two.
    # Format and provenance: docs/tokenizer-format-mamba.md
    class Tokenizer
      MAGIC = 0x4d62546b # "MbTk"
      VERSION = 1

      # mamba.c:506 makes both the same token, <|endoftext|>.
      BOS_TOKEN = 0
      EOS_TOKEN = 0

      # Kept for the one rule that needs it, though GPT-NeoX pieces are already
      # bytes and so never spell a byte out the way SentencePiece does.
      BYTE_PIECE = /\A<0x(\h{2})>/

      attr_reader :path, :vocab_size, :max_token_length

      # The file carries its own count, unlike llama2.c's.
      def self.load(path)
        new(path)
      end

      def initialize(path)
        @path = path
        File.open(path, 'rb') { |io| read_table(io) }
      end

      def [](id)
        unless id.is_a?(Integer) && id >= 0 && id < @vocab_size
          raise Error, "invalid token id #{id.inspect} (vocab_size=#{@vocab_size})"
        end

        @pieces[id]
      end

      def eot_token
        EOS_TOKEN
      end

      # mamba.c:570 decode.
      def decode_piece(prev_token, token)
        piece = self[token]
        piece = piece.byteslice(1..) if prev_token == EOS_TOKEN && piece.start_with?(' ')
        if (m = BYTE_PIECE.match(piece))
          return m[1].to_i(16).chr(Encoding::ASCII_8BIT)
        end

        piece
      end

      # mamba.c:583 safe_printf.
      def printable?(piece)
        return false if piece.empty?
        return true if piece.bytesize > 1

        byte = piece.getbyte(0)
        (0x20..0x7E).cover?(byte) || [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20].include?(byte)
      end

      def render(tokens)
        out = +''
        tokens.each_cons(2) do |prev, token|
          piece = decode_piece(prev, token)
          out << piece if printable?(piece)
        end
        out.force_encoding(Encoding::UTF_8)
      end

      def expected_file_size
        16 + @vocab_size * 4 + @pieces.sum(&:bytesize)
      end

      private

      def read_table(io)
        head = BinaryIO.read_exactly(io, 16, @path, 'header')
        magic, version, @vocab_size, @max_token_length = head.unpack('L<l<l<l<')
        unless magic == MAGIC
          raise FormatError, format('%<path>s: bad magic 0x%<magic>08x, expected MbTk',
                                    path: @path, magic: magic)
        end
        raise FormatError, "#{@path}: version #{version}, expected #{VERSION}" unless version == VERSION
        raise FormatError, "#{@path}: vocab_size #{@vocab_size}" unless @vocab_size.positive?

        @pieces = Array.new(@vocab_size)
        @vocab_size.times do |id|
          len = BinaryIO.read_exactly(io, 4, @path, "token #{id} length").unpack1('l<')
          raise FormatError, "#{@path}: token #{id} has length #{len}" if len.negative?

          @pieces[id] = BinaryIO.read_exactly(io, len, @path, "token #{id}")
        end
        return if io.eof?

        raise FormatError, "#{@path}: #{File.size(@path) - io.pos} trailing bytes after #{@vocab_size} tokens"
      end
    end
  end
end
