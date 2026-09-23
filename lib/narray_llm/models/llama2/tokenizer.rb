# frozen_string_literal: true

module NArrayLLM
  module Llama2
    # Decoder for llama2.c's tokenizer.bin / tok512.bin. Decoding only; the
    # SentencePiece BPE encoder is out of scope (docs/plans/PLAN-llama2.md).
    # Format and provenance: docs/tokenizer-format-llama2.md
    class Tokenizer
      # run.c:763 breaks the generation loop on BOS, not on EOS: in llama2.c the
      # BOS token is what delimits sequences. EOS is here for completeness.
      BOS_TOKEN = 1
      EOS_TOKEN = 2

      # Tokens standing for a raw byte are spelled out, e.g. "<0x0A>" (run.c:425).
      BYTE_PIECE = /\A<0x(\h{1,2})/

      attr_reader :path, :vocab_size, :max_token_length, :scores

      # The file carries no vocabulary size; run.c:387 takes it from the model
      # config, so the caller has to pass it too.
      def self.load(path, vocab_size:)
        new(path, vocab_size: vocab_size)
      end

      def initialize(path, vocab_size:)
        @path = path
        @vocab_size = vocab_size
        File.open(path, 'rb') { |io| read_table(io) }
      end

      # Raw bytes for one token id, as ASCII-8BIT. Pieces can contain any byte,
      # so they are kept by length rather than NUL-terminated the way run.c does.
      def [](id)
        unless id.is_a?(Integer) && id >= 0 && id < @vocab_size
          raise Error, "invalid token id #{id.inspect} (vocab_size=#{@vocab_size})"
        end

        @pieces[id]
      end

      def score(id)
        @scores[id]
      end

      # Generator stops on this. Named for the shared interface; in llama2.c the
      # delimiter really is BOS (run.c:763).
      def eot_token
        BOS_TOKEN
      end

      # run.c:418 decode. prev_token only matters for the one rule that depends
      # on it: SentencePiece drops the space a piece carries when it follows BOS.
      def decode_piece(prev_token, token)
        piece = self[token]
        piece = piece.byteslice(1..) if prev_token == BOS_TOKEN && piece.start_with?(' ')
        if (m = BYTE_PIECE.match(piece))
          return m[1].to_i(16).chr(Encoding::ASCII_8BIT)
        end

        piece
      end

      # run.c:431 safe_printf. A piece that is a single byte is dropped unless it
      # is printable or whitespace, because raw-byte tokens can be control codes.
      def printable?(piece)
        return false if piece.empty?
        return true if piece.bytesize > 1

        byte = piece.getbyte(0)
        (0x20..0x7E).cover?(byte) || [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20].include?(byte)
      end

      # The text run.c would have printed for this sequence: every token after
      # the first, decoded against its predecessor and filtered (run.c:765-767).
      def render(tokens)
        out = +''
        tokens.each_cons(2) do |prev, token|
          piece = decode_piece(prev, token)
          out << piece if printable?(piece)
        end
        out.force_encoding(Encoding::UTF_8)
      end

      def expected_file_size
        4 + @vocab_size * 8 + @pieces.sum(&:bytesize)
      end

      private

      def read_table(io)
        @max_token_length = BinaryIO.read_exactly(io, 4, @path, 'max_token_length').unpack1('l<')
        @pieces = Array.new(@vocab_size)
        @scores = Array.new(@vocab_size)
        @vocab_size.times do |id|
          head = BinaryIO.read_exactly(io, 8, @path, "entry #{id}")
          @scores[id] = head.unpack1('e')
          len = head[4, 4].unpack1('l<')
          raise FormatError, "#{@path}: token #{id} has length #{len}" if len.negative?

          @pieces[id] = BinaryIO.read_exactly(io, len, @path, "token #{id}")
        end
        return if io.eof?

        raise FormatError, "#{@path}: #{File.size(@path) - io.pos} trailing bytes after #{@vocab_size} tokens"
      end
    end
  end
end
