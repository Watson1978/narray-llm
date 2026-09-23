# frozen_string_literal: true

module NArrayLLM
  module Whisper
    # Encoder, decoder and the classifier, plus the parts of generation that
    # are not plain greedy: a forced prompt, and two sets of suppressed ids.
    class Model
      attr_reader :config, :encoder, :decoder

      def self.load(path, spelling: :shift, config_path: nil)
        checkpoint = Checkpoint.load(path, config_path: config_path)
        begin
          new(checkpoint, spelling: spelling)
        ensure
          checkpoint.close
        end
      end

      def initialize(checkpoint, spelling: :shift)
        @config = checkpoint.config
        @encoder = Encoder.new(checkpoint, spelling: spelling)
        @decoder = Decoder.new(checkpoint)
        # No proj_out in the file: the classifier is the embedding read the
        # other way. scale_embedding is false, so nothing is scaled.
        @classifier = Ops.contiguous(checkpoint[Checkpoint::EMBED].transpose)
      end

      def encode(mel, prof: Profiler::NULL)
        @encoder.forward(mel, prof: prof)
      end

      def new_cache(encoder_states)
        @decoder.new_cache(encoder_states)
      end

      # Answers [1, vocab_size].
      def decode(token_id, cache:, prof: Profiler::NULL)
        x = @decoder.decode(token_id, cache: cache, prof: prof)
        prof.section(:unembed) { x.dot(@classifier) }
      end

      # prompt: the forced ids, starting with decoder_start_token_id. Whisper
      # builds [<|startoftranscript|>, <|language|>, <|task|>, <|notimestamps|>]
      # and generation continues from there.
      #
      # suppress applies at every step; begin_suppress only at the first
      # generated one, which is what SuppressTokensAtBeginLogitsProcessor does.
      def generate(mel, prompt:, max_new_tokens:, suppress: nil, begin_suppress: nil,
                   prof: Profiler::NULL)
        states = encode(mel, prof: prof)
        cache = new_cache(states)
        tokens = Array(prompt).flatten.map(&:to_i)
        raise Error, 'prompt must not be empty' if tokens.empty?

        suppress = (suppress || @config.suppress_tokens).map(&:to_i)
        begin_suppress = (begin_suppress || @config.begin_suppress_tokens).map(&:to_i)
        logits = nil
        tokens.each { |id| logits = decode(id, cache: cache, prof: prof) }

        generated = []
        max_new_tokens.times do |step|
          hidden = step.zero? ? suppress + begin_suppress : suppress
          token = pick(logits, hidden)
          generated << token
          break if token == @config.eos_token_id
          break if generated.size == max_new_tokens

          logits = decode(token, cache: cache, prof: prof)
        end
        generated
      end

      private

      # The suppressed ids are pushed to -Infinity before the argmax, which is
      # what the reference's logits processors do. The list is host side and
      # fixed, so a Ruby Array subscript sets them without synchronizing.
      def pick(logits, hidden)
        row = logits.reshape!(@config.vocab_size)
        row[hidden] = XF.new(hidden.size).fill(-Float::INFINITY) unless hidden.empty?
        NArrayLLM.scalar(row.max_index).to_i
      end
    end
  end
end
