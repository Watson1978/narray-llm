# frozen_string_literal: true

module NArrayLLM
  module Switch
    # Encoder, decoder and the classifier. The published checkpoint ties the
    # embedding to the classifier, and transformers scales the decoder output
    # by d_model ** -0.5 before the classifier when they are tied
    # (SwitchTransformersForConditionalGeneration.forward). That scale is not
    # optional here: the weights are tied.
    class Model
      attr_reader :config, :encoder, :decoder

      def self.load(path, router: :dispatch, expert_capacity: nil, config_path: nil)
        checkpoint = Checkpoint.load(path, config_path: config_path)
        begin
          new(checkpoint, router: router, expert_capacity: expert_capacity)
        ensure
          checkpoint.close
        end
      end

      def initialize(checkpoint, router: :dispatch, expert_capacity: nil)
        @config = checkpoint.config
        @encoder = Encoder.new(checkpoint, router: router, expert_capacity: expert_capacity)
        @decoder = Decoder.new(checkpoint, router: router, expert_capacity: expert_capacity)
        @lm_head_t = Ops.contiguous(checkpoint[Checkpoint::SHARED].transpose)
        @scale = @config.d_model**-0.5
      end

      def encode(ids, prof: Profiler::NULL, trace: nil)
        @encoder.forward(ids, prof: prof, trace: trace)
      end

      def new_cache(encoder_states, max_seq_len:)
        @decoder.new_cache(encoder_states, max_seq_len: max_seq_len)
      end

      # Answers [1, vocab_size].
      def decode(token_id, cache:, prof: Profiler::NULL, trace: nil)
        x = @decoder.decode(token_id, cache: cache, prof: prof, trace: trace)
        prof.section(:unembed) { (x * @scale).dot(@lm_head_t) }
      end

      # Greedy, the way the reference's generate runs with no sampling. The
      # decoder starts on decoder_start_token_id, which is the pad id for this
      # family, and the answer includes it so the two sides line up.
      def generate(ids, max_new_tokens:, stop_at_eos: true, prof: Profiler::NULL)
        states = encode(ids, prof: prof)
        cache = new_cache(states, max_seq_len: max_new_tokens + 1)
        tokens = [@config.decoder_start_token_id]
        max_new_tokens.times do
          logits = decode(tokens.last, cache: cache, prof: prof)
          token = NArrayLLM.scalar(logits[0, true].max_index).to_i
          tokens << token
          break if stop_at_eos && token == @config.eos_token_id
        end
        tokens
      end
    end
  end
end
