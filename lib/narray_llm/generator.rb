# frozen_string_literal: true

module NArrayLLM
  # Greedy (argmax) decoding. No temperature, no top-k, no sampling: docs/plans/PLAN-gpt2.md
  # keeps those out until after stage 3 so that two backends must produce an
  # identical token sequence.
  #
  # No KV cache yet either -- every step recomputes the whole sequence. That is
  # stage 3's job, and the tokens/sec measured here is its "before" number.
  class Generator
    DEFAULT_MAX_NEW_TOKENS = 64

    attr_reader :model, :tokenizer

    def initialize(model, tokenizer: nil)
      @model = model
      @tokenizer = tokenizer
      @sampler = nil
    end

    def eot_token
      @tokenizer&.eot_token
    end

    # Returns the whole sequence, prompt included. With cache: true the prompt
    # runs once and each step afterwards processes a single token; with
    # cache: false every step recomputes the whole sequence (stage 2 behaviour,
    # kept so the two can be compared).
    # sampler: nil keeps the greedy path exactly as it is, down to the kernels
    # it launches. Anything else is a Sampler.
    def generate(prompt, max_new_tokens: DEFAULT_MAX_NEW_TOKENS, stop_at_eot: true,
                 prof: Profiler::NULL, cache: true, sampler: nil, &block)
      @sampler = sampler
      tokens = normalize(prompt)
      raise Error, 'prompt must contain at least one token' if tokens.empty?

      limit = @model.config.max_seq_len
      if tokens.size + max_new_tokens > limit
        raise Error, "#{tokens.size} prompt tokens + #{max_new_tokens} generated tokens " \
                     "exceeds max_seq_len #{limit}"
      end

      if cache
        unless @model.respond_to?(:new_cache)
          raise Error, "#{@model.class} has no KV cache; pass cache: false"
        end

        generate_cached(tokens, max_new_tokens, stop_at_eot, prof, &block)
      else
        generate_recomputing(tokens, max_new_tokens, stop_at_eot, prof, &block)
      end
    end

    def next_token(tokens, prof: Profiler::NULL)
      argmax(@model.forward([tokens], last_only: true, prof: prof))
    end

    # B sequences at once, prompts included. Every prompt has to be the same
    # length: one position is shared by the whole batch (docs/plans/PLAN-batch.md), so a
    # ragged batch would put some sequences at the wrong position.
    #
    # A sequence that reaches EOT is not taken out of the batch. The whole batch
    # keeps stepping and each sequence is cut afterwards, which costs work on
    # the finished rows and keeps the shapes fixed. Stepping stops early only
    # when every sequence has finished.
    def generate_batch(prompts, max_new_tokens: DEFAULT_MAX_NEW_TOKENS, stop_at_eot: true,
                       prof: Profiler::NULL)
      rows = prompts.map { |prompt| normalize(prompt) }
      raise Error, 'a batch needs at least one prompt' if rows.empty?
      raise Error, 'every prompt must contain at least one token' if rows.any?(&:empty?)

      lengths = rows.map(&:size).uniq
      unless lengths.size == 1
        raise Error, "every prompt must be the same length, got #{rows.map(&:size).inspect}"
      end

      length = lengths.first
      limit = @model.config.max_seq_len
      if length + max_new_tokens > limit
        raise Error, "#{length} prompt tokens + #{max_new_tokens} generated tokens " \
                     "exceeds max_seq_len #{limit}"
      end
      unless @model.respond_to?(:new_cache)
        raise Error, "#{@model.class} has no KV cache; batched generation needs one"
      end

      step_batch(rows, length, max_new_tokens, stop_at_eot, prof)
    end

    private

    def step_batch(rows, length, max_new_tokens, stop_at_eot, prof)
      cache = @model.new_cache(batch_size: rows.size)
      logits = @model.prefill(rows.map(&:dup), cache: cache, prof: prof)
      produced = Array.new(rows.size) { [] }

      max_new_tokens.times do |step|
        ids = argmax_rows(logits, rows.size)
        ids.each_with_index { |id, i| produced[i] << id }
        break if step == max_new_tokens - 1
        break if stop_at_eot && !eot_token.nil? && produced.all? { |seq| seq.include?(eot_token) }

        logits = @model.decode(ids, length + produced.first.size - 1, cache: cache, prof: prof)
      end

      rows.each_with_index.map { |prompt, i| prompt + cut_at_eot(produced[i], stop_at_eot) }
    end

    # One readback per step for the whole batch, not one per sequence. Cumo's
    # max_index answers a flattened running number, so the row offsets come off
    # on the device before the single read (AGENTS.md).
    def argmax_rows(logits, batch_size)
      rows = logits.reshape!(batch_size, @model.config.vocab_size)
      flat = rows.max_index(axis: 1)
      (flat - (flat.class.new(batch_size).seq * @model.config.vocab_size)).to_a
    end

    def cut_at_eot(sequence, stop_at_eot)
      return sequence unless stop_at_eot && !eot_token.nil?

      at = sequence.index(eot_token)
      at.nil? ? sequence : sequence[0..at]
    end

    def generate_recomputing(tokens, max_new_tokens, stop_at_eot, prof)
      max_new_tokens.times do
        token = next_token(tokens, prof: prof)
        tokens << token
        yield token if block_given?
        break if stop_at_eot && !eot_token.nil? && token == eot_token
      end
      tokens
    end

    def generate_cached(tokens, max_new_tokens, stop_at_eot, prof)
      kv = @model.new_cache
      logits = @model.prefill([tokens], cache: kv, prof: prof)
      produced = 0

      while produced < max_new_tokens
        token = argmax(logits)
        tokens << token
        produced += 1
        yield token if block_given?
        break if stop_at_eot && !eot_token.nil? && token == eot_token
        break if produced == max_new_tokens

        # The token just chosen is fed back at the position it now occupies.
        logits = @model.decode(token, tokens.size - 1, cache: kv, prof: prof)
      end

      tokens
    end

    def argmax(logits)
      return @sampler.call(logits, @model.config.vocab_size) if @sampler

      # Taking the row first makes this 1-D, so max_index is the token id
      # directly and there is no flattened-offset correction to get wrong.
      row = logits.reshape!(@model.config.vocab_size)
      # The single device-to-host readback per generated token. Anything else in
      # this loop would drain the queue every step (AGENTS.md).
      NArrayLLM.scalar(row.max_index).to_i
    end

    def normalize(prompt)
      ids = prompt.is_a?(Integer) ? [prompt] : prompt.to_a.flatten
      ids.map { |id| Integer(id) }
    end
  end
end
