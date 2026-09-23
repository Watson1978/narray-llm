# frozen_string_literal: true

module NArrayLLM
  # Temperature, top-k and top-p on top of one row of logits.
  #
  # The uniform comes from a Ruby Random and not from the device. Numo and Cumo
  # do not share a generator, so a device draw would break the acceptance
  # condition this repository keeps everywhere else -- that the two backends
  # answer the same token sequence (AGENTS.md). One scalar per token has nothing
  # to gain from the device anyway.
  class Sampler
    DEFAULT_TEMPERATURE = 1.0

    attr_reader :temperature, :top_k, :top_p, :seed

    def initialize(temperature: DEFAULT_TEMPERATURE, top_k: nil, top_p: nil, seed: nil)
      raise Error, "temperature must be positive, got #{temperature}" unless temperature.positive?

      @temperature = Float(temperature)
      @top_k = top_k&.to_i
      @top_p = top_p&.to_f
      @seed = seed
      @random = seed.nil? ? Random.new : Random.new(seed)
    end

    # logits: [V] or anything that reshapes to it. Answers the chosen id.
    def call(logits, vocab_size)
      row = logits.reshape!(vocab_size)
      probs = Ops.softmax_rows((row / @temperature).reshape!(1, vocab_size)).reshape!(vocab_size)
      probs = narrow(probs)
      Ops.sample_index(probs, @random.rand)
    end

    private

    # Both filters answer a 0/1 mask over the original positions, so they
    # compose by multiplication and the total is put back to 1 once at the end.
    def narrow(probs)
      return probs if @top_k.nil? && @top_p.nil?

      # Both filters want the same row in order, and a sort over the vocabulary
      # is 8 launches, so it is taken once when both of them are on.
      sorted = probs.sort if @top_k && @top_p
      keep = nil
      keep = Ops.top_k_keep(probs, @top_k, sorted: sorted) if @top_k
      if @top_p
        p_keep = Ops.top_p_keep(probs, @top_p, sorted: sorted)
        keep = keep.nil? ? p_keep : keep * p_keep
      end

      kept = probs * keep
      kept / kept.sum
    end
  end
end
