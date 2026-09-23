# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-sampling.md).
class TestSamplingOps < Test::Unit::TestCase
  include TestHelper

  # Deliberately out of order, with a tie at 0.5 so the boundary case is
  # covered by the same fixture.
  VALUES = [0.1, 0.9, 0.5, 0.7, 0.5, 0.2].freeze

  def values
    XF.cast(VALUES)
  end

  def keeps(mask)
    mask.to_a.map(&:to_i)
  end

  def test_top_k_keeps_the_k_largest
    assert_equal([0, 1, 0, 0, 0, 0], keeps(NArrayLLM::Ops.top_k_keep(values, 1)))
    assert_equal([0, 1, 0, 1, 0, 0], keeps(NArrayLLM::Ops.top_k_keep(values, 2)))
  end

  # k is a floor, not a ceiling: the threshold is a value, so everything equal
  # to it survives. Asking for 3 of these six answers four.
  def test_ties_at_the_boundary_all_survive
    assert_equal([0, 1, 1, 1, 1, 0], keeps(NArrayLLM::Ops.top_k_keep(values, 3)))
  end

  def test_top_k_past_the_vocabulary_keeps_everything
    assert_equal([1] * 6, keeps(NArrayLLM::Ops.top_k_keep(values, 6)))
    assert_equal([1] * 6, keeps(NArrayLLM::Ops.top_k_keep(values, 99)))
    assert_raise(NArrayLLM::Error) { NArrayLLM::Ops.top_k_keep(values, 0) }
  end

  # [0.5, 0.3, 0.15, 0.05] descending. p of 0.6 keeps the first two: the first
  # alone is 0.5, which is under 0.6, so the second is the one that crosses it.
  def probs
    XF.cast([0.05, 0.5, 0.15, 0.3])
  end

  def test_top_p_keeps_the_set_that_crosses_p
    assert_equal([0, 1, 0, 1], keeps(NArrayLLM::Ops.top_p_keep(probs, 0.6)))
    assert_equal([0, 1, 0, 0], keeps(NArrayLLM::Ops.top_p_keep(probs, 0.4)))
    assert_equal([0, 1, 1, 1], keeps(NArrayLLM::Ops.top_p_keep(probs, 0.9)))
  end

  def test_top_p_of_one_keeps_everything
    assert_equal([1] * 4, keeps(NArrayLLM::Ops.top_p_keep(probs, 1.0)))
    assert_raise(NArrayLLM::Error) { NArrayLLM::Ops.top_p_keep(probs, 0.0) }
  end

  def test_sample_index_walks_the_running_total
    p = XF.cast([0.2, 0.3, 0.5])
    assert_equal(0, NArrayLLM::Ops.sample_index(p, 0.0))
    assert_equal(0, NArrayLLM::Ops.sample_index(p, 0.1))
    assert_equal(1, NArrayLLM::Ops.sample_index(p, 0.25))
    assert_equal(2, NArrayLLM::Ops.sample_index(p, 0.6))
    assert_equal(2, NArrayLLM::Ops.sample_index(p, 0.999999))
  end

  # A zero probability must never be picked, whatever u lands on its boundary.
  def test_a_zero_probability_is_never_picked
    p = XF.cast([0.5, 0.0, 0.5])
    100.times do |i|
      assert_not_equal(1, NArrayLLM::Ops.sample_index(p, i / 100.0))
    end
  end

  # The vocabulary this is for. cumsum crosses out of the band where cumo
  # falls back to a host loop (8192 elements), which is the point of the
  # exercise (docs/plans/PLAN-sampling.md).
  def test_the_real_vocabulary_size_works
    size = 50_257
    logits = XF::Math.sin(XF.new(size).seq * 0.007) * 10.0
    kept = NArrayLLM::Ops.top_k_keep(logits, 40)
    assert_operator(host(kept.sum), :>=, 40)
    p = NArrayLLM::Ops.softmax_rows(logits.reshape(1, size)).reshape(size)
    assert_in_delta(1.0, host(p.sum), 1e-4)
    index = NArrayLLM::Ops.sample_index(p, 0.5)
    assert_operator(index, :>=, 0)
    assert_operator(index, :<, size)
  end
end

# Stage 2 acceptance tests (docs/plans/PLAN-sampling.md), against the real weights.
class TestSamplingGeneration < Test::Unit::TestCase
  include TestHelper

  PROMPT = [15496, 11, 616].freeze
  STEPS = 12

  def generator
    @generator ||= begin
      require_data('gpt2_124M.bin')
      NArrayLLM::Generator.new(NArrayLLM::GPT2::Model.load(data_path('gpt2_124M.bin')))
    end
  end

  def run_with(sampler)
    generator.generate(PROMPT.dup, max_new_tokens: STEPS, stop_at_eot: false, sampler: sampler)
  end

  def greedy
    @greedy ||= generator.generate(PROMPT.dup, max_new_tokens: STEPS, stop_at_eot: false)
  end

  # Keeping one candidate leaves nothing to draw, so the uniform cannot matter.
  def test_top_k_of_one_reproduces_greedy
    assert_equal(greedy, run_with(NArrayLLM::Sampler.new(top_k: 1, seed: 1)))
    assert_equal(greedy, run_with(NArrayLLM::Sampler.new(top_k: 1, seed: 999)))
  end

  # The gate the whole repository runs on: the two backends have to agree token
  # for token. The uniform comes from Ruby's Random, so they do, and the
  # sequence is written out here rather than compared to itself -- otherwise
  # this would only prove the sampler is deterministic within one backend.
  SEEDED = [1438, 338, 5689, 25, 198, 198, 1, 17250, 13, 2011, 1438, 338].freeze

  def test_a_seed_pins_the_sequence_in_both_backends
    got = run_with(NArrayLLM::Sampler.new(top_k: 50, top_p: 0.9, seed: 42))
    assert_equal(PROMPT + SEEDED, got)
    assert_equal(got, run_with(NArrayLLM::Sampler.new(top_k: 50, top_p: 0.9, seed: 42)))
  end

  # Not vacuous: a different seed has to move the sequence, or the test above
  # would pass on a sampler that ignores the draw.
  def test_a_different_seed_moves_the_sequence
    a = run_with(NArrayLLM::Sampler.new(top_k: 50, seed: 1))
    b = run_with(NArrayLLM::Sampler.new(top_k: 50, seed: 2))
    assert_not_equal(a, b)
  end

  # Cooling towards zero concentrates the mass on the argmax, so the draw stops
  # mattering. 0.01 is cold enough for this prompt; it is not a limit claim.
  def test_a_cold_temperature_converges_on_greedy
    assert_equal(greedy, run_with(NArrayLLM::Sampler.new(temperature: 0.01, seed: 7)))
  end

  def test_the_sampler_rejects_a_temperature_of_zero
    assert_raise(NArrayLLM::Error) { NArrayLLM::Sampler.new(temperature: 0.0) }
  end

  # top_p alone has to bite: with a tight p the sequence must differ from what
  # the same seed produces unfiltered.
  def test_top_p_narrows_the_draw
    wide = run_with(NArrayLLM::Sampler.new(seed: 3))
    tight = run_with(NArrayLLM::Sampler.new(top_p: 0.1, seed: 3))
    assert_not_equal(wide, tight)
  end
end
