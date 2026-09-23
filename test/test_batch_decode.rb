# frozen_string_literal: true

require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-batch.md), against the real weights.
#
# The gate needs no new reference: batch 1 is the reference. A batch of B
# sequences has to produce, token for token, what B separate batch-1 runs
# produce. Nothing here compares floats.
class TestBatchDecode < Test::Unit::TestCase
  include TestHelper

  STEPS = 6
  # Three prompts of the same length. Position is shared across the batch, so
  # ragged prompts are stage 4's problem (docs/plans/PLAN-batch.md).
  PROMPTS = [
    [15496, 11, 616],
    [464, 2068, 7586],
    [40, 716, 257]
  ].freeze

  def model
    @model ||= begin
      require_data('gpt2_124M.bin')
      NArrayLLM::GPT2::Model.load(data_path('gpt2_124M.bin'))
    end
  end

  def argmax(logits, row)
    NArrayLLM.scalar(logits[row, 0, true].max_index).to_i
  end

  # One sequence through the ordinary path.
  def decode_alone(prompt, steps)
    cache = model.new_cache
    logits = model.prefill([prompt.dup], cache: cache)
    produced = []
    steps.times do |step|
      produced << argmax(logits, 0)
      break if step == steps - 1

      logits = model.decode(produced.last, prompt.size + produced.size - 1, cache: cache)
    end
    produced
  end

  # B sequences through one batched cache, prompts included.
  def decode_together(prompts, steps)
    batch = prompts.size
    cache = model.new_cache(batch_size: batch)
    logits = model.prefill(prompts.map(&:dup), cache: cache)

    produced = Array.new(batch) { [] }
    steps.times do |step|
      ids = (0...batch).map { |i| argmax(logits, i) }
      ids.each_with_index { |id, i| produced[i] << id }
      break if step == steps - 1

      logits = model.decode(ids, prompts.first.size + produced.first.size - 1, cache: cache)
    end
    produced
  end

  def test_a_batch_of_one_matches_the_unbatched_path
    want = decode_alone(PROMPTS.first, STEPS)
    assert_equal([want], decode_together([PROMPTS.first], STEPS))
  end

  def test_the_same_prompt_repeated_gives_the_same_sequence_every_time
    want = decode_alone(PROMPTS.first, STEPS)
    got = decode_together([PROMPTS.first] * 3, STEPS)
    assert_equal([want] * 3, got)
  end

  def test_different_prompts_each_match_their_own_unbatched_run
    want = PROMPTS.map { |prompt| decode_alone(prompt, STEPS) }
    assert_equal(want, decode_together(PROMPTS, STEPS))
    # Not a vacuous check: the three sequences have to differ from each other.
    assert_equal(3, want.uniq.size)
  end

  def test_batch_sizes_two_and_four_agree_with_batch_one
    [2, 4].each do |batch|
      prompts = (0...batch).map { |i| PROMPTS[i % PROMPTS.size] }
      want = prompts.map { |prompt| decode_alone(prompt, 4) }
      assert_equal(want, decode_together(prompts, 4), "batch #{batch}")
    end
  end

  def test_a_batch_that_does_not_match_the_cache_is_rejected
    cache = model.new_cache(batch_size: 2)
    assert_raise(NArrayLLM::Error) { model.decode([1, 2, 3], 0, cache: cache) }
    assert_raise(NArrayLLM::Error) { model.decode(1, 0, cache: cache) }
    assert_raise(NArrayLLM::Error) { model.prefill(PROMPTS, cache: cache) }
  end

  # --- stage 4: the generator (docs/plans/PLAN-batch.md) ---

  def generator
    @generator ||= NArrayLLM::Generator.new(model)
  end

  def test_generate_batch_matches_one_generate_per_prompt
    want = PROMPTS.map do |prompt|
      NArrayLLM::Generator.new(model).generate(prompt.dup, max_new_tokens: STEPS)
    end
    assert_equal(want, generator.generate_batch(PROMPTS.map(&:dup), max_new_tokens: STEPS))
  end

  def test_generate_batch_returns_the_prompt_with_the_continuation
    got = generator.generate_batch(PROMPTS.map(&:dup), max_new_tokens: STEPS)
    got.each_with_index do |sequence, i|
      assert_equal(PROMPTS[i], sequence.first(PROMPTS[i].size), "prompt #{i}")
      assert_equal(PROMPTS[i].size + STEPS, sequence.size)
    end
  end

  def test_generate_batch_rejects_prompts_of_different_lengths
    assert_raise(NArrayLLM::Error) do
      generator.generate_batch([[1, 2], [3, 4, 5]], max_new_tokens: 1)
    end
    assert_raise(NArrayLLM::Error) { generator.generate_batch([], max_new_tokens: 1) }
    assert_raise(NArrayLLM::Error) { generator.generate_batch([[]], max_new_tokens: 1) }
  end

  # Every sequence keeps stepping after it finishes, so the cut has to happen
  # afterwards and has to keep the EOT itself (what generate does).
  def test_each_sequence_is_cut_at_its_own_eot
    tokenizer = Struct.new(:eot_token).new(13)
    cutting = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    got = cutting.generate_batch(PROMPTS.map(&:dup), max_new_tokens: STEPS)
    plain = generator.generate_batch(PROMPTS.map(&:dup), max_new_tokens: STEPS)

    got.each_with_index do |sequence, i|
      body = sequence.drop(PROMPTS[i].size)
      assert_nil(body[0..-2].index(13), "13 may only be the last token of #{i}")
      want = plain[i].drop(PROMPTS[i].size)
      at = want.index(13)
      assert_equal(at.nil? ? want : want[0..at], body, "sequence #{i}")
    end
    # Not vacuous: 13 is a token this prompt set actually produces.
    assert_true(plain.any? { |sequence| sequence.include?(13) })
  end

  # Both ends of this tolerance are measured, not chosen. Batching changes the
  # order the GEMM folds in, so the same sequence comes out 1.0e-05 different
  # (Numo) and 7.2e-06 (Cumo) on rows reaching 18.4. A wrong transpose would put
  # another sequence's rows in the slot, and the closest two sequences here are
  # 3.79 apart. 1e-3 is 100x above the noise and 3800x below a mix-up.
  PREFILL_TOLERANCE = 1.0e-3

  # Stage 3: the prompt goes in for the whole batch at once. The cache is laid
  # out time major and the block hands over batch major rows, so a wrong
  # transpose here would still give the right shape and the wrong answer.
  def test_batched_prefill_fills_the_cache_the_same_way_one_at_a_time_does
    batch = PROMPTS.size
    together = model.new_cache(batch_size: batch)
    model.prefill(PROMPTS.map(&:dup), cache: together)

    PROMPTS.each_with_index do |prompt, i|
      alone = model.new_cache
      model.prefill([prompt.dup], cache: alone)
      model.config.num_layers.times do |layer|
        want = alone.view(layer)
        got = together.view(layer)
        %w[keys values].each_with_index do |what, which|
          worst = host((want[which] - NArrayLLM::Ops.contiguous(got[which][true, i, true])).abs.max)
          assert_operator(worst, :<, PREFILL_TOLERANCE, "L#{layer} #{what} of #{i}")
        end
      end
    end
  end
end
