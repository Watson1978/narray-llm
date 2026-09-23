# frozen_string_literal: true

require 'json'
require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-mamba.md).
#
# The gate is exact: the token sequence has to be the one mamba.c produces,
# every id, on both backends. A run that drifted is not a worse run, it is a
# different computation.
#
# The expectation comes from the C reference, not from this repository.
# script/mamba_fixtures.rb drives mamba.c's own forward() through
# script/mamba_dump.c and feeds its argmax back in.
class TestMambaGenerate < Test::Unit::TestCase
  include TestHelper

  FIXTURE = File.expand_path('../python/fixtures/mamba-130m_greedy.json', __dir__)

  # Long enough that a single wrong id anywhere in 24 blocks of recurrence
  # would show. The reference runs to 256; this stops earlier so the suite
  # stays quick, and the fixture holds the whole sequence so any prefix works.
  STEPS = 64

  class << self
    def generated
      @generated ||= begin
        model = NArrayLLM::Mamba::Model.load(File.join(TestHelper::DATA_DIR, 'mamba-130m.bin'))
        generator = NArrayLLM::Generator.new(model)
        # stop_at_eot would end the run at the first <|endoftext|>, and the
        # reference does not stop there.
        { model: model,
          tokens: generator.generate([0], max_new_tokens: STEPS, stop_at_eot: false) }
      end
    end
  end

  def fixture
    omit("#{FIXTURE} not found; run `ruby script/mamba_fixtures.rb`") unless File.exist?(FIXTURE)
    @fixture ||= JSON.parse(File.read(FIXTURE))
  end

  def generated
    require_data('mamba-130m.bin')
    fixture
    self.class.generated
  end

  # --- 受け入れ条件: mamba.c と完全一致する ---

  def test_tokens_match_the_reference_exactly
    expected = fixture['tokens'][0, STEPS + 1]
    assert_equal(expected, generated[:tokens], "#{NArrayLLM.gpu? ? 'Cumo' : 'Numo'} の生成列")
  end

  def test_the_prompt_is_the_delimiter
    assert_equal([0], fixture['prompt'])
    assert_equal(0, NArrayLLM::Mamba::Tokenizer::BOS_TOKEN)
    assert_equal(0, generated[:tokens].first)
  end

  # Mamba allocates nothing per position, so a sequence longer than anything
  # the reference was run at is not a different code path.
  def test_there_is_no_positional_limit
    assert_equal(Float::INFINITY, generated[:model].config.max_seq_len)
  end

  # The state is the whole of what carries between tokens, so resetting it has
  # to put the model back where it started.
  def test_resetting_the_state_repeats_the_run
    model = generated[:model]
    state = model.new_state
    first = Array.new(4) { |i| host(model.decode(i.zero? ? 0 : 50, cache: state).max_index) }
    state.reset
    again = Array.new(4) { |i| host(model.decode(i.zero? ? 0 : 50, cache: state).max_index) }
    assert_equal(first, again)
  end

  # The classifier is kept in the stored order, so a shared one is held once.
  # Transposing it cost another 147.3 MiB here.
  def test_parameter_bytes_counts_a_shared_classifier_once
    model = generated[:model]
    config = model.config
    assert_true(config.shared_classifier)
    assert_equal(492.6, (model.parameter_bytes / 1_048_576.0).round(1))
    assert_equal(4 * NArrayLLM::Mamba::Checkpoint.num_parameters(config),
                 model.parameter_bytes)
  end

  def test_an_out_of_range_token_is_rejected
    model = generated[:model]
    state = model.new_state
    assert_raise_kind_of(NArrayLLM::Error) { model.decode(-1, cache: state) }
    assert_raise_kind_of(NArrayLLM::Error) { model.decode(50_280, cache: state) }
  end
end
