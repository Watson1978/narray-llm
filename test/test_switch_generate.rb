# frozen_string_literal: true

require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-switch.md).
#
# The gate is the same shape as the other models' — the generated token
# sequence has to match, exactly — but the reference is transformers rather
# than a C implementation. python/switch_fixtures.py writes the sequences with
# greedy forced explicitly.
class TestSwitchGenerate < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/switch-base-8.safetensors', __dir__)
  FIXTURE = File.expand_path('../python/fixtures/switch-base-8_greedy.json', __dir__)

  def setup
    omit("#{MODEL} not found; run `rake prepare:switch`") unless File.exist?(MODEL)
    omit("#{FIXTURE} not found; run python/switch_fixtures.py") unless File.exist?(FIXTURE)

    @fixture = JSON.parse(File.read(FIXTURE))
  end

  def model(router: :dispatch)
    @models ||= {}
    @models[router] ||= NArrayLLM::Switch::Model.load(MODEL, router: router)
  end

  def prompt(name)
    @fixture.fetch('prompts').fetch(name)
  end

  data('dispatch', :dispatch)
  data('dense', :dense)
  def test_the_token_sequence_matches_transformers(router)
    %w[short sentinel long].each do |name|
      spec = prompt(name)
      tokens = model(router: router).generate(spec['input_ids'],
                                              max_new_tokens: @fixture['max_new_tokens'])
      assert_equal(spec['tokens'], tokens, "#{router} #{name}")
    end
  end

  # The decoder starts on decoder_start_token_id, which is the pad id here, and
  # the answer carries it so both sides count from the same place.
  def test_the_sequence_starts_on_the_decoder_start_token
    spec = prompt('short')
    tokens = model.generate(spec['input_ids'], max_new_tokens: 4)
    assert_equal(model.config.decoder_start_token_id, tokens.first)
    assert_equal(5, tokens.size)
  end

  # The sentinel prompt runs into eos before the budget, which is what makes
  # it worth keeping: the loop has to stop there rather than at the limit.
  def test_generation_stops_at_eos
    spec = prompt('sentinel')
    assert_operator(spec['tokens'].size, :<, @fixture['max_new_tokens'] + 1,
                    'この fixture は eos で止まっていること')
    assert_equal(model.config.eos_token_id, spec['tokens'].last)
    assert_equal(spec['tokens'], model.generate(spec['input_ids'],
                                                max_new_tokens: @fixture['max_new_tokens']))
  end

  def test_a_second_run_repeats_the_first
    spec = prompt('sentinel')
    first = model.generate(spec['input_ids'], max_new_tokens: 8)
    assert_equal(first, model.generate(spec['input_ids'], max_new_tokens: 8))
  end

  def test_an_out_of_range_token_is_rejected
    spec = prompt('short')
    states = model.encode(spec['input_ids'])
    cache = model.new_cache(states, max_seq_len: 4)
    assert_raise_kind_of(NArrayLLM::Error) { model.decode(-1, cache: cache) }
    assert_raise_kind_of(NArrayLLM::Error) { model.decode(model.config.vocab_size, cache: cache) }
  end

  # The decoder's bias is one directional: everything ahead of the query folds
  # onto bucket 0, so a single query row at the end never sees a distinct
  # bucket for a position it has not reached.
  def test_the_decoder_bias_is_one_directional
    config = model.config
    buckets = NArrayLLM::Switch::Stack.buckets(1, 5, config, bidirectional: false, offset: 2)
    assert_equal([2, 1, 0, 0, 0], buckets, '先の位置は 0 に畳まれる')
    forward = NArrayLLM::Switch::Stack.buckets(1, 5, config, bidirectional: true, offset: 2)
    assert_operator(forward[3], :>=, config.relative_attention_num_buckets / 2,
                    '双方向なら先の位置は上半分に入る')
  end
end
