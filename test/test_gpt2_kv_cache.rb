# frozen_string_literal: true

require_relative 'test_helper'

class TestKVCache < Test::Unit::TestCase
  include TestHelper

  LAYERS = 3
  MAX_T = 6
  CHANNELS = 4

  def cache
    @cache ||= NArrayLLM::KVCache.new(num_layers: LAYERS, max_seq_len: MAX_T, channels: CHANNELS)
  end

  def row(value)
    XM::SFloat.new(CHANNELS).seq + value
  end

  def test_starts_empty
    assert_equal(0, cache.length)
    LAYERS.times { |l| assert_equal([0, CHANNELS], cache.view(l).first.shape) }
  end

  def test_appending_one_row_at_a_time_grows_the_view
    3.times do |t|
      cache.append(0, row(t * 10), row(t * 10 + 100))
      keys, values = cache.view(0)
      assert_equal([t + 1, CHANNELS], keys.shape)
      assert_equal([t + 1, CHANNELS], values.shape)
      assert_equal(t + 1, cache.length(0))
    end

    keys, values = cache.view(0)
    3.times do |t|
      assert_equal(row(t * 10).to_a, keys[t, true].to_a, "key row #{t}")
      assert_equal(row(t * 10 + 100).to_a, values[t, true].to_a, "value row #{t}")
    end
  end

  def test_a_later_append_does_not_disturb_earlier_rows
    cache.append(0, row(1), row(2))
    before = cache.view(0).first[0, true].to_a
    cache.append(0, row(50), row(60))
    assert_equal(before, cache.view(0).first[0, true].to_a)
  end

  def test_layers_are_independent
    cache.append(0, row(1), row(2))
    cache.append(0, row(3), row(4))
    cache.append(1, row(9), row(9))
    assert_equal(2, cache.length(0))
    assert_equal(1, cache.length(1))
    assert_equal(0, cache.length(2))
    assert_equal(row(9).to_a, cache.view(1).first[0, true].to_a)
  end

  def test_accepts_a_block_of_rows_for_prefill
    keys = XM::SFloat.new(4, CHANNELS).seq
    values = XM::SFloat.new(4, CHANNELS).seq + 100
    cache.append(2, keys, values)
    assert_equal(4, cache.length(2))
    got_keys, got_values = cache.view(2)
    assert_equal(keys.to_a, got_keys.to_a)
    assert_equal(values.to_a, got_values.to_a)

    # And a single row can follow a block.
    cache.append(2, row(7), row(8))
    assert_equal(5, cache.length(2))
    assert_equal(row(7).to_a, cache.view(2).first[4, true].to_a)
  end

  def test_accepts_a_two_dimensional_single_row
    cache.append(0, row(1).reshape(1, CHANNELS), row(2).reshape(1, CHANNELS))
    assert_equal(1, cache.length(0))
    assert_equal(row(1).to_a, cache.view(0).first[0, true].to_a)
  end

  def test_view_reflects_writes_made_after_it_was_taken
    cache.append(0, row(1), row(2))
    keys, = cache.view(0)
    assert_equal([1, CHANNELS], keys.shape)
    cache.append(0, row(3), row(4))
    fresh, = cache.view(0)
    assert_equal([2, CHANNELS], fresh.shape, 'view must be taken after the append')
  end

  def test_reset_empties_every_layer
    LAYERS.times { |l| cache.append(l, row(l), row(l)) }
    cache.reset
    LAYERS.times { |l| assert_equal(0, cache.length(l)) }
  end

  # --- errors and sizing ---

  def test_rejects_appending_past_max_seq_len
    MAX_T.times { cache.append(0, row(1), row(2)) }
    assert_equal(MAX_T, cache.length(0))
    assert_raise(NArrayLLM::Error) { cache.append(0, row(1), row(2)) }
  end

  def test_rejects_a_prefill_block_that_would_overflow
    assert_raise(NArrayLLM::Error) do
      cache.append(0, XM::SFloat.zeros(MAX_T + 1, CHANNELS), XM::SFloat.zeros(MAX_T + 1, CHANNELS))
    end
  end

  def test_rejects_a_bad_layer_or_shape
    assert_raise(NArrayLLM::Error) { cache.append(LAYERS, row(1), row(2)) }
    assert_raise(NArrayLLM::Error) { cache.view(-1) }
    assert_raise(NArrayLLM::Error) { cache.append(0, XM::SFloat.zeros(1, CHANNELS + 1), XM::SFloat.zeros(1, CHANNELS + 1)) }
    assert_raise(NArrayLLM::Error) { cache.append(0, XM::SFloat.zeros(2, CHANNELS), XM::SFloat.zeros(1, CHANNELS)) }
  end

  # K and V, fp32: 12 * 1024 * 768 * 2 * 4 bytes for GPT-2 124M.
  def test_reports_its_size
    assert_equal(2 * LAYERS * MAX_T * CHANNELS * 4, cache.bytes)
    assert_equal(75_497_472,
                 NArrayLLM::KVCache.bytes_for(num_layers: 12, max_seq_len: 1024, channels: 768))
  end

  # K and V, fp32: 12 * 1024 * 768 * 2 * 4 bytes for GPT-2 124M.
  def test_reports_its_size_for_gpt2_124m
    assert_equal(75_497_472,
                 NArrayLLM::KVCache.bytes_for(num_layers: 12, max_seq_len: 1024, channels: 768))
  end

  # --- batch axis (docs/plans/PLAN-batch.md, stage 1) ---

  BATCH = 3

  def batched
    @batched ||= NArrayLLM::KVCache.new(num_layers: LAYERS, max_seq_len: MAX_T,
                                        channels: CHANNELS, batch_size: BATCH)
  end

  # One row per sequence, each carrying its own value.
  def rows(value)
    XM::SFloat.new(BATCH, CHANNELS).seq + value
  end

  def test_batch_size_one_is_the_default_and_keeps_the_flat_shape
    assert_equal(1, cache.batch_size)
    assert_false(cache.batched?)
    cache.append(0, row(1), row(2))
    assert_equal([1, CHANNELS], cache.view(0).first.shape)
  end

  # Time is the outer axis: a growing slice of [maxT, B, C] is contiguous, and
  # decode_attention reshapes the view in place.
  def test_batched_views_are_time_major_and_contiguous
    assert_equal([0, BATCH, CHANNELS], batched.view(0).first.shape)
    batched.append(0, rows(1), rows(100))
    keys, values = batched.view(0)
    assert_equal([1, BATCH, CHANNELS], keys.shape)
    assert_equal([1, BATCH, CHANNELS], values.shape)
    assert_equal(rows(1).to_a, keys[0, true, true].to_a)
    assert_equal(rows(100).to_a, values[0, true, true].to_a)
    assert_nothing_raised('the view has to survive reshape!') do
      keys.reshape!(1, BATCH, 2, CHANNELS / 2)
    end
  end

  def test_each_sequence_keeps_its_own_rows
    3.times { |t| batched.append(0, rows(t * 10), rows(t * 10 + 100)) }
    keys, = batched.view(0)
    assert_equal([3, BATCH, CHANNELS], keys.shape)
    3.times do |t|
      assert_equal(rows(t * 10).to_a, keys[t, true, true].to_a, "position #{t}")
    end
  end

  def test_batched_prefill_takes_a_block_per_sequence
    keys = XM::SFloat.new(4, BATCH, CHANNELS).seq
    values = XM::SFloat.new(4, BATCH, CHANNELS).seq + 1000
    batched.append(1, keys, values)
    assert_equal(4, batched.length(1))
    got_keys, got_values = batched.view(1)
    assert_equal(keys.to_a, got_keys.to_a)
    assert_equal(values.to_a, got_values.to_a)

    batched.append(1, rows(7), rows(8))
    assert_equal(5, batched.length(1))
    assert_equal(rows(7).to_a, batched.view(1).first[4, true, true].to_a)
  end

  def test_batched_size_scales_with_the_batch
    assert_equal(BATCH * 2 * LAYERS * MAX_T * CHANNELS * 4, batched.bytes)
    assert_equal(8 * 75_497_472,
                 NArrayLLM::KVCache.bytes_for(num_layers: 12, max_seq_len: 1024,
                                              channels: 768, batch_size: 8))
  end

  def test_batched_rejects_a_wrong_batch_or_overflow
    assert_raise(NArrayLLM::Error) { batched.append(0, rows(1)[0...2, true], rows(2)[0...2, true]) }
    assert_raise(NArrayLLM::Error) do
      batched.append(0, XM::SFloat.zeros(MAX_T + 1, BATCH, CHANNELS),
                     XM::SFloat.zeros(MAX_T + 1, BATCH, CHANNELS))
    end
    assert_raise(NArrayLLM::Error) do
      NArrayLLM::KVCache.new(num_layers: 1, max_seq_len: 1, channels: 1, batch_size: 0)
    end
  end
end

# Stage 3 acceptance tests (docs/plans/PLAN-gpt2.md), against the real weights.
class TestKVCacheGeneration < Test::Unit::TestCase
  include TestHelper

  # The two backends need different lengths to show the same trend, for opposite
  # reasons. On CPU the recompute is FLOP-bound and its O(n^2) cost dominates
  # immediately, but a long no-cache run takes minutes. On GPU a step is bound by
  # kernel launches, which the cache does not reduce, so its advantage only
  # becomes measurable once the recomputed sequence is long -- and a long run
  # there costs seconds. Measured speedups: 4.5x -> 9.7x on Numo, 1.1x -> 1.3x
  # on Cumo.
  SHORT = Integer(ENV.fetch('KV_TEST_SHORT') { NArrayLLM.gpu? ? 16 : 12 })
  LONG = Integer(ENV.fetch('KV_TEST_LONG') { NArrayLLM.gpu? ? 256 : 32 })

  class << self
    def model
      @model ||= NArrayLLM::GPT2::Model.load(File.join(TestHelper::DATA_DIR, 'gpt2_124M.bin'))
    end

    def tokenizer
      @tokenizer ||= NArrayLLM::GPT2::Tokenizer.load(File.join(TestHelper::DATA_DIR, 'gpt2_tokenizer.bin'))
    end

    def generator
      @generator ||= NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    end
  end

  def setup
    require_data('gpt2_124M.bin')
    require_data('gpt2_tokenizer.bin')
  end

  def prompt
    [self.class.tokenizer.eot_token]
  end

  # --- 受け入れ条件 1: キャッシュ有無で生成トークン列が一致する ---

  def test_cached_and_recomputing_generation_agree
    recomputed = self.class.generator.generate(prompt, max_new_tokens: LONG, cache: false)
    cached = self.class.generator.generate(prompt, max_new_tokens: LONG, cache: true)
    flunk(divergence_report(recomputed, cached)) if recomputed != cached
    assert_equal(recomputed, cached)
  end

  def test_the_cached_path_produces_the_same_first_token
    one = self.class.generator.generate(prompt, max_new_tokens: 1, cache: true)
    many = self.class.generator.generate(prompt, max_new_tokens: SHORT, cache: true)
    assert_equal(one, many[0, one.size])
  end

  def test_a_multi_token_prompt_is_prefilled_correctly
    seed = self.class.generator.generate(prompt, max_new_tokens: 4, cache: false)
    from_seed = self.class.generator.generate(seed, max_new_tokens: SHORT, cache: true)
    straight = self.class.generator.generate(prompt, max_new_tokens: 4 + SHORT, cache: false)
    assert_equal(straight, from_seed, 'prefill over a longer prompt must match a plain run')
  end

  # --- 受け入れ条件 2: tokens/sec が改善し、系列長が伸びるほど差が開く ---

  # A timing assertion, so the thresholds are the weakest ones that still state
  # the property: never slower, a real gain at the longer length, and a gain that
  # grows. The 1.05 factor is measurement slack -- the observed growth is
  # 4.5x -> 9.7x on Numo and 1.13x -> 1.29x on Cumo.
  def test_the_cache_is_faster_and_the_gap_widens_with_length
    short_speedup = speedup(SHORT)
    long_speedup = speedup(LONG)
    # In-suite ratios only. They run inside a process that has already loaded
    # several models and run other tests, and on Cumo they come out noticeably
    # higher than the same measurement in a dedicated process (1.3x there versus
    # up to 2.9x here). The cause is not pinned down, so do not quote these as
    # results -- script/gpt2_generate.rb is the benchmark.
    notify(format('KV cache speedup, in-suite ratio only (%s): %d tokens %.2fx, %d tokens %.2fx ' \
                  '-- benchmark with script/gpt2_generate.rb, not these numbers',
                  NArrayLLM.gpu? ? 'Cumo' : 'Numo', SHORT, short_speedup, LONG, long_speedup))

    assert_operator(short_speedup, :>, 1.0, "cache must not be slower at #{SHORT} tokens")
    assert_operator(long_speedup, :>, 1.1, "cache must be faster at #{LONG} tokens")
    assert_operator(long_speedup, :>, short_speedup * 1.05,
                    'the advantage must grow with sequence length')
  end

  # --- 受け入れ条件 3: maxT を超える生成要求を明示的なエラーで拒否する ---

  def test_rejects_generation_beyond_max_seq_len
    limit = self.class.model.config.max_seq_len
    assert_raise(NArrayLLM::Error) do
      self.class.generator.generate(prompt, max_new_tokens: limit, cache: true)
    end
  end

  def test_the_cache_itself_refuses_to_overflow
    cache = self.class.model.new_cache
    limit = self.class.model.config.max_seq_len
    channels = self.class.model.config.channels
    cache.append(0, XM::SFloat.zeros(limit, channels), XM::SFloat.zeros(limit, channels))
    assert_raise(NArrayLLM::Error) do
      cache.append(0, XM::SFloat.zeros(1, channels), XM::SFloat.zeros(1, channels))
    end
  end

  def test_decode_refuses_a_position_beyond_max_seq_len
    cache = self.class.model.new_cache
    assert_raise(NArrayLLM::Error) do
      self.class.model.decode(0, self.class.model.config.max_seq_len, cache: cache)
    end
  end

  def test_decode_refuses_an_invalid_token_id
    cache = self.class.model.new_cache
    vocab = self.class.model.config.vocab_size
    assert_raise(NArrayLLM::Error) { self.class.model.decode(vocab, 0, cache: cache) }
  end

  private

  def speedup(length)
    self.class.generator.generate(prompt, max_new_tokens: 2, cache: true) # warm up
    without = best_of(3) { self.class.generator.generate(prompt, max_new_tokens: length, cache: false) }
    with = best_of(3) { self.class.generator.generate(prompt, max_new_tokens: length, cache: true) }
    without / with
  end

  def best_of(count)
    Array.new(count) do
      started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      yield
      NArrayLLM::Profiler::NULL.synchronize
      Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
    end.min
  end

  def divergence_report(recomputed, cached)
    index = recomputed.each_index.find { |i| recomputed[i] != cached[i] }
    prefix = recomputed[0, index]
    without = self.class.model.forward([prefix], last_only: true)
    with = cached_logits_after(prefix)

    lines = ["キャッシュ有無で位置 #{index} の生成が割れた",
             "  cache 無し: #{recomputed.inspect}",
             "  cache 有り: #{cached.inspect}",
             "  共通接頭辞: #{self.class.tokenizer.decode(prefix).inspect}",
             "  cache 無しの logits 上位 5: #{format_top(without)}",
             "  cache 有りの logits 上位 5: #{format_top(with)}"]

    gap_without = top_gap(without)
    gap_with = top_gap(with)
    lines << format('  1 位と 2 位の差: cache 無し %.6f / cache 有り %.6f', gap_without, gap_with)
    if [gap_without, gap_with].min <= 1e-3
      lines << '  差が 1e-3 以下の僅差。docs/cumo-issues.md に記録した「cuBLAS が M=1 と'
      lines << '  M=n でカーネルを切り替えて結果が変わる」現象と同根の可能性が高い。'
    end
    lines.join("\n")
  end

  # Replays the cached path so the two logits come from the paths under test.
  def cached_logits_after(prefix)
    cache = self.class.model.new_cache
    logits = self.class.model.prefill([prefix[0, 1]], cache: cache)
    (1...prefix.size).each { |i| logits = self.class.model.decode(prefix[i], i, cache: cache) }
    logits
  end

  def sorted_top(logits, count = 5)
    vocab = self.class.model.config.vocab_size
    logits.reshape(vocab).to_a.each_with_index.max_by(count) { |value, _| value }
  end

  def format_top(logits)
    sorted_top(logits).map do |value, id|
      format('%d(%s)=%.6f', id, self.class.tokenizer.decode(id).inspect, value)
    end.join(', ')
  end

  def top_gap(logits)
    top = sorted_top(logits, 2)
    top[0][0] - top[1][0]
  end
end
