# frozen_string_literal: true

require_relative 'test_helper'

# Stage 3 acceptance tests (docs/plans/PLAN-llama2.md).
class TestLlama2KVCache < Test::Unit::TestCase
  include TestHelper

  # The same string test_llama2_generate.rb checks, so the cached path is held
  # to llama2.c's output and not merely to the uncached path.
  STORIES260K_200 =
    'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. ' \
    "One day, she saw a big, red ball. She wanted to play with it, but it was too high.\n" \
    'Lily\'s mom said, "Lily, let\'s go to the park." Lily was sad and didn\'t know what to do. ' \
    'She said, "I want to play with your ball, but I can\'t find it."' "\n" \
    'Lily was sad and didn\'t know what to do. She said, "I\'m sorry, Lily. I didn\'t know what ' \
    'to do."' "\n" \
    'Lily didn\'t want to help her mom, so she'

  TOKENIZERS = { 'stories260K' => 'tok512.bin', 'stories110M' => 'tokenizer.bin' }.freeze

  class << self
    def prepared(name)
      @prepared ||= {}
      @prepared[name] ||= begin
        model = NArrayLLM::Llama2::Model.load(File.join(TestHelper::DATA_DIR, "#{name}.bin"))
        tokenizer = NArrayLLM::Llama2::Tokenizer.load(
          File.join(TestHelper::DATA_DIR, TOKENIZERS.fetch(name)),
          vocab_size: model.config.vocab_size
        )
        { model: model, tokenizer: tokenizer,
          generator: NArrayLLM::Generator.new(model, tokenizer: tokenizer) }
      end
    end
  end

  def prepared(name)
    require_data("#{name}.bin")
    require_data(TOKENIZERS.fetch(name))
    self.class.prepared(name)
  end

  def bos
    NArrayLLM::Llama2::Tokenizer::BOS_TOKEN
  end

  # --- 受け入れ条件 1: キャッシュ有無で生成トークン列が完全一致する ---

  data('stories260K' => ['stories260K', 200], 'stories110M' => ['stories110M', 32])
  def test_the_cache_does_not_change_a_single_token(params)
    name, length = params
    f = prepared(name)
    without = f[:generator].generate([bos], max_new_tokens: length, cache: false)
    with = f[:generator].generate([bos], max_new_tokens: length, cache: true)

    if without != with
      first = without.zip(with).index { |a, b| a != b }
      flunk("diverged at #{first}: uncached #{without[first]}, cached #{with[first]}")
    end
    assert_equal(without, with)
  end

  def test_the_cached_path_still_matches_run_c
    f = prepared('stories260K')
    tokens = f[:generator].generate([bos], max_new_tokens: 200, cache: true)
    assert_equal(STORIES260K_200.b, f[:tokenizer].render(tokens).b)
  end

  # --- K は RoPE の後に積む ---

  # The one ordering docs/plans/PLAN-llama2.md warns about. run.c rotates s->k in place and
  # s->k already points into the cache row (run.c:259, :279), so the cached key
  # is the rotated one. Caching the pre-RoPE key would rotate it again by the
  # wrong position on every later step, which the reference dump can see
  # directly: its "k" record is post-RoPE, "k_pre_rope" is not.
  def test_the_cached_key_is_the_rotated_one
    require_data('stories260K_debug_state.bin')
    f = prepared('stories260K')
    state = NArrayLLM::Llama2::DebugState.load(data_path('stories260K_debug_state.bin'))
    cache = f[:model].new_cache
    f[:model].prefill(state.tokens, cache: cache)

    config = f[:model].config
    config.num_layers.times do |layer|
      keys, values = cache.view(layer)
      assert_equal(state.steps, keys.shape[0], "layer #{layer} row count")
      state.steps.times do |pos|
        post = NArrayLLM::Compare.diff(state['k', pos, layer], keys[pos, true],
                                       tolerance: 1e-4, label: "k pos=#{pos} layer=#{layer}")
        assert_true(post.ok?, post.to_s)
        assert_close(state['v', pos, layer], values[pos, true], "v pos=#{pos} layer=#{layer}")
      end
      # And the un-rotated key is a different thing, so the check above is not
      # passing by accident at every position.
      pre = NArrayLLM::Compare.diff(state['k_pre_rope', state.steps - 1, layer],
                                    keys[state.steps - 1, true])
      assert_operator(pre.max_abs, :>, 1e-3,
                      "layer #{layer}: the pre-RoPE key must not equal the cached one")
    end
  end

  def test_decode_continues_the_sequence_the_prefill_started
    require_data('stories260K_debug_state.bin')
    f = prepared('stories260K')
    state = NArrayLLM::Llama2::DebugState.load(data_path('stories260K_debug_state.bin'))
    cache = f[:model].new_cache
    logits = f[:model].prefill(state.tokens, cache: cache)

    # The prompt was run.c's own greedy output, so the token after it is known.
    next_token = NArrayLLM.scalar(logits.reshape(f[:model].config.vocab_size).max_index)
    stepped = f[:model].decode(next_token, state.tokens.size, cache: cache)
    assert_equal([1, f[:model].config.vocab_size], stepped.shape)
    assert_equal(state.tokens.size + 1, cache.length)
  end

  # --- 受け入れ条件 2: キャッシュが仕事を減らしている ---

  # Counted, not timed. Timing both paths inside one process measures the
  # allocator as much as the work: stories260K at 128 tokens comes out 2.7x
  # faster with the cache when each condition gets its own process, and 0.66x
  # when they share one, which is AGENTS.md's rule 11 exactly. The tokens/sec
  # claim belongs to script/llama2_generate.rb and docs/results/llama2-110m.md.
  def test_the_cache_replaces_the_growing_forward_passes
    f = prepared('stories260K')

    without = CountingModel.new(f[:model])
    NArrayLLM::Generator.new(without, tokenizer: f[:tokenizer])
                        .generate([bos], max_new_tokens: 8, cache: false)
    with = CountingModel.new(f[:model])
    NArrayLLM::Generator.new(with, tokenizer: f[:tokenizer])
                        .generate([bos], max_new_tokens: 8, cache: true)

    # Without the cache every step runs the whole sequence again.
    assert_equal(8, without.forwards, 'one full forward per generated token')
    assert_equal([1, 2, 3, 4, 5, 6, 7, 8], without.forward_lengths)
    assert_equal(0, without.decodes)

    # With it the prompt runs once and each step afterwards is a single token.
    assert_equal(1, with.forwards, 'the prompt is the only full forward')
    assert_equal([1], with.forward_lengths)
    assert_equal(7, with.decodes, 'and the rest are single-token steps')

    # The quantity that matters: rows pushed through the blocks.
    assert_equal(36, without.rows, '1 + 2 + ... + 8 grows with the square of the length')
    assert_equal(8, with.rows, 'the cache makes it linear')
  end

  # --- 受け入れ条件 3: seq_len を超える要求を明示的に拒否する ---

  def test_a_generation_past_max_seq_len_is_rejected
    f = prepared('stories260K')
    limit = f[:model].config.max_seq_len
    assert_raise(NArrayLLM::Error) do
      f[:generator].generate([bos], max_new_tokens: limit, cache: true)
    end
  end

  def test_a_decode_past_max_seq_len_is_rejected
    f = prepared('stories260K')
    cache = f[:model].new_cache
    assert_raise(NArrayLLM::Error) do
      f[:model].decode(bos, f[:model].config.max_seq_len, cache: cache)
    end
  end

  def test_the_cache_refuses_to_overflow
    f = prepared('stories260K')
    config = f[:model].config
    cache = f[:model].new_cache
    rows = XM::SFloat.zeros(config.max_seq_len + 1, config.kv_dim)
    assert_raise(NArrayLLM::Error) { cache.append(0, rows, rows) }
  end

  # --- GQA のぶんキャッシュが狭い ---

  def test_the_cache_is_kv_dim_wide_not_dim_wide
    f = prepared('stories260K')
    config = f[:model].config
    cache = f[:model].new_cache
    assert_equal(config.kv_dim, cache.channels)
    assert_equal(config.dim / 2, cache.channels, 'kv_mul is 2, so K and V are half as wide')
    assert_equal(2 * config.num_layers * config.max_seq_len * config.kv_dim * 4, cache.bytes)
  end

  def test_multi_head_models_cache_the_full_width
    f = prepared('stories110M')
    config = f[:model].config
    assert_equal(config.dim, f[:model].new_cache.channels)
  end

  # Counts what the generator asks the model to do, and how many rows each call
  # carries. Delegates everything else so the numbers are the real run's.
  class CountingModel
    attr_reader :forwards, :decodes, :forward_lengths

    def initialize(model)
      @model = model
      @forwards = 0
      @decodes = 0
      @forward_lengths = []
    end

    def config
      @model.config
    end

    def new_cache
      @model.new_cache
    end

    def rows
      @forward_lengths.sum + @decodes
    end

    def forward(tokens, **kwargs)
      @forwards += 1
      @forward_lengths << Array(tokens).flatten.size
      @model.forward(tokens, **kwargs)
    end

    def prefill(tokens, cache:, **kwargs)
      @model.prefill(tokens, cache: cache, **kwargs).tap { record_prefill(tokens) }
    end

    def decode(token_id, position, cache:, **kwargs)
      @decodes += 1
      @model.decode(token_id, position, cache: cache, **kwargs)
    end

    private

    # prefill runs one forward internally; count it once, not twice.
    def record_prefill(tokens)
      @forwards += 1
      @forward_lengths << Array(tokens).flatten.size
    end
  end

  private

  def assert_close(expected, actual, label)
    d = NArrayLLM::Compare.diff(expected.reshape(*actual.shape), actual,
                                tolerance: 1e-4, label: label)
    assert_true(d.ok?, d.to_s)
  end
end
