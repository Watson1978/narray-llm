# frozen_string_literal: true

require 'open3'
require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-gpt2.md).
class TestGenerate < Test::Unit::TestCase
  include TestHelper

  ROOT = File.expand_path('..', __dir__)

  # Long enough for a divergence to show up, short enough that the CPU backend
  # finishes in seconds: without a KV cache every step recomputes the sequence.
  GENERATED_TOKENS = Integer(ENV.fetch('GENERATE_TEST_TOKENS', 32))

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

    def unconditional
      @unconditional ||= generator.generate([tokenizer.eot_token], max_new_tokens: GENERATED_TOKENS)
    end
  end

  def setup
    require_data('gpt2_124M.bin')
    require_data('gpt2_tokenizer.bin')
  end

  # --- 受け入れ条件 1: Numo と Cumo が同一のトークン列を生成する ---

  def test_backends_generate_the_same_token_sequence
    mine = self.class.unconditional
    theirs = other_backend_tokens(mine.first, GENERATED_TOKENS)
    omit(@omit_reason) if theirs.nil?

    here = NArrayLLM.gpu? ? 'Cumo' : 'Numo'
    there = NArrayLLM.gpu? ? 'Numo' : 'Cumo'
    if mine != theirs
      flunk(divergence_report(mine, theirs, here, there))
    end
    notify("#{here} と #{there} が #{GENERATED_TOKENS} トークン一致: " \
           "#{self.class.tokenizer.decode(mine[1..]).inspect}")
    assert_equal(mine, theirs)
  end

  # --- 受け入れ条件 2: 生成長 1 と生成長 N で先頭トークンが一致する ---

  def test_first_token_is_the_same_for_length_one_and_length_n
    prompt = [self.class.tokenizer.eot_token]
    one = self.class.generator.generate(prompt, max_new_tokens: 1)
    many = self.class.unconditional
    assert_equal(prompt.size + 1, one.size)
    assert_equal(one[prompt.size], many[prompt.size],
                 'carrying state between steps must not change the first token')
    assert_equal(one, many[0, one.size])
  end

  # --- 受け入れ条件 3: EOT トークンで停止する ---

  # Both generation paths have to stop the same way, so the control-flow tests
  # run against each of them.
  data('recomputing', false)
  data('cached', true)
  def test_stops_at_the_eot_token(use_cache)
    eot = 7
    model = ScriptedModel.new([5, 6, eot, 9], vocab_size: 16, max_seq_len: 32)
    generator = NArrayLLM::Generator.new(model, tokenizer: FakeTokenizer.new(eot))
    tokens = generator.generate([1], max_new_tokens: 10, cache: use_cache)
    assert_equal([1, 5, 6, eot], tokens)
    assert_equal(3, model.calls, 'generation must stop the moment EOT is emitted')
  end

  data('recomputing', false)
  data('cached', true)
  def test_keeps_going_past_eot_when_stopping_is_disabled(use_cache)
    eot = 7
    model = ScriptedModel.new([5, eot, 9, 2], vocab_size: 16, max_seq_len: 32)
    generator = NArrayLLM::Generator.new(model, tokenizer: FakeTokenizer.new(eot))
    assert_equal([1, 5, eot, 9, 2],
                 generator.generate([1], max_new_tokens: 4, stop_at_eot: false, cache: use_cache))
  end

  # last_only belongs to the recomputing path; the cached path reaches the same
  # place through prefill and decode instead.
  def test_generation_uses_the_last_position_only_path
    model = ScriptedModel.new([5, 6], vocab_size: 16, max_seq_len: 32)
    NArrayLLM::Generator.new(model).generate([1], max_new_tokens: 2, cache: false)
    assert_equal([true, true], model.last_only_flags,
                 'greedy decoding must not compute logits for every position')
  end

  def test_the_cached_path_prefills_once_then_decodes
    model = ScriptedModel.new([5, 6, 7], vocab_size: 16, max_seq_len: 32)
    NArrayLLM::Generator.new(model).generate([1], max_new_tokens: 3, cache: true)
    assert_equal(%i[prefill decode decode], model.phases)
  end

  # --- 最終位置版と全位置版の等価性 ---

  def test_last_position_logits_choose_the_same_token_as_all_positions
    prefix = self.class.unconditional[0, 8]
    vocab = self.class.model.config.vocab_size
    full = self.class.model.forward([prefix])
    last = self.class.model.forward([prefix], last_only: true)

    assert_equal([1, prefix.size, vocab], full.shape)
    assert_equal([1, 1, vocab], last.shape)
    from_full = NArrayLLM.scalar(full[0, prefix.size - 1, true].max_index).to_i
    from_last = NArrayLLM.scalar(last.reshape(vocab).max_index).to_i
    assert_equal(from_full, from_last)
    assert_equal(from_full, self.class.unconditional[prefix.size])
  end

  # --- 縮退・エラー ---

  def test_rejects_a_prompt_plus_generation_longer_than_max_seq_len
    limit = self.class.model.config.max_seq_len
    assert_raise(NArrayLLM::Error) do
      self.class.generator.generate([0], max_new_tokens: limit)
    end
  end

  def test_rejects_an_empty_prompt
    assert_raise(NArrayLLM::Error) { self.class.generator.generate([], max_new_tokens: 1) }
  end

  data('recomputing', false)
  data('cached', true)
  def test_yields_each_generated_token(use_cache)
    model = ScriptedModel.new([5, 6, 7], vocab_size: 16, max_seq_len: 32)
    seen = []
    NArrayLLM::Generator.new(model).generate([1], max_new_tokens: 3, cache: use_cache) { |t| seen << t }
    assert_equal([5, 6, 7], seen)
  end

  # --- test doubles ---

  # Returns logits whose argmax is the next scripted token, so the generator's
  # own control flow is what is under test.
  class ScriptedModel
    attr_reader :config, :calls, :last_only_flags, :phases

    def initialize(script, vocab_size:, max_seq_len:)
      @script = script
      @calls = 0
      @last_only_flags = []
      @phases = []
      @config = NArrayLLM::GPT2::Config.new(max_seq_len: max_seq_len, vocab_size: vocab_size,
                                       num_layers: 0, num_heads: 0, channels: 0,
                                       padded_vocab_size: vocab_size)
    end

    def forward(_tokens, last_only: false, prof: nil, cache: nil)
      @last_only_flags << last_only
      @phases << :forward
      next_logits
    end

    def new_cache
      Object.new
    end

    def prefill(_tokens, cache:, prof: nil)
      @phases << :prefill
      next_logits
    end

    def decode(_token_id, _position, cache:, prof: nil)
      @phases << :decode
      next_logits
    end

    private

    def next_logits
      wanted = @script.fetch(@calls)
      @calls += 1
      logits = XM::SFloat.zeros(1, 1, @config.vocab_size)
      logits[0, 0, wanted] = 1.0
      logits
    end
  end

  FakeTokenizer = Struct.new(:eot_token)

  private

  def divergence_report(mine, theirs, here, there)
    index = mine.each_index.find { |i| mine[i] != theirs[i] } || [mine.size, theirs.size].min
    prefix = mine[0, index]
    lines = ["#{here} と #{there} の生成が位置 #{index} で割れた",
             "  #{here}:  #{mine.inspect}",
             "  #{there}: #{theirs.inspect}",
             "  共通接頭辞: #{self.class.tokenizer.decode(prefix).inspect}"]

    lines << "  #{here} の logits 上位 5:  #{format_top(self.class.model.forward([prefix], last_only: true))}"
    other = other_backend_logits(prefix)
    lines << "  #{there} の logits 上位 5: #{other ? format_top(other) : '(取得不可)'}"
    lines.join("\n")
  end

  def format_top(logits, count: 5)
    vocab = self.class.model.config.vocab_size
    values = logits.reshape(vocab).to_a
    values.each_with_index.max_by(count) { |value, _| value }
          .map { |value, id| format('%d(%s)=%.6f', id, self.class.tokenizer.decode(id).inspect, value) }
          .join(', ')
  end

  def other_backend_tokens(prompt_token, count)
    return nil unless cross_backend_available?

    out = run_other_backend(GENERATE_SCRIPT, prompt_token.to_s, count.to_s)
    out.split(',').map { |id| Integer(id) }
  end

  def other_backend_logits(prefix)
    return nil unless cross_backend_available?

    raw = run_other_backend(LOGITS_SCRIPT, prefix.join(','), binmode: true)
    XM::SFloat.from_binary(raw, [self.class.model.config.vocab_size])
  end

  def run_other_backend(script, *args, binmode: false)
    out, err, status = Open3.capture3(
      env_for(!NArrayLLM.gpu?), RbConfig.ruby, '-e', script,
      File.join(TestHelper::DATA_DIR, 'gpt2_124M.bin'),
      File.join(TestHelper::DATA_DIR, 'gpt2_tokenizer.bin'),
      *args, chdir: ROOT, binmode: binmode
    )
    raise "cross-backend subprocess failed: #{err}" unless status.success?

    out
  end

  def cross_backend_available?
    want_gpu = !NArrayLLM.gpu?
    return true unless want_gpu

    _out, _err, status = Open3.capture3(env_for(true), RbConfig.ruby, '-e',
                                        "require 'cumo/narray'; Cumo::SFloat.zeros(1)")
    return true if status.success?

    @omit_reason = 'Cumo (GPU) is not available; skipping the cross-backend comparison'
    false
  end

  def env_for(gpu)
    { 'GPU' => gpu ? '1' : '0', 'NARRAY_LLM_DATA' => TestHelper::DATA_DIR }
  end

  GENERATE_SCRIPT = <<~'RUBY'
    require_relative 'lib/narray_llm'
    model = NArrayLLM::GPT2::Model.load(ARGV[0])
    tokenizer = NArrayLLM::GPT2::Tokenizer.load(ARGV[1])
    generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    print generator.generate([Integer(ARGV[2])], max_new_tokens: Integer(ARGV[3])).join(',')
  RUBY

  LOGITS_SCRIPT = <<~'RUBY'
    require_relative 'lib/narray_llm'
    model = NArrayLLM::GPT2::Model.load(ARGV[0])
    prefix = ARGV[2].split(',').map { |id| Integer(id) }
    $stdout.binmode
    $stdout.write(model.forward([prefix], last_only: true)
                       .reshape(model.config.vocab_size).to_binary)
  RUBY
end
