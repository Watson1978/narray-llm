# frozen_string_literal: true

require_relative 'test_helper'
require 'open3'
require 'rbconfig'

# Stage 2 acceptance tests (docs/plans/PLAN-llama2.md).
#
# The gate is that the text comes out byte for byte the same as llama2.c's own
# run.c at temperature 0. For stories260K that string is published in llama2.c's
# test_all.py, so the fixture below is external to this repository.
class TestLlama2Generate < Test::Unit::TestCase
  include TestHelper

  ROOT = File.expand_path('..', __dir__)

  # test_all.py:37 expected_stdout, for
  # `./run stories260K.bin -z tok512.bin -t 0.0 -n 200`. run.c adds one trailing
  # newline for looks (run.c:774); that is not part of this.
  STORIES260K_200 =
    'Once upon a time, there was a little girl named Lily. She loved to play outside in the park. ' \
    "One day, she saw a big, red ball. She wanted to play with it, but it was too high.\n" \
    'Lily\'s mom said, "Lily, let\'s go to the park." Lily was sad and didn\'t know what to do. ' \
    'She said, "I want to play with your ball, but I can\'t find it."' "\n" \
    'Lily was sad and didn\'t know what to do. She said, "I\'m sorry, Lily. I didn\'t know what ' \
    'to do."' "\n" \
    'Lily didn\'t want to help her mom, so she'

  class << self
    def prepared
      @prepared ||= begin
        model = NArrayLLM::Llama2::Model.load(File.join(TestHelper::DATA_DIR, 'stories260K.bin'))
        tokenizer = NArrayLLM::Llama2::Tokenizer.load(
          File.join(TestHelper::DATA_DIR, 'tok512.bin'), vocab_size: model.config.vocab_size
        )
        generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
        tokens = generator.generate([NArrayLLM::Llama2::Tokenizer::BOS_TOKEN],
                                    max_new_tokens: 200, cache: false)
        { model: model, tokenizer: tokenizer, generator: generator, tokens: tokens }
      end
    end
  end

  def prepared
    require_data('stories260K.bin')
    require_data('tok512.bin')
    self.class.prepared
  end

  # --- 受け入れ条件 1: run.c の温度 0 の出力と完全一致する ---

  def test_the_text_matches_run_c_byte_for_byte
    f = prepared
    assert_equal(STORIES260K_200.b, f[:tokenizer].render(f[:tokens]).b)
  end

  def test_two_hundred_tokens_come_out
    f = prepared
    # The prompt is BOS, and nothing stopped it early.
    assert_equal(201, f[:tokens].size)
    assert_equal(NArrayLLM::Llama2::Tokenizer::BOS_TOKEN, f[:tokens].first)
  end

  # --- 受け入れ条件 2: Numo と Cumo が同一のトークン列を出す ---

  def test_backends_produce_an_identical_token_sequence
    f = prepared
    other = other_backend_tokens
    omit(@omit_reason) if other.nil?

    assert_equal(f[:tokens], other, 'greedy decoding is deterministic, so these must be equal')
  end

  # --- 受け入れ条件 3: BOS で停止する ---

  # run.c:763 breaks on BOS, not EOS: in llama2.c it is BOS that delimits
  # sequences. Nothing in stories260K emits it within 200 steps, so the stop is
  # driven here instead of waiting for one.
  def test_generation_stops_on_the_delimiter_token
    f = prepared
    stop = f[:tokenizer].eot_token
    assert_equal(NArrayLLM::Llama2::Tokenizer::BOS_TOKEN, stop)

    model = AlwaysEmits.new(stop, f[:model].config)
    generator = NArrayLLM::Generator.new(model, tokenizer: f[:tokenizer])
    tokens = generator.generate([5], max_new_tokens: 10, cache: false)
    assert_equal([5, stop], tokens, 'the loop must stop at the delimiter, not run to the budget')
  end

  def test_generation_runs_to_the_budget_when_the_stop_is_disabled
    f = prepared
    model = AlwaysEmits.new(f[:tokenizer].eot_token, f[:model].config)
    generator = NArrayLLM::Generator.new(model, tokenizer: f[:tokenizer])
    tokens = generator.generate([5], max_new_tokens: 4, cache: false, stop_at_eot: false)
    assert_equal(5, tokens.size)
  end

  # --- トークナイザ ---

  def test_the_table_accounts_for_the_whole_file
    f = prepared
    assert_equal(File.size(data_path('tok512.bin')), f[:tokenizer].expected_file_size)
  end

  def test_a_byte_token_decodes_to_that_byte
    f = prepared
    # run.c:425 turns "<0xNN>" into the byte it names.
    assert_equal("\n".b, f[:tokenizer].decode_piece(0, 13).b)
    assert_equal('<0x0A>', f[:tokenizer][13])
  end

  # run.c:421, citing llama2.c PR #89.
  def test_a_leading_space_is_dropped_after_bos
    f = prepared
    spaced = (0...f[:tokenizer].vocab_size).find { |id| f[:tokenizer][id].start_with?(' ') }
    omit('no piece in this vocabulary starts with a space') if spaced.nil?

    after_bos = f[:tokenizer].decode_piece(NArrayLLM::Llama2::Tokenizer::BOS_TOKEN, spaced)
    elsewhere = f[:tokenizer].decode_piece(99, spaced)
    assert_equal(f[:tokenizer][spaced].byteslice(1..), after_bos)
    assert_equal(f[:tokenizer][spaced], elsewhere)
  end

  # run.c:431 safe_printf drops a single byte that is neither printable nor
  # whitespace, which is how control codes stay out of the output.
  def test_an_unprintable_single_byte_is_dropped
    f = prepared
    assert_false(f[:tokenizer].printable?("\x01".b))
    assert_true(f[:tokenizer].printable?("\n".b))
    assert_true(f[:tokenizer].printable?('a'))
    assert_false(f[:tokenizer].printable?(''.b))
    # Only single bytes are filtered; a multi-byte piece goes through as it is.
    assert_true(f[:tokenizer].printable?("\x01\x02".b))
  end

  def test_an_out_of_range_token_is_rejected
    f = prepared
    assert_raise(NArrayLLM::Error) { f[:tokenizer][f[:tokenizer].vocab_size] }
    assert_raise(NArrayLLM::Error) { f[:tokenizer][-1] }
  end

  # A model that cannot cache has to say so rather than fail somewhere inside
  # the generator. Llama2::Model gained a cache in stage 3, so the stub below is
  # what stands in for one that has not.
  def test_asking_a_cacheless_model_for_a_cache_reports_that_there_is_none
    f = prepared
    model = AlwaysEmits.new(7, f[:model].config)
    assert_false(model.respond_to?(:new_cache))
    generator = NArrayLLM::Generator.new(model, tokenizer: f[:tokenizer])
    assert_raise(NArrayLLM::Error) { generator.generate([1], max_new_tokens: 1, cache: true) }
  end

  # A stand-in that always points argmax at one token, so the stop condition can
  # be exercised without a model that happens to emit it.
  class AlwaysEmits
    attr_reader :config

    def initialize(token, config)
      @token = token
      @config = config
    end

    def forward(_tokens, last_only: false, prof: nil)
      logits = XM::SFloat.zeros(1, @config.vocab_size)
      logits[0, @token] = 1.0
      logits
    end
  end

  private

  def other_backend_tokens
    want_gpu = !NArrayLLM.gpu?
    unless backend_available?(want_gpu)
      @omit_reason = "#{want_gpu ? 'Cumo (GPU)' : 'Numo (CPU)'} is not available; " \
                     'skipping the cross-backend comparison'
      return nil
    end

    out, err, status = Open3.capture3(
      env_for(want_gpu), RbConfig.ruby, '-e', DUMP_TOKENS_SCRIPT,
      File.join(TestHelper::DATA_DIR, 'stories260K.bin'),
      File.join(TestHelper::DATA_DIR, 'tok512.bin'),
      chdir: ROOT
    )
    raise "cross-backend subprocess failed: #{err}" unless status.success?

    out.split.map { |id| Integer(id) }
  end

  def backend_available?(want_gpu)
    return true unless want_gpu

    _out, _err, status = Open3.capture3(env_for(true), RbConfig.ruby, '-e',
                                        "require 'cumo/narray'; Cumo::SFloat.zeros(1)")
    status.success?
  end

  def env_for(gpu)
    { 'GPU' => gpu ? '1' : '0', 'NARRAY_LLM_DATA' => TestHelper::DATA_DIR }
  end

  DUMP_TOKENS_SCRIPT = <<~'RUBY'
    require_relative 'lib/narray_llm'
    model = NArrayLLM::Llama2::Model.load(ARGV[0])
    tokenizer = NArrayLLM::Llama2::Tokenizer.load(ARGV[1], vocab_size: model.config.vocab_size)
    generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    tokens = generator.generate([1], max_new_tokens: 200, cache: false)
    puts tokens.join(' ')
  RUBY
end
