# frozen_string_literal: true

require_relative 'test_helper'
require 'open3'
require 'rbconfig'

# Acceptance tests for the int8 (Q8_0) path, against llama2.c's runq.c.
#
# The fixtures below come from runq.c itself, built from vendor/llama2.c and run
# on data/stories15M_q80.bin. Nothing here is this implementation's own output
# recorded back.
#
# stories15M_q80.bin is produced rather than downloaded; docs/results/llama2-int8.md
# has the two commands.
class TestLlama2Quantized < Test::Unit::TestCase
  include TestHelper

  ROOT = File.expand_path('..', __dir__)
  MODEL = 'stories15M_q80.bin'
  TOKENS = 24

  # runq.c's logits[0...8] after forward(BOS, 0). fp32, so these are exact.
  RUNQ_LOGITS = [-6.80550957, 0.81357491, -6.81164885, -6.81182241,
                 -6.81171274, -6.81175613, -6.81182671, -6.81175613].freeze

  # Greedy ids from the same binary, BOS followed by 24 steps.
  RUNQ_IDS = [1, 9038, 2501, 263, 931, 29892, 727, 471, 263, 2217, 7826, 4257,
              365, 2354, 29889, 2296, 18012, 304, 1708, 5377, 297, 278, 6575,
              845, 457].freeze

  # export.py reports the largest quantization error over all of this model's
  # weights as 0.0023326119. Nothing may exceed what it measured.
  MAX_QUANTIZATION_ERROR = 0.0024

  def quantized
    require_data(MODEL)
    @quantized ||= NArrayLLM::Llama2::QuantizedCheckpoint.load(data_path(MODEL))
  end

  def model
    require_data(MODEL)
    self.class.model ||= NArrayLLM::Llama2::QuantizedModel.load(data_path(MODEL))
  end

  class << self
    attr_accessor :model
  end

  # --- 受け入れ条件 1: ヘッダが runq.c の読むとおりである ---

  def test_the_header_says_what_runq_c_expects
    c = quantized.config
    assert_equal(288, c.dim)
    assert_equal(768, c.hidden_dim)
    assert_equal(6, c.num_layers)
    assert_equal(6, c.num_heads)
    assert_equal(6, c.num_kv_heads)
    assert_equal(32_000, c.vocab_size)
    assert_equal(256, c.max_seq_len)
    assert(c.shared_classifier, 'stories15M shares its classifier with the embedding')
  end

  # export.py halves the group size until it divides dim, and 288 is not a
  # multiple of 64. Reading it from the header rather than assuming 64 is the
  # whole point of this one.
  def test_the_group_size_comes_from_the_file
    assert_equal(32, quantized.group_size)
  end

  def test_the_classifier_is_the_embedding_itself
    assert_same(quantized[:q_tokens], quantized[:wcls])
  end

  # --- 受け入れ条件 2: 脱量子化が fp32 の重みと一致する ---

  def test_dequantized_weights_stay_within_the_reported_error
    require_data('stories15M.bin')
    fp32 = NArrayLLM::Llama2::Checkpoint.load(data_path('stories15M.bin'))

    { token_embedding_table: quantized[:q_tokens],
      wq: quantized[:wq][0],
      w2: quantized[:w2][3],
      wo: quantized[:wo][5] }.each do |name, q|
      reference = fp32[name]
      reference = reference[{ wq: 0, w2: 3, wo: 5 }[name], true, true] unless name == :token_embedding_table
      difference = host((reference - dequantize(q)).abs.max)
      assert_operator(difference, :<=, MAX_QUANTIZATION_ERROR, "#{name} drifted further than export.py measured")
    end
  end

  def test_the_rmsnorm_weights_are_not_quantized_at_all
    require_data('stories15M.bin')
    fp32 = NArrayLLM::Llama2::Checkpoint.load(data_path('stories15M.bin'))

    %i[rms_att_weight rms_ffn_weight rms_final_weight].each do |name|
      assert_equal(0.0, host((fp32[name] - quantized[name]).abs.max), "#{name} is stored in fp32")
    end
  end

  # --- 受け入れ条件 3: runq.c と数値が一致する ---

  # Only where the arithmetic is this side's. Cumo folds rmsnorm into a kernel
  # whose order of operations it chooses itself, and that lands 2.4e-06 away
  # (docs/results/llama2-int8.md).
  def test_the_first_logits_match_runq_c_bit_for_bit
    omit('the fused rmsnorm decides the last bits, not this code') if NArrayLLM::Ops::FUSED_RMSNORM

    logits = model.decode(1, 0, cache: model.new_cache)
    assert_bit_identical(RUNQ_LOGITS, logits.flatten[0...RUNQ_LOGITS.size])
  end

  # What both backends owe: close enough that a last bit is all that separates
  # them from the reference. 1e-5 is about 80 ulp at this magnitude, and the
  # fused path measured 2.4e-06.
  def test_the_first_logits_are_a_few_ulp_from_runq_c
    logits = host(model.decode(1, 0, cache: model.new_cache).flatten[0...RUNQ_LOGITS.size])
    RUNQ_LOGITS.each_with_index do |want, i|
      assert_in_delta(want, host(logits[i]), 1e-5, "logit #{i}")
    end
  end

  def test_the_first_tokens_match_runq_c
    tokenizer = NArrayLLM::Llama2::Tokenizer.load(data_path('tokenizer.bin'),
                                                  vocab_size: model.config.vocab_size)
    generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    tokens = generator.generate([NArrayLLM::Llama2::Tokenizer::BOS_TOKEN],
                                max_new_tokens: TOKENS, cache: true, stop_at_eot: false)
    assert_equal(RUNQ_IDS, tokens)
  end

  # --- 受け入れ条件 4: 2 つのバックエンドが一致する ---
  #
  # Only over this prefix. The two diverge later: Cumo folds rmsnorm and softmax
  # in kernels whose arithmetic this side does not choose, and quantization
  # makes the trajectory sensitive enough for a last bit to change a token
  # (docs/results/llama2-int8.md).

  def test_backends_agree_over_the_prefix
    require_data(MODEL)
    other = other_backend_tokens
    omit(@omit_reason) if other.nil?

    assert_equal(RUNQ_IDS, other)
  end

  private

  def dequantize(weight)
    groups = weight.scales.shape[1]
    (XF.cast(weight.q) * weight.scales.reshape(weight.shape[0], groups, 1)).reshape(*weight.shape)
  end

  def other_backend_tokens
    want_gpu = !NArrayLLM.gpu?
    unless backend_available?(want_gpu)
      @omit_reason = "#{want_gpu ? 'Cumo (GPU)' : 'Numo (CPU)'} is not available; " \
                     'skipping the cross-backend comparison'
      return nil
    end

    out, err, status = Open3.capture3(
      env_for(want_gpu), RbConfig.ruby, '-e', DUMP_TOKENS_SCRIPT,
      data_path(MODEL), data_path('tokenizer.bin'), TOKENS.to_s, chdir: ROOT
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
    model = NArrayLLM::Llama2::QuantizedModel.load(ARGV[0])
    tokenizer = NArrayLLM::Llama2::Tokenizer.load(ARGV[1], vocab_size: model.config.vocab_size)
    generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
    tokens = generator.generate([1], max_new_tokens: Integer(ARGV[2]), cache: true, stop_at_eot: false)
    puts tokens.join(' ')
  RUBY
end
