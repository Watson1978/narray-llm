# frozen_string_literal: true

require_relative 'test_helper'
require 'open3'
require 'rbconfig'

# Stage 1 acceptance tests (docs/plans/PLAN-llama2.md).
#
# The reference is data/<model>_debug_state.bin, written by script/llama2_dump.rb.
# That generator includes run.c whole and checks its copy of forward() against
# the original on every step, so what is compared here is llama2.c's arithmetic.
class TestLlama2Forward < Test::Unit::TestCase
  include TestHelper

  ROOT = File.expand_path('..', __dir__)
  MODELS = %w[stories260K stories110M].freeze

  # llama2.c publishes no tolerance of its own, so this is set from what fp32
  # rounding can produce here rather than copied from somewhere. Logits reach
  # ~20 in magnitude and fp32 carries ~7 digits, and the value accumulates
  # through 12 blocks, so a few times 1e-5 is the floor. Measured: 1.0e-05
  # (stories260K) and 2.3e-05 (stories110M). 1e-3 leaves ~40x headroom and is
  # still far below the gap between competing logits, which is what argmax needs.
  LOGITS_TOLERANCE = 1e-3

  # Intermediates are smaller than logits, so they get a tighter bound.
  ACTIVATION_TOLERANCE = 1e-4

  PER_LAYER = %w[rms_att q_pre_rope k_pre_rope v q k attn_out attproj res_att
                 rms_ffn w1h w3h swiglu ffn_out res_ffn].freeze

  class << self
    def prepared(name)
      @prepared ||= {}
      @prepared[name] ||= begin
        model = NArrayLLM::Llama2::Model.load(File.join(TestHelper::DATA_DIR, "#{name}.bin"))
        state = NArrayLLM::Llama2::DebugState.load(
          File.join(TestHelper::DATA_DIR, "#{name}_debug_state.bin")
        )
        trace = {}
        logits = model.forward(state.tokens, trace: trace)
        { model: model, state: state, logits: logits, trace: trace }
      end
    end
  end

  def prepared(name)
    require_data("#{name}.bin")
    require_data("#{name}_debug_state.bin")
    self.class.prepared(name)
  end

  # --- 受け入れ条件 1: 参照 logits と閾値内で一致する ---

  data(MODELS.to_h { |n| [n, n] })
  def test_logits_match_the_reference(name)
    f = prepared(name)
    worst = 0.0
    f[:state].steps.times do |pos|
      d = NArrayLLM::Compare.diff(f[:state].logits(pos), f[:logits][pos, true],
                                  tolerance: LOGITS_TOLERANCE, label: "logits pos=#{pos}")
      assert_true(d.ok?, d.to_s)
      worst = [worst, d.max_abs].max
    end
    notify(format('%s: 最大 max|d| = %.3e (閾値 %.1e)', name, worst, LOGITS_TOLERANCE))
  end

  # The check that does not depend on a tolerance at all: run.c fed its own
  # argmax back in, so the reference token at position p+1 is what the logits at
  # position p chose.
  data(MODELS.to_h { |n| [n, n] })
  def test_the_greedy_token_at_every_position_matches(name)
    f = prepared(name)
    expected = f[:state].tokens[1..]
    actual = (0...(f[:state].steps - 1)).map do |pos|
      NArrayLLM.scalar(f[:logits][pos, true].max_index)
    end
    assert_equal(expected, actual, 'greedy token sequence')
  end

  # --- 受け入れ条件: どの段で最初にずれるかを押さえる ---

  def test_every_intermediate_matches_the_reference
    f = prepared('stories260K')
    state = f[:state]
    config = state.config
    state.steps.times do |pos|
      assert_close(state['embed', pos], f[:trace]['embed'][pos, true], "embed pos=#{pos}")
      config.num_layers.times do |layer|
        PER_LAYER.each do |name|
          got = f[:trace]["L#{layer}/#{name}"][pos, true]
          assert_close(state[name, pos, layer], got, "#{name} pos=#{pos} layer=#{layer}")
        end
        # The reference packs att down to the live pos+1 entries per head; this
        # side keeps the full masked row.
        expected = state['att', pos, layer].reshape(config.num_heads, pos + 1)
        assert_close(expected, f[:trace]["L#{layer}/att"][true, pos, 0..pos],
                     "att pos=#{pos} layer=#{layer}")
      end
      assert_close(state['rms_final', pos], f[:trace]['rms_final'][pos, true], "rms_final pos=#{pos}")
    end
  end

  # The embedding is a row copy, so the one-hot GEMM has to reproduce it exactly.
  def test_the_embedding_is_exact
    f = prepared('stories260K')
    f[:state].steps.times do |pos|
      d = NArrayLLM::Compare.diff(f[:state]['embed', pos], f[:trace]['embed'][pos, true])
      assert_equal(0.0, d.max_abs, "embed pos=#{pos} is not exact")
    end
  end

  # --- 受け入れ条件 2: Cumo と Numo が閾値の一桁下で一致する ---

  def test_backends_agree_within_the_sum_of_their_reference_errors
    f = prepared('stories260K')
    other = other_backend_logits
    omit(@omit_reason) if other.nil?

    reference = XM::SFloat.cast((0...f[:state].steps).map { |p| f[:state].logits(p).to_a })
    mine = NArrayLLM::Compare.diff(reference, f[:logits], label: 'mine vs reference')
    theirs = NArrayLLM::Compare.diff(reference, other, label: 'theirs vs reference')
    cross = NArrayLLM::Compare.diff(other, f[:logits],
                                    tolerance: mine.max_abs + theirs.max_abs, label: 'cumo vs numo')
    notify(format('%s %.3e / %s %.3e (参照との差) -> 相互差 %.3e, 上限 %.3e',
                  NArrayLLM.gpu? ? 'Cumo' : 'Numo', mine.max_abs,
                  NArrayLLM.gpu? ? 'Numo' : 'Cumo', theirs.max_abs,
                  cross.max_abs, mine.max_abs + theirs.max_abs))

    assert_true(cross.ok?, cross.to_s)
    assert_operator(cross.max_abs, :<, LOGITS_TOLERANCE / 10.0)
  end

  # --- 受け入れ条件 3: RoPE 単体 ---

  # Getting the rotation backwards (x0 sin + x1 cos instead of x0 cos - x1 sin)
  # barely moves the final error, so it is pinned here instead.
  def test_rope_rotates_each_pair_by_the_position_angle
    head_size = 4
    cos_t, sin_t = NArrayLLM::Ops.rope_tables(3, head_size)
    x = XM::SFloat[[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
    got = NArrayLLM::Ops.rope(x, cos_t, sin_t, num_heads: 1)

    (0...3).each do |pos|
      (0...2).each do |pair|
        freq = NArrayLLM::Ops::ROPE_THETA**(-2.0 * pair / head_size)
        c = Math.cos(pos * freq)
        s = Math.sin(pos * freq)
        x0 = NArrayLLM.scalar(x[pos, 2 * pair])
        x1 = NArrayLLM.scalar(x[pos, 2 * pair + 1])
        assert_in_delta(x0 * c - x1 * s, NArrayLLM.scalar(got[pos, 2 * pair]), 1e-6,
                        "even element, pos=#{pos} pair=#{pair}")
        assert_in_delta(x0 * s + x1 * c, NArrayLLM.scalar(got[pos, 2 * pair + 1]), 1e-6,
                        "odd element, pos=#{pos} pair=#{pair}")
      end
    end
  end

  def test_rope_at_position_zero_is_the_identity
    cos_t, sin_t = NArrayLLM::Ops.rope_tables(1, 8)
    x = XM::SFloat.new(1, 16).seq / 3.0
    got = NArrayLLM::Ops.rope(x, cos_t, sin_t, num_heads: 2)
    assert_bit_identical(x.flatten.to_a, got.flatten, 'position 0 rotates by 0 radians')
  end

  def test_rope_uses_the_same_angles_in_every_head
    cos_t, sin_t = NArrayLLM::Ops.rope_tables(2, 4)
    one = XM::SFloat[[1.0, 2.0, 3.0, 4.0]]
    two = XM::SFloat[[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]]
    single = NArrayLLM::Ops.rope(one, cos_t[1...2, true], sin_t[1...2, true], num_heads: 1)
    double = NArrayLLM::Ops.rope(two, cos_t[1...2, true], sin_t[1...2, true], num_heads: 2)
    assert_bit_identical(single.flatten.to_a + single.flatten.to_a, double.flatten,
                         'the frequency resets at each head boundary')
  end

  # --- 受け入れ条件 4: RMSNorm 単体 ---

  # The whole point of rmsnorm is that it does not centre, so a row with a
  # non-zero mean has to come out different from layernorm.
  def test_rmsnorm_does_not_centre_the_row
    x = XM::SFloat[[1.0, 2.0, 3.0, 4.0]]
    weight = XM::SFloat.ones(4)
    rms = NArrayLLM::Ops.rmsnorm(x, weight)
    ln = NArrayLLM::Ops.layernorm(x, weight, XM::SFloat.zeros(4))

    scale = Math.sqrt((1 + 4 + 9 + 16) / 4.0 + NArrayLLM::Ops::RMSNORM_EPS)
    [1.0, 2.0, 3.0, 4.0].each_with_index do |v, i|
      assert_in_delta(v / scale, NArrayLLM.scalar(rms[0, i]), 1e-6, "element #{i}")
    end
    assert_operator(NArrayLLM.scalar(rms[0, 0]), :>, 0.0, 'rmsnorm keeps the sign')
    assert_operator(NArrayLLM.scalar(ln[0, 0]), :<, 0.0, 'layernorm centres, so the first goes negative')
  end

  def test_rmsnorm_and_layernorm_agree_once_the_mean_is_zero
    x = XM::SFloat[[-3.0, -1.0, 1.0, 3.0]]
    weight = XM::SFloat.ones(4)
    rms = NArrayLLM::Ops.rmsnorm(x, weight)
    ln = NArrayLLM::Ops.layernorm(x, weight, XM::SFloat.zeros(4))
    d = NArrayLLM::Compare.diff(rms, ln, tolerance: 1e-6, label: 'rmsnorm vs layernorm')
    assert_true(d.ok?, d.to_s)
  end

  # --- 受け入れ条件 5: GQA 単体 ---

  def test_each_key_value_head_serves_kv_mul_query_heads
    kv_heads = 3
    kv_mul = 2
    head_size = 4
    x = XM::SFloat.new(2, kv_heads * head_size).seq
    got = NArrayLLM::Ops.repeat_kv_heads(x, num_kv_heads: kv_heads, kv_mul: kv_mul)

    assert_equal([2, kv_heads * kv_mul * head_size], got.shape)
    (0...2).each do |t|
      (0...(kv_heads * kv_mul)).each do |h|
        # run.c:295: query head h reads key/value head h / kv_mul.
        source = h / kv_mul
        expected = (0...head_size).map { |i| NArrayLLM.scalar(x[t, source * head_size + i]) }
        actual = got[t, (h * head_size)...((h + 1) * head_size)]
        assert_bit_identical(expected, actual, "query head #{h} must read kv head #{source}")
      end
    end
  end

  def test_repeat_is_a_no_op_for_multi_head_attention
    x = XM::SFloat.new(2, 8).seq
    assert_same(x, NArrayLLM::Ops.repeat_kv_heads(x, num_kv_heads: 2, kv_mul: 1))
  end

  def test_the_grouped_model_actually_groups
    f = prepared('stories260K')
    assert_true(f[:model].config.grouped_query?, 'stories260K is the multiquery case')
    assert_equal(2, f[:model].config.kv_mul)
  end

  def test_rejects_a_sequence_longer_than_max_seq_len
    f = prepared('stories260K')
    too_long = Array.new(f[:model].config.max_seq_len + 1, 1)
    assert_raise(NArrayLLM::Error) { f[:model].forward(too_long) }
  end

  private

  def assert_close(expected, actual, label)
    d = NArrayLLM::Compare.diff(expected.reshape(*actual.shape), actual,
                                tolerance: ACTIVATION_TOLERANCE, label: label)
    assert_true(d.ok?, d.to_s)
  end

  # There is only one XM constant per process, so the other backend runs in a
  # subprocess (same approach as test_gpt2_forward.rb).
  def other_backend_logits
    want_gpu = !NArrayLLM.gpu?
    unless backend_available?(want_gpu)
      @omit_reason = "#{want_gpu ? 'Cumo (GPU)' : 'Numo (CPU)'} is not available; " \
                     'skipping the cross-backend comparison'
      return nil
    end

    out, err, status = Open3.capture3(
      env_for(want_gpu), RbConfig.ruby, '-e', DUMP_LOGITS_SCRIPT,
      File.join(TestHelper::DATA_DIR, 'stories260K.bin'),
      File.join(TestHelper::DATA_DIR, 'stories260K_debug_state.bin'),
      chdir: ROOT, binmode: true
    )
    raise "cross-backend subprocess failed: #{err}" unless status.success?

    XM::SFloat.from_binary(out, self.class.prepared('stories260K')[:logits].shape)
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

  DUMP_LOGITS_SCRIPT = <<~'RUBY'
    require_relative 'lib/narray_llm'
    model = NArrayLLM::Llama2::Model.load(ARGV[0])
    state = NArrayLLM::Llama2::DebugState.load(ARGV[1])
    $stdout.binmode
    $stdout.write(model.forward(state.tokens).to_binary)
  RUBY
end
