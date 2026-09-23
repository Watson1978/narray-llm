# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-mamba.md).
#
# The reference is script/mamba_dump.c, which includes mamba.c whole and so
# uses its arithmetic. That binary also checks its own copy of forward()
# against the original and refuses to write a dump unless the two agree
# exactly, so a drifted copy cannot become the expectation here.
class TestMambaForward < Test::Unit::TestCase
  include TestHelper

  MODELS = %w[mamba_tiny mamba-130m].freeze

  # mamba.c publishes no tolerance of its own, so these are set from what was
  # measured here rather than copied from somewhere. The logits reach ~55 and
  # fp32 carries ~7 digits, and the value passes through 24 blocks each of
  # which sums 1536 products, so a few times 1e-4 is the floor.
  #
  # Measured: 9.5e-07 (mamba_tiny) and 4.1e-04 (mamba-130m) on Numo,
  # 1.9e-06 and 4.3e-04 on Cumo. 5e-3 leaves about 12x headroom.
  #
  # It stays far below what argmax needs: over 16 positions the closest the top
  # two logits ever came was 0.0954, which is 230x the worst difference seen.
  LOGITS_TOLERANCE = 5e-3

  # The two recurrences are smaller than the logits and get a tighter bound.
  # Measured: 1.3e-05 (conv) and 7.6e-05 (ssm) on mamba-130m.
  STATE_TOLERANCE = 1e-3

  class << self
    def prepared(name)
      @prepared ||= {}
      @prepared[name] ||= begin
        state = NArrayLLM::Mamba::DebugState.load(
          File.join(TestHelper::DATA_DIR, "#{name}_debug_state.bin")
        )
        model = NArrayLLM::Mamba::Model.load(File.join(TestHelper::DATA_DIR, "#{name}.bin"))
        run(model, state)
      end
    end

    # Mamba carries its state forward, so the whole sequence has to be replayed
    # in order; there is no forwarding a single position on its own.
    def run(model, state)
      mamba_state = model.new_state
      logits = []
      conv = []
      ssm = []
      state.tokens.each do |token|
        logits << model.decode(token, cache: mamba_state).flatten
        conv << Array.new(model.config.num_layers) { |l| mamba_state.conv(l).flatten.clone }
        ssm << Array.new(model.config.num_layers) { |l| mamba_state.ssm(l).flatten.clone }
      end
      { model: model, state: state, logits: logits, conv: conv, ssm: ssm }
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
    vocab = f[:model].config.vocab_size
    f[:state].steps.times do |pos|
      d = NArrayLLM::Compare.diff(f[:state].logits(pos)[0...vocab], f[:logits][pos],
                                  tolerance: LOGITS_TOLERANCE, label: "logits pos=#{pos}")
      assert_true(d.ok?, d.to_s)
      worst = [worst, d.max_abs].max
    end
    notify(format('%s: 最大 max|d| = %.3e (閾値 %.1e)', name, worst, LOGITS_TOLERANCE))
  end

  # --- 受け入れ条件 2: 両方の再帰が閾値内で一致する ---

  # The states are what a KV cache would be, and they are the part a forward
  # pass could get wrong without the first token's logits showing it.
  data(MODELS.to_h { |n| [n, n] })
  def test_recurrent_states_match_the_reference(name)
    f = prepared(name)
    worst = Hash.new(0.0)
    f[:state].steps.times do |pos|
      f[:model].config.num_layers.times do |layer|
        { 'conv_state' => f[:conv], 'ssm_state' => f[:ssm] }.each do |dumped, mine|
          d = NArrayLLM::Compare.diff(f[:state][dumped, pos, layer], mine[pos][layer],
                                      tolerance: STATE_TOLERANCE,
                                      label: "#{dumped} pos=#{pos} layer=#{layer}")
          assert_true(d.ok?, d.to_s)
          worst[dumped] = [worst[dumped], d.max_abs].max
        end
      end
    end
    notify(format('%s: conv %.3e / ssm %.3e (閾値 %.1e)',
                  name, worst['conv_state'], worst['ssm_state'], STATE_TOLERANCE))
  end

  # --- 受け入れ条件 3: 閾値に依存しない検査 ---

  # mamba-130m's dump was produced by feeding mamba.c's own argmax back in, so
  # the reference token at position p+1 is what the logits at position p chose.
  # Matching it is exact: either the same id comes out or it does not.
  def test_argmax_reproduces_the_reference_sequence
    f = prepared('mamba-130m')
    (f[:state].steps - 1).times do |pos|
      chosen = host(f[:logits][pos].max_index)
      assert_equal(f[:state].tokens[pos + 1], chosen, "argmax at pos=#{pos}")
    end
  end

  # mamba.c computes the rounded width because the table is stored that way,
  # then samples over vocab_size. The tail belongs to no token, so decode drops
  # it and the dump keeps it.
  def test_the_padded_tail_is_dropped
    f = prepared('mamba-130m')
    assert_equal(50_280, f[:state].logits(0).size)
    assert_equal(50_277, f[:logits][0].size)
    assert_equal(50_277, f[:model].config.vocab_size)
  end

  # --- 受け入れ条件 4: Numo と Cumo が一致する ---

  # Only meaningful on the GPU: on Numo this compares a run with itself.
  def test_backends_agree
    omit('GPU=1 でのみ意味がある') unless NArrayLLM.gpu?
    f = prepared('mamba_tiny')
    vocab = f[:model].config.vocab_size
    f[:state].steps.times do |pos|
      d = NArrayLLM::Compare.diff(f[:state].logits(pos)[0...vocab], f[:logits][pos],
                                  tolerance: LOGITS_TOLERANCE / 10.0, label: "pos=#{pos}")
      assert_true(d.ok?, d.to_s)
    end
  end
end
