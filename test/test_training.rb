# frozen_string_literal: true

require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-training.md), against llm.c's own gate.
#
# The debug state carries the gradient llm.c computes for every one of the
# 124,475,904 parameters, and test_gpt2.c:151-166 compares them with an absolute
# tolerance of 2e-2. That tolerance is borrowed rather than invented: it is what
# llm.c itself accepts between its C forward and the PyTorch run that produced
# the file.
class TestTraining < Test::Unit::TestCase
  include TestHelper

  # test_gpt2.c:20, check_tensor.
  GRADIENT_TOLERANCE = 2e-2
  # test_gpt2.c:141, against the file's own expected_loss.
  LOSS_TOLERANCE = 1e-2

  def checkpoint
    @checkpoint ||= NArrayLLM::GPT2::Checkpoint.load(require_data('gpt2_124M.bin'))
  end

  def state
    @state ||= NArrayLLM::GPT2::DebugState.load(require_data('gpt2_124M_debug_state.bin'),
                                                config: checkpoint.config)
  end

  def trained
    @trained ||= begin
      model = NArrayLLM::GPT2::Model.new(checkpoint)
      loss, acts = model.forward_train(state.inputs, state.targets)
      [model, loss, acts]
    end
  end

  def test_the_training_forward_reaches_the_reference_loss
    _model, loss, = trained
    assert_in_delta(state.loss, loss, LOSS_TOLERANCE)
    notify(format('loss %.6f (参照 %.6f、差 %.2e)', loss, state.loss, (loss - state.loss).abs))
  end

  def test_every_gradient_matches_llm_c
    model, _loss, acts = trained
    grads = model.backward(acts)
    worst = {}
    state.grad_names.each do |name|
      want = state.grad(name)
      # The model drops the padded vocabulary rows at load, so only the real
      # rows exist here. llm.c checks V*C of them too (test_gpt2.c:151).
      want = want[0...checkpoint.config.vocab_size, true] if name == :wte
      worst[name] = host((want - grads.fetch(name)).abs.max).to_f
      assert_operator(worst[name], :<, GRADIENT_TOLERANCE, "d#{name}")
    end
    notify(format('16 テンソルの最大 max|d| = %.3e (%s, 閾値 %.0e)',
                  worst.values.max, worst.max_by { |_, v| v }.first, GRADIENT_TOLERANCE))
  end

  # A backward that dropped a layer would still pass every tensor whose
  # gradient is dominated by one place, so the shapes are checked too.
  def test_the_gradients_have_the_shapes_the_parameters_have
    model, _loss, acts = trained
    grads = model.backward(acts)
    shapes = NArrayLLM::GPT2::Checkpoint.tensor_shapes(checkpoint.config)
    shapes.each do |name, shape|
      shape = [checkpoint.config.vocab_size, shape[1]] if name == :wte
      assert_equal(shape, grads.fetch(name).shape, name.to_s)
    end
  end

  # --- stage 3: AdamW (docs/plans/PLAN-training.md) ---

  # test_gpt2.c:89-99. Ten steps of AdamW at the settings test_gpt2.c:172
  # passes, checked with the tolerance test_gpt2.c:141 uses.
  EXPECTED_LOSSES = [
    5.270007133483887, 4.059706687927246, 3.3751230239868164, 2.8007826805114746,
    2.315382242202759, 1.8490285873413086, 1.3946564197540283, 0.9991465210914612,
    0.6240804195404053, 0.37651097774505615
  ].freeze

  # The whole point of this one: forward, backward and the optimiser all have to
  # be right together, because any of the three drifts the sequence. It costs
  # about 80 seconds on Numo and 2 on Cumo, which is the price of the only gate
  # that covers all three at once.
  def test_ten_steps_follow_llm_c_loss_for_loss
    model = NArrayLLM::GPT2::Model.new(checkpoint)
    optimiser = NArrayLLM::AdamW.new(model)
    worst = 0.0

    EXPECTED_LOSSES.each_with_index do |want, step|
      loss, acts = model.forward_train(state.inputs, state.targets)
      assert_in_delta(want, loss, LOSS_TOLERANCE, "step #{step}")
      worst = [worst, (loss - want).abs].max
      optimiser.step(model.backward(acts))
    end

    assert_equal(EXPECTED_LOSSES.size, optimiser.steps)
    notify(format('10 ステップの最大 |d| = %.2e (閾値 %.0e), AdamW の状態 %.1f MiB',
                  worst, LOSS_TOLERANCE, optimiser.bytes / 1048576.0))
  end

  # Not vacuous: without an optimiser the loss would sit at the first value, so
  # the sequence has to actually move.
  def test_the_loss_sequence_moves
    assert_operator(EXPECTED_LOSSES.first / EXPECTED_LOSSES.last, :>, 10)
  end
end
