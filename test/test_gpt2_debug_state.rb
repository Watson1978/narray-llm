# frozen_string_literal: true

require_relative 'test_helper'

# The debug state format is part of stage 0 (docs/plans/PLAN-gpt2.md says to pin it down by
# reading test_gpt2.c). Its fixtures come from the same doc.
class TestDebugState < Test::Unit::TestCase
  include TestHelper

  BATCH_SIZE = 4
  SEQ_LEN = 64
  VOCAB_SIZE = 50_257

  # 1024 header + x + y + logits + loss + grads
  STATE_BYTES = 1024 +
                4 * BATCH_SIZE * SEQ_LEN +
                4 * BATCH_SIZE * SEQ_LEN +
                4 * BATCH_SIZE * SEQ_LEN * VOCAB_SIZE +
                4 +
                4 * 124_475_904

  # test_gpt2.c:89-90 expected_losses[0], which :141 checks the file's own
  # expected_loss against with a 1e-2 tolerance.
  REFERENCE_LOSS = 5.270007133483887
  LOSS_TOLERANCE = 1e-2

  def checkpoint
    @checkpoint ||= NArrayLLM::GPT2::Checkpoint.load(require_data('gpt2_124M.bin'))
  end

  def state
    @state ||= NArrayLLM::GPT2::DebugState.load(require_data('gpt2_124M_debug_state.bin'),
                                           config: checkpoint.config)
  end

  def test_header_matches_test_gpt2_c
    assert_equal(NArrayLLM::GPT2::DebugState::MAGIC, state.header[0], 'magic')
    assert_equal(NArrayLLM::GPT2::DebugState::VERSION, state.header[1], 'version')
    assert_equal(BATCH_SIZE, state.batch_size, 'B')
    assert_equal(SEQ_LEN, state.seq_len, 'T')
  end

  def test_section_layout_covers_the_whole_file
    path = require_data('gpt2_124M_debug_state.bin')
    assert_equal(STATE_BYTES, File.size(path), 'file size')
    assert_equal(File.size(path), state.expected_file_size)
  end

  def test_inputs_and_targets_are_token_ids
    assert_equal([BATCH_SIZE, SEQ_LEN], state.inputs.shape)
    assert_equal([BATCH_SIZE, SEQ_LEN], state.targets.shape)
    assert_operator(state.inputs.min, :>=, 0)
    assert_operator(state.inputs.max, :<, VOCAB_SIZE)
    assert_operator(state.targets.min, :>=, 0)
    assert_operator(state.targets.max, :<, VOCAB_SIZE)
    # y is x shifted by one within each row (train_gpt2.py builds it that way).
    assert_equal(state.inputs[true, 1...SEQ_LEN].to_a, state.targets[true, 0...(SEQ_LEN - 1)].to_a)
  end

  # expected_logits is V wide, NOT Vp. If it were read as Vp the loss that
  # follows it would land on garbage, so this pins the whole section layout.
  def test_logits_are_v_wide_and_loss_matches_pytorch
    assert_equal([BATCH_SIZE, SEQ_LEN, VOCAB_SIZE], state.logits.shape)
    assert_in_delta(REFERENCE_LOSS, state.loss, LOSS_TOLERANCE)
    assert_equal(false, NArrayLLM.scalar(state.logits.sum).nan?, 'no NaN in reference logits')
  end

  def test_logits_agree_with_an_independent_raw_read
    path = require_data('gpt2_124M_debug_state.bin')
    offset = 1024 + 2 * 4 * BATCH_SIZE * SEQ_LEN
    count = BATCH_SIZE * SEQ_LEN * VOCAB_SIZE
    flat = state.logits.flatten
    assert_bit_identical(raw_fp32(path, offset, 5), flat[0...5], 'head of logits')
    assert_bit_identical(raw_fp32(path, offset + 4 * (count - 5), 5), flat[(count - 5)...count],
                         'tail of logits')
  end

  # --- reference gradients (docs/plans/PLAN-training.md, stage 0) ---

  # Counted arithmetically rather than with a Bit mask (AGENTS.md).
  def nonzero_rows(tensor)
    host(tensor.abs.sum(axis: 1).clip(0.0, 1.0).ceil.sum).to_i
  end

  def test_every_gradient_has_the_shape_the_parameter_has
    shapes = NArrayLLM::GPT2::Checkpoint.tensor_shapes(checkpoint.config)
    state.grad_names.each do |name|
      assert_equal(shapes.fetch(name), state.grad(name).shape, name.to_s)
    end
  end

  def test_the_gradients_use_the_file_to_its_end
    total = state.grad_names.sum { |name| state.grad(name).size }
    assert_equal(NArrayLLM::GPT2::Checkpoint.num_parameters(checkpoint.config), total)
    assert_equal(File.size(data_path('gpt2_124M_debug_state.bin')),
                 state.grads_byte_offset + (4 * total))
  end

  # The size check above would pass on a permutation of the tensors, so the
  # alignment is pinned by shape instead. Only the T positions the batch used
  # can have a position gradient, and only the real vocabulary can have a token
  # gradient -- llm.c stops softmax and cross-entropy at V (train_gpt2.c).
  def test_the_gradients_line_up_with_the_parameters
    assert_equal(SEQ_LEN, nonzero_rows(state.grad(:wpe)), 'dwpe rows past T must be zero')

    wte = state.grad(:wte)
    assert_equal(VOCAB_SIZE, nonzero_rows(wte), 'dwte rows past V must be zero')
    # Not vacuous: the table is wider than the vocabulary here.
    assert_operator(wte.shape[0], :>, VOCAB_SIZE)
  end

  # wte is the classifier too, so every vocabulary row takes a gradient from the
  # softmax even though the batch only holds 135 distinct tokens.
  def test_the_tied_classifier_reaches_every_vocabulary_row
    assert_operator(state.inputs.to_a.flatten.uniq.size, :<, 1000)
    assert_equal(VOCAB_SIZE, nonzero_rows(state.grad(:wte)))
  end
end
