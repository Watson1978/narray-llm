# frozen_string_literal: true

require_relative 'test_helper'

# Stage 3 acceptance tests (docs/plans/PLAN-conv2d.md).
#
# The gate is the class number: all sixteen have to be the ones transformers
# answers, for both spellings and with the batch norm folded or not. The true
# label is not the gate, and two of the sixteen are not the true label.
class TestResNetModel < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/resnet-18/model.safetensors', __dir__)
  STATE = File.expand_path('../data/resnet-18_state.safetensors', __dir__)
  FIXTURE = File.expand_path('../python/fixtures/resnet-18_classes.json', __dir__)

  # Twenty convolutions deep, over logits that reach 12. The worst seen is
  # 1.8e-05.
  TOLERANCE = 1.0e-4

  def setup
    omit("#{MODEL} not found") unless File.exist?(MODEL)
    omit("#{STATE} not found; run `rake prepare:resnet`") unless File.exist?(STATE)
    omit("#{FIXTURE} not found; run python/resnet_fixtures.py") unless File.exist?(FIXTURE)

    @checkpoint = NArrayLLM::ResNet::Checkpoint.load(MODEL)
    @state = NArrayLLM::Safetensors.new(STATE)
    @want = JSON.parse(File.read(FIXTURE))['images']
  end

  def teardown
    @checkpoint&.close
    @state&.close
  end

  def model(spelling:, fold:)
    NArrayLLM::ResNet::Model.new(@checkpoint, spelling: spelling, fold: fold)
  end

  def test_every_class_number_is_the_one_the_reference_answers
    pixels = @state['pixel_values']
    expected = @want.map { |row| row['class'] }

    [[:shift, false], [:unfold, false], [:shift, true], [:unfold, true]].each do |spelling, fold|
      got = model(spelling: spelling, fold: fold).classify(pixels)
      assert_equal(expected, got, "#{spelling} fold=#{fold}")
    end
  end

  def test_the_logits_stay_close_to_the_reference
    pixels = @state['pixel_values']
    expected = @state['logits']

    [[:shift, false], [:unfold, true]].each do |spelling, fold|
      logits = model(spelling: spelling, fold: fold).forward(pixels)
      assert_equal(expected.shape, logits.shape, "#{spelling} fold=#{fold}")
      worst = host((logits - expected).abs.max).to_f
      assert_operator(worst, :<, TOLERANCE, "#{spelling} fold=#{fold}: max|d| #{worst}")
    end
  end

  # Folding is exact arithmetic but not the same rounding, so it is allowed to
  # move the logits. It is not allowed to move a class number.
  def test_folding_moves_the_logits_and_not_the_answer
    pixels = @state['pixel_values']
    plain = model(spelling: :unfold, fold: false)
    folded = model(spelling: :unfold, fold: true)

    assert_equal(plain.classify(pixels), folded.classify(pixels))
    worst = host((plain.forward(pixels) - folded.forward(pixels)).abs.max).to_f
    assert_operator(worst, :>, 0.0, 'folding changed nothing at all, which it should')
    assert_operator(worst, :<, TOLERANCE, "max|d| #{worst}")
  end

  # A batch of one takes the same path as a batch of sixteen, and the answer
  # for an image must not depend on what it was batched with.
  def test_one_image_answers_what_it_did_in_the_batch
    pixels = @state['pixel_values']
    expected = @want.map { |row| row['class'] }
    subject = model(spelling: :unfold, fold: false)

    [0, 4].each do |i|
      one = NArrayLLM::Ops.contiguous(pixels[i...(i + 1), true, true, true])
      assert_equal([expected[i]], subject.classify(one), "image #{i}")
    end
  end
end
