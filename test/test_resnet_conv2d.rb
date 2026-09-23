# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-conv2d.md).
#
# The reference is transformers, taken with forward hooks on the real model by
# python/resnet_dump.py. Three convolutions are checked rather than one,
# because the model has three kernel shapes and each has its own index
# arithmetic: the 7x7 stride 2 stem, a 3x3 stride 1, and the 1x1 stride 2
# shortcut.
#
# The two spellings are each other's gate as well. They share no code past the
# window, so agreement between them is independent of agreement with torch.
class TestResNetConv2d < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/resnet-18/model.safetensors', __dir__)
  STATE = File.expand_path('../data/resnet-18_state.safetensors', __dir__)

  # The stem sums 147 products into values that reach 32.8, and the worst seen
  # is 1.9e-05. 5e-05 leaves room for the two spellings and the two backends to
  # disagree with each other without hiding a real drift.
  TOLERANCE = 5.0e-5

  # prefix, input activation, expected activation, stride, padding.
  CASES = [
    ['resnet.embedder.embedder', 'pixel_values', 'conv1', 2, 3],
    ['resnet.encoder.stages.0.layers.0.layer.0', 'embedder', 'conv3x3', 1, 1],
    ['resnet.encoder.stages.1.layers.0.shortcut', 'stage.0', 'conv1x1', 2, 0]
  ].freeze

  def setup
    omit("#{MODEL} not found") unless File.exist?(MODEL)
    omit("#{STATE} not found; run `rake prepare:resnet`") unless File.exist?(STATE)

    @checkpoint = NArrayLLM::ResNet::Checkpoint.load(MODEL)
    @state = NArrayLLM::Safetensors.new(STATE)
  end

  def teardown
    @checkpoint&.close
    @state&.close
  end

  # torch keeps activations channel first and this works channel last.
  def channels_last(name)
    NArrayLLM::Ops.contiguous(@state[name].transpose(0, 2, 3, 1))
  end

  def convolution(prefix, stride, padding, spelling)
    NArrayLLM::ResNet::Conv2d.new(@checkpoint["#{prefix}.convolution.weight"], nil,
                                  stride: stride, padding: padding, spelling: spelling)
  end

  def test_every_kernel_shape_matches_the_reference
    CASES.each do |prefix, from, want, stride, padding|
      x = channels_last(from)
      expected = channels_last(want)
      %i[shift unfold].each do |spelling|
        y = convolution(prefix, stride, padding, spelling).call(x)
        assert_equal(expected.shape, y.shape, "#{want} #{spelling}")
        worst = host((y - expected).abs.max).to_f
        assert_operator(worst, :<, TOLERANCE, "#{want} #{spelling}: max|d| #{worst}")
      end
    end
  end

  # Not a consequence of the test above: both could be wrong in the same way
  # only if they shared the arithmetic, and past the window they do not.
  def test_the_two_spellings_agree
    CASES.each do |prefix, from, want, stride, padding|
      x = channels_last(from)
      a = convolution(prefix, stride, padding, :shift).call(x)
      b = convolution(prefix, stride, padding, :unfold).call(x)
      worst = host((a - b).abs.max).to_f
      assert_operator(worst, :<, TOLERANCE, "#{want}: max|d| #{worst}")
    end
  end

  def test_the_output_size_follows_the_padding_and_stride
    stem = convolution(*CASES[0].values_at(0, 3, 4), :shift)
    assert_equal(112, stem.out_size(224))
    assert_equal(7, stem.kernel)

    inner = convolution(*CASES[1].values_at(0, 3, 4), :shift)
    assert_equal(56, inner.out_size(56))
    assert_equal(3, inner.kernel)

    shortcut = convolution(*CASES[2].values_at(0, 3, 4), :shift)
    assert_equal(28, shortcut.out_size(56))
    assert_equal(1, shortcut.kernel)
  end

  # A bias is not used by this model, since every convolution is followed by a
  # batch norm, but folding the norm in will produce one.
  def test_a_bias_is_added_once_in_both_spellings
    prefix, from, _want, stride, padding = CASES[1]
    x = NArrayLLM::Ops.contiguous(channels_last(from)[0...2, true, true, true])
    weight = @checkpoint["#{prefix}.convolution.weight"]
    bias = @checkpoint["#{prefix}.normalization.bias"]

    %i[shift unfold].each do |spelling|
      plain = NArrayLLM::ResNet::Conv2d.new(weight, nil, stride: stride, padding: padding,
                                                         spelling: spelling).call(x)
      biased = NArrayLLM::ResNet::Conv2d.new(weight, bias, stride: stride, padding: padding,
                                                          spelling: spelling).call(x)
      worst = host(((biased - plain) - bias).abs.max).to_f
      assert_operator(worst, :<, TOLERANCE, "#{spelling}: max|d| #{worst}")
    end
  end

  def test_an_unknown_spelling_is_refused
    assert_raise(NArrayLLM::Error) do
      NArrayLLM::ResNet::Conv2d.new(@checkpoint["#{CASES[1][0]}.convolution.weight"], nil,
                                    spelling: :im2col)
    end
  end
end
