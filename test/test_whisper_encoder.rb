# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-whisper.md).
#
# The reference is transformers, taken with forward hooks on the real model by
# python/whisper_dump.py over a waveform built from a formula. The mel is
# given rather than computed: the front end is a later stage, and separating
# them is what keeps an error attributable.
class TestWhisperEncoder < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/whisper-tiny/model.safetensors', __dir__)
  DUMP = File.expand_path('../data/whisper-tiny_encoder_state.safetensors', __dir__)

  # Four layers over 1500 positions, and the residual reaches 270 inside the
  # stack before the final norm brings it back to 19.7. The worst seen is
  # 3.2e-04 on Cumo; 5e-04 leaves room for the backends to disagree with each
  # other without hiding a real drift.
  TOLERANCE = 5.0e-4

  # conv1d of a [7, 2] ramp with a [3, 2, 3] ramp weight, from torch.
  CONV_X = [[0.0, 0.25], [0.5, 0.75], [1.0, 1.25], [1.5, 1.75],
            [2.0, 2.25], [2.5, 2.75], [3.0, 3.25]].freeze
  CONV_STRIDE1_HEAD = [1.075, 0.975, 2.625, 1.75, 3.0, 6.0].freeze
  CONV_STRIDE2_HEAD = [1.075, 0.975, 2.625, 2.5, 5.55, 10.35].freeze

  def setup
    omit("#{MODEL} not found; run `rake prepare:whisper`") unless File.exist?(MODEL)

    @checkpoint = NArrayLLM::Whisper::Checkpoint.load(MODEL)
    @dump = nil
  end

  def teardown
    @dump&.close
    @checkpoint&.close
  end

  def dump
    omit("#{DUMP} not found; run `rake prepare:whisper`") unless File.exist?(DUMP)

    @dump ||= NArrayLLM::Safetensors.new(DUMP)
  end

  def mel
    @mel ||= NArrayLLM::Ops.contiguous(dump['input_features'])
  end

  data('shift', :shift)
  data('unfold', :unfold)
  def test_the_encoder_matches_transformers(spelling)
    reference = dump['encoder_last_hidden_state']
    encoder = NArrayLLM::Whisper::Encoder.new(@checkpoint, spelling: spelling)
    worst = host((reference - encoder.forward(mel)).abs.max)
    assert_operator(worst, :<, TOLERANCE, spelling.to_s)
    notify(format('%s: 最大 max|d| = %.3e (閾値 %.1e)', spelling, worst, TOLERANCE))
  end

  def test_the_two_convolution_spellings_agree
    shift = NArrayLLM::Whisper::Encoder.new(@checkpoint, spelling: :shift).forward(mel)
    unfold = NArrayLLM::Whisper::Encoder.new(@checkpoint, spelling: :unfold).forward(mel)
    worst = host((shift - unfold).abs.max)
    assert_operator(worst, :<, TOLERANCE)
    notify(format('ずらして足す版と窓を展開する版の差: %.3e', worst))
  end

  # Both spellings against values taken from torch.nn.functional.conv1d, which
  # is what pins down the padding and the stride rather than the two of them
  # agreeing with each other.
  data('shift', :shift)
  data('unfold', :unfold)
  def test_conv1d_matches_torch(spelling)
    weight = XF.new(3, 2, 3).seq * 0.1
    bias = XF[0.5, -0.5, 0.25]
    x = XF.cast(CONV_X)
    { 1 => CONV_STRIDE1_HEAD, 2 => CONV_STRIDE2_HEAD }.each do |stride, want|
      conv = NArrayLLM::Whisper::Conv1d.new(weight, bias, stride: stride, padding: 1,
                                                          spelling: spelling)
      got = conv.call(x)
      assert_equal(stride == 1 ? [7, 3] : [4, 3], got.shape, "stride #{stride}")
      head = host(got.flatten[0...want.size]).to_a
      want.each_with_index do |value, i|
        assert_in_delta(value, head[i], 1.0e-5, "stride #{stride}, #{i}")
      end
    end
  end

  # The padding only reaches the first and last output rows, so a kernel of 3
  # with padding 1 has to leave the middle untouched by it.
  def test_the_padding_only_touches_the_ends
    weight = XF.ones(1, 1, 3)
    x = XF.ones(5, 1)
    got = host(NArrayLLM::Whisper::Conv1d.new(weight, nil, stride: 1, padding: 1).call(x))
    assert_equal([[2.0], [3.0], [3.0], [3.0], [2.0]], got.to_a)
  end

  # Whisper was trained with the erf gelu, not the tanh one GPT-2 uses. The two
  # differ by 4.1e-04 at their widest, which is above this stage's tolerance.
  def test_the_erf_gelu_is_not_the_tanh_one
    x = XF[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
    torch = [-0.00404969, -0.15865529, -0.15426877, 0.0, 0.34573123, 0.84134471, 2.99595031]
    erf = host(NArrayLLM::Ops.gelu_erf(x)).to_a
    torch.each_with_index { |want, i| assert_in_delta(want, erf[i], 1.0e-6, i.to_s) }
    assert_operator(host((NArrayLLM::Ops.gelu_erf(x) - NArrayLLM::Ops.gelu(x)).abs.max),
                    :>, 1.0e-4, '2 つは別の関数')
  end

  def test_a_mel_of_the_wrong_shape_is_rejected
    encoder = NArrayLLM::Whisper::Encoder.new(@checkpoint)
    assert_raise_kind_of(NArrayLLM::Error) { encoder.forward(XF.zeros(80, 100)) }
    assert_raise_kind_of(NArrayLLM::Error) { encoder.forward(XF.zeros(40, 3000)) }
  end

  def test_an_unknown_spelling_is_rejected
    assert_raise_kind_of(NArrayLLM::Error) do
      NArrayLLM::Whisper::Conv1d.new(XF.ones(1, 1, 3), nil, spelling: :whatever)
    end
  end
end
