# frozen_string_literal: true

require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-conv2d.md).
#
# The reference is transformers, taken with forward hooks by
# python/resnet_dump.py. Every stage is checked rather than the first,
# because the shortcut only appears from the second one and the stride only
# changes there.
class TestResNetBlock < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/resnet-18/model.safetensors', __dir__)
  STATE = File.expand_path('../data/resnet-18_state.safetensors', __dir__)

  # Eight convolutions deep by the last stage, where the residual reaches
  # 27.3. The worst seen is 3.6e-05.
  TOLERANCE = 1.0e-4

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

  def channels_last(name)
    NArrayLLM::Ops.contiguous(@state[name].transpose(0, 2, 3, 1))
  end

  def test_the_stem_and_every_stage_match_the_reference
    %i[shift unfold].each do |spelling|
      x = NArrayLLM::ResNet::Embedder.new(@checkpoint, spelling: spelling)
                                     .call(channels_last('pixel_values'))
      worst = host((x - channels_last('embedder')).abs.max).to_f
      assert_operator(worst, :<, TOLERANCE, "embedder #{spelling}: max|d| #{worst}")

      4.times do |stage|
        x = NArrayLLM::ResNet::Stage.new(@checkpoint, stage, spelling: spelling).call(x)
        worst = host((x - channels_last("stage.#{stage}")).abs.max).to_f
        assert_operator(worst, :<, TOLERANCE, "stage.#{stage} #{spelling}: max|d| #{worst}")
      end
    end
  end

  # On the GPU shift accumulates with gemm, whose answer carries the inplace
  # flag. A caller that then compared the answer with anything would have the
  # comparison write over it, which is how this was found.
  def test_the_answer_does_not_carry_the_inplace_flag
    x = channels_last('pixel_values')
    %i[shift unfold].each do |spelling|
      y = NArrayLLM::ResNet::Embedder.new(@checkpoint, spelling: spelling).call(x)
      assert_false(y.inplace?, "embedder #{spelling}")

      keep = y.dup
      y - channels_last('embedder')
      assert_equal(0.0, host((y - keep).abs.max).to_f, "embedder #{spelling} was written over")
    end
  end

  # ResNetEmbeddings halves twice and every stage after the first halves
  # again, so 224 reaches 7.
  def test_the_shape_halves_where_the_model_says
    x = NArrayLLM::ResNet::Embedder.new(@checkpoint, spelling: :unfold)
                                   .call(channels_last('pixel_values'))
    assert_equal([16, 56, 56, 64], x.shape)

    [[56, 64], [28, 128], [14, 256], [7, 512]].each_with_index do |(side, width), stage|
      x = NArrayLLM::ResNet::Stage.new(@checkpoint, stage, spelling: :unfold).call(x)
      assert_equal([16, side, side, width], x.shape, "stage.#{stage}")
    end
  end

  # A window that fell entirely in the padding has to lose to a real value,
  # and the activation before the pooler can produce zero.
  def test_max_pooling_pads_below_anything_it_can_see
    x = XF.zeros(1, 4, 4, 1)
    y = NArrayLLM::Ops.max_pool2d(x, kernel: 3, stride: 2, padding: 1)
    assert_equal([1, 2, 2, 1], y.shape)
    assert_equal(0.0, host(y.max).to_f)
    assert_equal(0.0, host(y.min).to_f)
  end

  def test_global_average_pooling_answers_one_row_per_image
    x = XF.new(2, 3, 3, 4).seq
    y = NArrayLLM::Ops.global_average_pool(x)
    assert_equal([2, 4], y.shape)
    assert_in_delta(host(x[0, true, true, 0].mean).to_f, host(y[0, 0]).to_f, 1.0e-6)
  end
end
