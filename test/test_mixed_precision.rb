# frozen_string_literal: true

require_relative 'test_helper'

class TestMixedPrecision < Test::Unit::TestCase
  include TestHelper

  class OneTensorModel
    attr_reader :param, :master

    def initialize(param, master = nil)
      @param = param
      @master = master
    end

    def each_parameter(grads)
      yield [:w], grads[:w], @param
    end

    def master_weights(_checkpoint)
      { [:w] => @master }
    end

    def refresh_derived; end
  end

  def setup
    omit('16 ビットの型は Cumo (GPU=1) にしか無い') unless NArrayLLM.gpu?
  end

  def halves
    %i[HFloat BFloat].filter_map { |name| XM.const_get(name) if XM.const_defined?(name) }
  end

  def wide(array)
    NArrayLLM::Ops.contiguous(array, XM::SFloat)
  end

  def max_abs(array)
    host(array.abs.max).to_f
  end

  def test_one_hot_keeps_ids_past_the_16_bit_mantissa
    ids = [100, 3000, 50_000, 50_256]
    want = NArrayLLM::Backward.one_hot(XM::SFloat, ids, 50_257).to_a
    halves.each do |klass|
      hot = NArrayLLM::Backward.one_hot(klass, ids, 50_257)
      assert_equal(klass, hot.class)
      assert_equal(want, wide(hot).to_a, klass.name)
    end
  end

  def test_layernorm_backward_keeps_a_row_whose_square_overflows_fp16
    inp32 = XM::NMath.sin(XM::SFloat.new(4, 8).seq)
    inp32[0, true] = inp32[0, true] * 300.0
    dout32 = XM::NMath.cos(XM::SFloat.new(4, 8).seq * 0.7)
    weight32 = XM::SFloat.new(8).seq(0.5, 0.1)
    inp, dout, weight = [inp32, dout32, weight32].map { |a| NArrayLLM::Ops.contiguous(a, XM::HFloat) }

    want = NArrayLLM::Backward.layernorm(wide(dout), wide(inp), wide(weight))
    got = NArrayLLM::Backward.layernorm(dout, inp, weight)
    %w[dinp dweight dbias].each_with_index do |name, i|
      assert_equal(XM::HFloat, got[i].class, name)
      error = max_abs(wide(got[i]) - want[i]) / max_abs(want[i])
      assert_operator(error, :<, 1e-3, name)
    end
    assert_operator(max_abs(wide(got[0])[0, true]), :>, 0.0, 'the outlier row keeps its gradient')
  end

  def test_adamw_on_16_bit_weights_follows_fp32_from_the_master
    values = [1.0, -0.5, 0.25, 2.0, 0.001, -3.0]
    grad32 = XM::SFloat[2.0**-14, -2.0**-10, 2.0**-6, 0.5, -2.0**-12, 2.0**-8]

    halves.each do |klass|
      model = OneTensorModel.new(NArrayLLM::Ops.contiguous(XM::SFloat[*values], klass), XM::SFloat[*values])
      mixed = NArrayLLM::AdamW.new(model, checkpoint: :unused, loss_scale: 1024.0)
      grads = { w: NArrayLLM::Ops.contiguous(grad32 * 1024.0, klass) }
      reference = OneTensorModel.new(XM::SFloat[*values])
      fp32 = NArrayLLM::AdamW.new(reference)
      5.times do
        mixed.step(grads)
        fp32.step({ w: grad32 })
      end

      assert_equal(5, mixed.steps)
      assert_operator(max_abs(model.master - reference.param), :<, 1e-6, klass.name)
      assert_equal(NArrayLLM::Ops.contiguous(model.master, klass).to_a, model.param.to_a, klass.name)
    end
  end

  def test_adamw_skips_a_step_whose_scaled_gradient_overflowed
    master = XM::SFloat[1.0, 2.0]
    model = OneTensorModel.new(NArrayLLM::Ops.contiguous(master, XM::HFloat), master)
    optimiser = NArrayLLM::AdamW.new(model, checkpoint: :unused, loss_scale: 1024.0)
    grads = { w: NArrayLLM::Ops.contiguous(XM::SFloat[1.0e5, 1.0], XM::HFloat) }

    assert_equal(0, optimiser.step(grads))
    assert_equal([1, 512.0], [optimiser.skipped, optimiser.loss_scale])
    assert_equal([1.0, 2.0], model.master.to_a)
    assert_equal([1.0, 2.0], wide(model.param).to_a)
  end

  def test_adamw_refuses_16_bit_weights_without_a_checkpoint
    model = OneTensorModel.new(XM::HFloat[1.0])
    optimiser = NArrayLLM::AdamW.new(model)
    error = assert_raise(NArrayLLM::Error) { optimiser.step({ w: XM::HFloat[1.0] }) }
    assert_match(/checkpoint/, error.message)
  end
end
