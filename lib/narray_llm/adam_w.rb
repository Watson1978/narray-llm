# frozen_string_literal: true

module NArrayLLM
  # AdamW, following gpt2_update in llm.c's train_gpt2.c.
  #
  # Every parameter is updated in place. The model hands out the arrays it
  # actually computes with, and two of them alias one parameter (the token table
  # is kept in both orientations), so rebinding would leave one of the pair
  # behind and the fused decode path pointing at a stale buffer.
  class AdamW
    # test_gpt2.c:172 passes these.
    DEFAULTS = { learning_rate: 1e-4, beta1: 0.9, beta2: 0.999,
                 eps: 1e-8, weight_decay: 0.01 }.freeze

    FUSED =
      if defined?(XM::CUDA::ElementwiseKernel)
        XM::CUDA::ElementwiseKernel.new(
          'T g, T lr, T beta1, T one_minus_beta1, T beta2, T one_minus_beta2, ' \
          'T eps, T wd, T bias1, T bias2',
          'T p, T m, T v',
          <<~CUDA, 'narray_llm_adamw')
            m = beta1 * m + one_minus_beta1 * g;
            v = beta2 * v + one_minus_beta2 * g * g;
            T corrected = (m / bias1) / (sqrt(v / bias2) + eps);
            p = p - lr * (corrected + wd * p);
          CUDA
      end

    MIXED =
      if defined?(XM::CUDA::ElementwiseKernel)
        XM::CUDA::ElementwiseKernel.new(
          'H g, float32 inv_scale, float32 lr, float32 beta1, float32 one_minus_beta1, ' \
          'float32 beta2, float32 one_minus_beta2, float32 eps, float32 wd, ' \
          'float32 bias1, float32 bias2',
          'float32 p, float32 m, float32 v, H w',
          <<~CUDA, 'narray_llm_adamw_mixed')
            float gf = (float)g * inv_scale;
            m = beta1 * m + one_minus_beta1 * gf;
            v = beta2 * v + one_minus_beta2 * gf * gf;
            float corrected = (m / bias1) / (sqrtf(v / bias2) + eps);
            p = p - lr * (corrected + wd * p);
            w = H(p);
          CUDA
      end

    attr_reader :steps, :skipped, :loss_scale

    def initialize(model, checkpoint: nil, loss_scale: NArrayLLM.fp16? ? 1024.0 : 1.0, **options)
      @model = model
      @checkpoint = checkpoint
      @loss_scale = loss_scale
      DEFAULTS.merge(options).each { |name, value| instance_variable_set("@#{name}", value) }
      @moments = {}
      @masters = nil
      @steps = 0
      @skipped = 0
    end

    # grads is what Model#backward answers, in the checkpoint's layout.
    def step(grads)
      if @loss_scale != 1.0 && overflowed?(grads)
        @loss_scale /= 2
        @skipped += 1
        return @steps
      end

      @steps += 1
      bias1 = 1.0 - (@beta1**@steps)
      bias2 = 1.0 - (@beta2**@steps)

      @model.each_parameter(grads) do |key, grad, param|
        master = master_for(key, param)
        klass = (master || param).class
        first, second = (@moments[key] ||= [klass.zeros(*param.shape), klass.zeros(*param.shape)])
        update(grad, param, master, first, second, bias1, bias2)
      end
      @model.refresh_derived
      @steps
    end

    # Both moments, for every parameter, and the fp32 master copy of a 16-bit
    # model. Two or three more copies of the model.
    def bytes
      held = @moments.values.flatten + (@masters&.values || [])
      held.sum { |a| a.size * a.class::ELEMENT_BYTE_SIZE }
    end

    private

    def master_for(key, param)
      return unless NArrayLLM.half_width?(param.class)
      unless @checkpoint
        raise Error, "a #{param.class} parameter needs fp32 master weights; " \
                     'pass the checkpoint as AdamW.new(model, checkpoint:)'
      end

      (@masters ||= @model.master_weights(@checkpoint)).fetch(key)
    end

    def overflowed?(grads)
      total = grads.values.map { |g| (g - g).sum }.inject(:+)
      !NArrayLLM.scalar(total).to_f.finite?
    end

    def update(grad, param, master, first, second, bias1, bias2)
      if master
        if MIXED
          return MIXED.call(grad, 1.0 / @loss_scale, @learning_rate, @beta1, 1.0 - @beta1,
                            @beta2, 1.0 - @beta2, @eps, @weight_decay, bias1, bias2,
                            master, first, second, param)
        end

        descend(Ops.contiguous(grad, master.class) / @loss_scale, master, first, second, bias1, bias2)
        return param.store(master)
      end

      grad = grad / @loss_scale unless @loss_scale == 1.0
      if FUSED
        return FUSED.call(grad, @learning_rate, @beta1, 1.0 - @beta1, @beta2, 1.0 - @beta2,
                          @eps, @weight_decay, bias1, bias2, param, first, second)
      end

      descend(grad, param, first, second, bias1, bias2)
    end

    def descend(grad, param, first, second, bias1, bias2)
      first.store((@beta1 * first) + ((1.0 - @beta1) * grad))
      second.store((@beta2 * second) + ((1.0 - @beta2) * grad * grad))
      corrected = (first / bias1) / (XM::NMath.sqrt(second / bias2) + @eps)
      param.store(param - (@learning_rate * (corrected + (@weight_decay * param))))
    end
  end
end
