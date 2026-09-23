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

    attr_reader :steps

    def initialize(model, **options)
      @model = model
      DEFAULTS.merge(options).each { |name, value| instance_variable_set("@#{name}", value) }
      @moments = {}
      @steps = 0
    end

    # grads is what Model#backward answers, in the checkpoint's layout.
    def step(grads)
      @steps += 1
      bias1 = 1.0 - (@beta1**@steps)
      bias2 = 1.0 - (@beta2**@steps)

      @model.each_parameter(grads) do |key, grad, param|
        first, second = (@moments[key] ||= [XF.zeros(*param.shape), XF.zeros(*param.shape)])
        first.store((@beta1 * first) + ((1.0 - @beta1) * grad))
        second.store((@beta2 * second) + ((1.0 - @beta2) * grad * grad))
        corrected = (first / bias1) / (XM::NMath.sqrt(second / bias2) + @eps)
        param.store(param - (@learning_rate * (corrected + (@weight_decay * param))))
      end
      @model.refresh_derived
      @steps
    end

    # Both moments, for every parameter. Two more copies of the model.
    def bytes
      @moments.values.sum { |pair| pair.sum { |a| a.size * XF::ELEMENT_BYTE_SIZE } }
    end
  end
end
