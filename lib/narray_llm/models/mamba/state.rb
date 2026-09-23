# frozen_string_literal: true

module NArrayLLM
  module Mamba
    # The two recurrences a Mamba block carries, one pair per layer
    # (mamba.c:64). This is what a KV cache would be, except that every entry
    # is rewritten on every token rather than a new row being appended.
    class State
      attr_reader :num_layers, :d_inner, :d_conv, :d_state

      def initialize(num_layers:, d_inner:, d_conv:, d_state:)
        @num_layers = num_layers
        @d_inner = d_inner
        @d_conv = d_conv
        @d_state = d_state
        @conv = Array.new(num_layers) { XF.zeros(d_inner, d_conv) }
        @ssm = Array.new(num_layers) { XF.zeros(d_inner, d_state) }
        @position = 0
      end

      attr_reader :position

      def self.bytes_for(num_layers:, d_inner:, d_conv:, d_state:)
        XF::ELEMENT_BYTE_SIZE * num_layers * d_inner * (d_conv + d_state)
      end

      def bytes
        State.bytes_for(num_layers: @num_layers, d_inner: @d_inner,
                        d_conv: @d_conv, d_state: @d_state)
      end

      def conv(layer)
        @conv.fetch(layer)
      end

      def ssm(layer)
        @ssm.fetch(layer)
      end

      def conv=(pair)
        layer, value = pair
        @conv[layer] = value
      end

      def ssm=(pair)
        layer, value = pair
        @ssm[layer] = value
      end

      def advance
        @position += 1
      end

      # mamba.c:101 reset_internal_state.
      def reset
        @conv.each { |a| a.store(0.0) }
        @ssm.each { |a| a.store(0.0) }
        @position = 0
        self
      end
    end
  end
end
