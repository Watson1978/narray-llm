# frozen_string_literal: true

module NArrayLLM
  module ResNet
    # BatchNorm at inference is an affine over the channel axis, so the four
    # buffers collapse into one scale and one shift at load. This is not the
    # folding the last stage does: the convolution still runs on its own.
    class Norm
      EPS = 1.0e-5

      def initialize(checkpoint, prefix, layout: :nhwc)
        @layout = layout
        weight = checkpoint["#{prefix}.normalization.weight"]
        bias = checkpoint["#{prefix}.normalization.bias"]
        mean = checkpoint["#{prefix}.normalization.running_mean"]
        variance = checkpoint["#{prefix}.normalization.running_var"]
        @scale = weight / XM::NMath.sqrt(variance + EPS)
        @shift = bias - (mean * @scale)
        return if layout == :nhwc

        # Channels first, so the axis to broadcast over is no longer the last
        # one and both have to carry the shape that says so.
        @scale = Ops.contiguous(@scale.reshape(1, @scale.size, 1, 1))
        @shift = Ops.contiguous(@shift.reshape(1, @shift.size, 1, 1))
      end

      attr_reader :scale, :shift

      def call(x)
        (x * @scale) + @shift
      end

      # Folding wants the per-channel numbers flat, whatever the layout.
      def flat_scale
        @layout == :nhwc ? @scale : @scale.reshape(@scale.size)
      end

      def flat_shift
        @layout == :nhwc ? @shift : @shift.reshape(@shift.size)
      end
    end

    # ResNetConvLayer: convolution, norm, and the activation unless the caller
    # is the one that adds the residual first.
    class ConvLayer
      attr_reader :conv, :norm

      def initialize(checkpoint, prefix, stride:, padding:, spelling:, activation: true,
                     fold: false)
        weight = checkpoint["#{prefix}.convolution.weight"]
        norm = Norm.new(checkpoint, prefix, layout: Conv2d.layout_for(spelling))
        bias = nil
        if fold
          # The norm scales each output channel, so it rides into the weight
          # of the convolution that produced the channel.
          scale = norm.flat_scale
          weight = weight * scale.reshape(scale.size, 1, 1, 1)
          bias = norm.flat_shift
        end
        @conv = Conv2d.new(weight, bias, stride: stride, padding: padding, spelling: spelling)
        @norm = norm unless fold
        @activation = activation
      end

      def call(x, prof: Profiler::NULL)
        y = @conv.call(x, prof: prof)
        y = prof.section(:norm) { @norm.call(y) } unless @norm.nil?
        return y unless @activation

        prof.section(:relu) { y.clip(0.0, nil) }
      end
    end

    # ResNetBasicLayer: two convolutions, the residual, then the activation.
    # The shortcut is a 1x1 convolution where the shape changes and the
    # identity where it does not.
    class BasicLayer
      def initialize(checkpoint, base, stride:, shortcut:, spelling:, fold: false)
        @first = ConvLayer.new(checkpoint, "#{base}.layer.0", stride: stride, padding: 1,
                                                             spelling: spelling, fold: fold)
        @second = ConvLayer.new(checkpoint, "#{base}.layer.1", stride: 1, padding: 1,
                                                              spelling: spelling,
                                                              activation: false, fold: fold)
        @shortcut = return_shortcut(checkpoint, base, stride, spelling, fold) if shortcut
      end

      def call(x, prof: Profiler::NULL)
        residual = @shortcut.nil? ? x : @shortcut.call(x, prof: prof)
        y = @second.call(@first.call(x, prof: prof), prof: prof)
        prof.section(:residual) { (y + residual).clip(0.0, nil) }
      end

      private

      def return_shortcut(checkpoint, base, stride, spelling, fold)
        ConvLayer.new(checkpoint, "#{base}.shortcut", stride: stride, padding: 0,
                                                     spelling: spelling, activation: false,
                                                     fold: fold)
      end
    end

    # One of the four stages. Only the first layer changes the shape, so only
    # it can need the shortcut.
    class Stage
      def initialize(checkpoint, index, spelling:, fold: false)
        config = checkpoint.config
        stride = config.stride_for(index)
        shortcut = config.shortcut?(index)
        @layers = Array.new(config.depths[index]) do |layer|
          BasicLayer.new(checkpoint, "resnet.encoder.stages.#{index}.layers.#{layer}",
                         stride: layer.zero? ? stride : 1,
                         shortcut: layer.zero? && shortcut, spelling: spelling, fold: fold)
        end
      end

      def call(x, prof: Profiler::NULL)
        @layers.reduce(x) { |acc, layer| layer.call(acc, prof: prof) }
      end
    end

    # ResNetEmbeddings: the 7x7 stem and the pooler, which halve the input
    # twice between them.
    class Embedder
      POOL = { kernel: 3, stride: 2, padding: 1 }.freeze

      def initialize(checkpoint, spelling:, fold: false)
        @stem = ConvLayer.new(checkpoint, Checkpoint::STEM, stride: 2, padding: 3,
                                                           spelling: spelling, fold: fold)
        @layout = Conv2d.layout_for(spelling)
      end

      def call(x, prof: Profiler::NULL)
        y = @stem.call(x, prof: prof)
        prof.section(:pool) { Ops.max_pool2d(y, layout: @layout, **POOL) }
      end
    end
  end
end
