# frozen_string_literal: true

module NArrayLLM
  module ResNet
    # Conv2d over [N, H, W, in], which is the layout the matrix product wants.
    # torch stores the weight as [out, in, kh, kw] and the activations channel
    # first, so the model transposes once on the way in and once on the way out.
    #
    # The two spellings are the ones Whisper's Conv1d has, one axis up.
    #
    #   shift: one matmul per kernel position, accumulated into the output.
    #     Nine [rows, in] x [in, out] products for a 3x3.
    #   unfold: one matmul. The [rows, kh * kw * in] window matrix is built
    #     first, which is nine copies into a buffer, and then a single
    #     [rows, kh * kw * in] x [kh * kw * in, out] product.
    #
    # In one dimension unfold won by 1.6% (docs/results/whisper-tiny.md). Here
    # the window matrix is nine times the input rather than three, so which one
    # wins is the question this stage is here to answer (docs/plans/PLAN-conv2d.md).
    class Conv2d
      # cudnn is cumo's own conv, which cuDNN backs and which wants the
      # channels first. The other two are matrix products and want them last,
      # so the spelling decides the layout the whole model runs in.
      SPELLINGS = %i[shift unfold cudnn].freeze
      LAYOUTS = { shift: :nhwc, unfold: :nhwc, cudnn: :nchw }.freeze

      # Numo has no conv of its own, so that spelling is refused rather than
      # answered with a NoMethodError from three calls down.
      CUDNN = XF.method_defined?(:conv)

      def self.layout_for(spelling)
        LAYOUTS.fetch(spelling) { raise Error, "unknown spelling #{spelling.inspect}" }
      end

      def self.available?(spelling)
        spelling != :cudnn || CUDNN
      end

      attr_reader :in_channels, :out_channels, :kernel, :stride, :padding, :spelling

      # weight: [out, in, kh, kw] as stored. bias: [out] or nil.
      def initialize(weight, bias = nil, stride: 1, padding: 0, spelling: :shift)
        raise Error, "unknown spelling #{spelling.inspect}" unless SPELLINGS.include?(spelling)
        raise Error, "#{spelling} needs a backend with conv; this one has none" unless
          Conv2d.available?(spelling)

        @out_channels, @in_channels, height, width = weight.shape
        raise Error, "expected a square kernel, got #{height}x#{width}" unless height == width

        @kernel = height
        @stride = stride
        @padding = padding
        @spelling = spelling
        @weight = weight
        @bias_1d = bias
        @bias = bias.nil? ? nil : Ops.contiguous(bias.reshape(1, @out_channels))
        prepare(weight) unless @spelling == :cudnn
      end

      def layout
        Conv2d.layout_for(@spelling)
      end

      def out_size(size)
        ((size + (2 * @padding) - @kernel) / @stride) + 1
      end

      # x: [N, H, W, in], or [N, in, H, W] for cudnn. Answers the same layout.
      def call(x, prof: Profiler::NULL)
        return cudnn(x, prof) if @spelling == :cudnn

        batch, height, width, channels = x.shape
        raise Error, "expected #{@in_channels} channels, got #{channels}" unless channels == @in_channels

        out_h = out_size(height)
        out_w = out_size(width)
        source = @padding.zero? ? x : pad(x)
        rows = batch * out_h * out_w
        out = if @spelling == :unfold
                unfold(source, batch, rows, out_h, out_w, prof)
              else
                shift(source, rows, out_h, out_w, prof)
              end
        # view rather than the array itself: on the GPU shift accumulates with
        # gemm, whose answer carries the inplace flag, and the next operation
        # the caller writes would silently overwrite this buffer. view shares
        # it and drops the flag, and costs no kernel.
        out.view.reshape!(batch, out_h, out_w, @out_channels)
      end

      private

      # cuDNN takes the padding and the stride itself, and folds the bias in,
      # so the whole of the work above is one call.
      def cudnn(x, prof)
        channels = x.shape[1]
        raise Error, "expected #{@in_channels} channels, got #{channels}" unless channels == @in_channels

        prof.section(:conv_gemm) do
          @bias_1d.nil? ? x.conv(@weight, stride: [@stride, @stride], pad: [@padding, @padding])
                        : x.conv(@weight, stride: [@stride, @stride], pad: [@padding, @padding],
                                 b: @bias_1d)
        end
      end

      def shift(source, rows, out_h, out_w, prof)
        out = start(rows)
        @kernel.times do |i|
          @kernel.times do |j|
            patch = prof.section(:conv_window) do
              Ops.contiguous(window(source, i, j, out_h, out_w)).reshape!(rows, @in_channels)
            end
            out = prof.section(:conv_gemm) { Ops.linear_add(patch, @taps[(i * @kernel) + j], out) }
          end
        end
        out
      end

      # The buffer is [N, out_h, out_w, kh * kw * in] while it is filled, so
      # each tap is assigned at the shape it already has and nothing is
      # reshaped per tap. It is contiguous, so the flattening at the end is a
      # view.
      def unfold(source, batch, rows, out_h, out_w, prof)
        span = @kernel * @kernel * @in_channels
        # The taps together write every element, so the buffer is allocated
        # without a fill. new alone is lazy and refuses a write through a
        # slice.
        windows = XF.new(batch, out_h, out_w, span)
        windows.allocate
        prof.section(:conv_window) do
          @kernel.times do |i|
            @kernel.times do |j|
              at = ((i * @kernel) + j) * @in_channels
              windows[true, true, true, at...(at + @in_channels)] = window(source, i, j, out_h, out_w)
            end
          end
        end
        y = prof.section(:conv_gemm) { windows.reshape!(rows, span).dot(@unfolded) }
        @bias.nil? ? y : y + @bias
      end

      # The [N, out_h, out_w, in] view the kernel position (i, j) reads. Pure
      # strides, no index array, so it does not synchronize (AGENTS.md).
      def window(source, i, j, out_h, out_w)
        rows = (i..(i + ((out_h - 1) * @stride))).step(@stride)
        cols = (j..(j + ((out_w - 1) * @stride))).step(@stride)
        source[true, rows, cols, true]
      end

      # The accumulator shift adds into. store broadcasts the bias over the
      # rows, so the zeros fill is not paid on top of it.
      def start(rows)
        return XF.zeros(rows, @out_channels) if @bias.nil?

        XF.new(rows, @out_channels).store(@bias)
      end

      def pad(x)
        Ops.pad_with(x, @padding, 0.0)
      end

      # [out, in, kh, kw] -> one [in, out] per tap for shift, and one
      # [kh * kw * in, out] laid out tap major for unfold.
      def prepare(weight)
        @taps = Array.new(@kernel * @kernel) do |tap|
          Ops.contiguous(weight[true, true, tap / @kernel, tap % @kernel].transpose)
        end
        span = @kernel * @kernel * @in_channels
        @unfolded = Ops.contiguous(weight.transpose(2, 3, 1, 0)).reshape!(span, @out_channels)
      end
    end
  end
end
