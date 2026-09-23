# frozen_string_literal: true

module NArrayLLM
  module Whisper
    # Conv1d over [frames, channels], which is the layout the rest of the
    # encoder wants. torch stores the weight as [out, in, kernel].
    #
    # This repository had never written a convolution, and there are two ways
    # to build one out of the array operations it already has. Which is faster
    # is the question the Whisper stage is here to answer; both are kept and
    # they have to agree.
    class Conv1d
      # shift: one matmul per kernel position, accumulated into the output.
      #   Three [frames, in] x [in, out] products and two adds.
      # unfold: one matmul. The [frames, in * kernel] window matrix is built
      #   first, which is three copies into a buffer, and then a single
      #   [frames, in * kernel] x [in * kernel, out] product.
      SPELLINGS = %i[shift unfold].freeze

      attr_reader :in_channels, :out_channels, :kernel, :stride, :padding, :spelling

      # weight: [out, in, kernel] as stored. bias: [out] or nil.
      def initialize(weight, bias, stride: 1, padding: 1, spelling: :shift)
        raise Error, "unknown spelling #{spelling.inspect}" unless SPELLINGS.include?(spelling)

        @out_channels, @in_channels, @kernel = weight.shape
        @stride = stride
        @padding = padding
        @spelling = spelling
        @bias = bias.nil? ? nil : Ops.contiguous(bias.reshape(1, @out_channels))
        prepare(weight)
      end

      def out_frames(frames)
        ((frames + (2 * @padding) - @kernel) / @stride) + 1
      end

      # x: [frames, in_channels]. Answers [out_frames, out_channels].
      def call(x, prof: Profiler::NULL)
        frames = x.shape[0]
        raise Error, "expected #{@in_channels} channels, got #{x.shape[1]}" unless x.shape[1] == @in_channels

        @spelling == :unfold ? unfold(x, frames, prof) : shift(x, frames, prof)
      end

      private

      def shift(x, frames, prof)
        out = @bias.nil? ? XF.zeros(out_frames(frames), @out_channels) : bias_rows(frames)
        @kernel.times do |tap|
          span = tap_span(frames, tap)
          next if span.nil?

          source, taps = span
          product = prof.section(:conv_gemm) { Ops.contiguous(x[source, true]).dot(@taps[tap]) }
          out[taps, true] = out[taps, true] + product
        end
        out
      end

      def unfold(x, frames, prof)
        # Zeroed, so the positions the padding covers stay zero.
        window = XF.zeros(out_frames(frames), @in_channels * @kernel)
        @kernel.times do |tap|
          span = tap_span(frames, tap)
          next if span.nil?

          source, taps = span
          window[taps, (tap * @in_channels)...((tap + 1) * @in_channels)] = Ops.contiguous(x[source, true])
        end
        product = prof.section(:conv_gemm) { window.dot(@flat) }
        @bias.nil? ? product : product + @bias
      end

      # Which output rows this kernel position reaches, and which input rows
      # feed them. source = t * stride + tap - padding, inside 0...frames.
      def tap_span(frames, tap)
        last = out_frames(frames) - 1
        first_t = [0, ((@padding - tap).to_f / @stride).ceil].max
        last_t = [last, (frames - 1 - tap + @padding) / @stride].min
        return nil if first_t > last_t

        first = (first_t * @stride) + tap - @padding
        final = (last_t * @stride) + tap - @padding
        [@stride == 1 ? first..final : (first..final).step(@stride), first_t..last_t]
      end

      def bias_rows(frames)
        XF.zeros(out_frames(frames), @out_channels) + @bias
      end

      # Stored [out, in, kernel]; every matmul here wants [in, out] per tap,
      # and the unfolded form wants them stacked as [in * kernel, out].
      def prepare(weight)
        @taps = Array.new(@kernel) do |tap|
          Ops.contiguous(weight[true, true, tap].transpose)
        end
        @flat = Ops.contiguous(XF.vstack(@taps))
      end
    end
  end
end
