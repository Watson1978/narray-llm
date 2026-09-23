# frozen_string_literal: true

module NArrayLLM
  # Numerical diff between two tensors. This is the main debugging tool for the
  # forward pass: when logits disagree, the answer is always "which layer went
  # wrong first", not "how big is the final error".
  module Compare
    Result = Struct.new(:label, :size, :shape, :tolerance, :max_abs, :mean_abs,
                        :ref_max_abs, :first_bad_flat_index, :first_bad_index,
                        :first_bad_expected, :first_bad_actual, :bad_count,
                        keyword_init: true) do
      def ok?
        max_abs <= tolerance
      end

      # max|d| divided by the largest magnitude in the reference. Activations in
      # GPT-2 range from ~1 to ~3000, and fp32 only carries ~7 digits, so an
      # absolute error alone cannot tell rounding apart from a real bug.
      def rel_max
        ref_max_abs.zero? ? max_abs : max_abs / ref_max_abs
      end

      def to_s
        base = format('%-22s n=%-9d max|d|=%.3e mean|d|=%.3e rel=%.3e',
                      label, size, max_abs, mean_abs, rel_max)
        return "#{base} OK" if ok?

        "#{base} NG (#{bad_count} over tol=#{format('%.1e', tolerance)}, " \
          "first at #{first_bad_index.inspect}: " \
          "expected #{format('%.6g', first_bad_expected)}, got #{format('%.6g', first_bad_actual)})"
      end
    end

    module_function

    # expected / actual may be any matching shape. tolerance only affects the
    # "first position over the threshold" report, never max_abs / mean_abs.
    def diff(expected, actual, tolerance: 0.0, label: 'diff')
      unless expected.shape == actual.shape
        raise Error, "#{label}: shape mismatch #{expected.shape.inspect} vs #{actual.shape.inspect}"
      end

      shape = expected.shape
      a = flatten_to(expected, XM::SFloat)
      b = flatten_to(actual, XM::SFloat)
      d = (a - b).abs

      # ceil(clip(d - tol, 0, 1)) is 1 exactly where d > tol. Arithmetic, so no
      # Bit array is involved (see AGENTS.md).
      over = (d - tolerance).clip(0.0, 1.0).ceil
      bad_count = NArrayLLM.scalar(over.sum).round

      result = Result.new(label: label, size: a.size, shape: shape, tolerance: tolerance,
                          max_abs: NArrayLLM.scalar(d.max), mean_abs: NArrayLLM.scalar(d.mean),
                          ref_max_abs: NArrayLLM.scalar(a.abs.max), bad_count: bad_count)
      return result if bad_count.zero?

      # Weight each violation by (size - i) so the maximum lands on the smallest
      # violating index, without relying on how max_index breaks ties.
      weights = XM::SFloat.new(a.size).seq(a.size, -1)
      flat = NArrayLLM.scalar((over * weights).max_index)
      result.first_bad_flat_index = flat
      result.first_bad_index = unravel(flat, shape)
      result.first_bad_expected = NArrayLLM.scalar(a[flat])
      result.first_bad_actual = NArrayLLM.scalar(b[flat])
      result
    end

    def stats(tensor, label: 'stats')
      t = flatten_to(tensor, XM::SFloat)
      # A NaN anywhere poisons the sum, which is how we detect it without a Bit
      # array (NaN != NaN cannot be asked directly here).
      total = NArrayLLM.scalar(t.sum)
      {
        label: label, size: t.size, shape: tensor.shape,
        mean: total / t.size, min: NArrayLLM.scalar(t.min),
        max: NArrayLLM.scalar(t.max), max_abs: NArrayLLM.scalar(t.abs.max),
        finite: !(total.nan? || total.infinite?)
      }
    end

    def unravel(flat_index, shape)
      shape.reverse.map { |dim| i = flat_index % dim; flat_index /= dim; i }.reverse
    end

    def flatten_to(tensor, klass)
      t = tensor.is_a?(klass) ? tensor : klass.cast(tensor)
      t.flatten
    end
  end
end
