# frozen_string_literal: true

require_relative 'test_helper'

class TestCompare < Test::Unit::TestCase
  include TestHelper

  def test_identical_tensors_have_zero_error
    a = XM::SFloat.new(3, 4).seq
    r = NArrayLLM::Compare.diff(a, a.clone, tolerance: 1e-6, label: 'same')
    assert_equal(0.0, r.max_abs)
    assert_equal(0.0, r.mean_abs)
    assert_equal(0, r.bad_count)
    assert_true(r.ok?)
    assert_nil(r.first_bad_index)
  end

  def test_max_and_mean_absolute_error
    a = XM::SFloat[[0.0, 0.0], [0.0, 0.0]]
    b = XM::SFloat[[1.0, -3.0], [0.0, 0.0]]
    r = NArrayLLM::Compare.diff(a, b, tolerance: 0.5)
    assert_equal(3.0, r.max_abs)
    assert_in_delta(1.0, r.mean_abs, 1e-6) # (1 + 3 + 0 + 0) / 4
    assert_equal(2, r.bad_count)
    assert_false(r.ok?)
  end

  # The whole point of the tool: report where the divergence starts, not just
  # how large it gets.
  def test_reports_the_first_position_over_the_threshold
    a = XM::SFloat.zeros(2, 3, 4)
    b = XM::SFloat.zeros(2, 3, 4)
    b[1, 2, 1] = 5.0  # flat index 1*12 + 2*4 + 1 = 21
    b[0, 1, 3] = 0.7  # flat index 0*12 + 1*4 + 3 = 7, comes first
    r = NArrayLLM::Compare.diff(a, b, tolerance: 0.5)
    assert_equal(7, r.first_bad_flat_index)
    assert_equal([0, 1, 3], r.first_bad_index)
    assert_equal(0.0, r.first_bad_expected)
    assert_in_delta(0.7, r.first_bad_actual, 1e-6)
    assert_equal(5.0, r.max_abs)
    assert_equal(2, r.bad_count)
  end

  def test_exactly_at_the_tolerance_is_not_a_violation
    a = XM::SFloat[0.0]
    b = XM::SFloat[0.25] # exactly representable in fp32, so no rounding slack
    assert_equal(0, NArrayLLM::Compare.diff(a, b, tolerance: 0.25).bad_count)
    assert_equal(1, NArrayLLM::Compare.diff(a, b, tolerance: 0.2).bad_count)
  end

  def test_shape_mismatch_raises
    assert_raise(NArrayLLM::Error) do
      NArrayLLM::Compare.diff(XM::SFloat.zeros(2, 3), XM::SFloat.zeros(3, 2))
    end
  end

  def test_unravel_matches_row_major_order
    assert_equal([1, 2, 3], NArrayLLM::Compare.unravel(1 * 12 + 2 * 4 + 3, [2, 3, 4]))
    assert_equal([0, 0, 0], NArrayLLM::Compare.unravel(0, [2, 3, 4]))
    assert_equal([1, 2, 3], NArrayLLM::Compare.unravel(23, [2, 3, 4]))
  end

  def test_stats_flags_non_finite_values
    ok = NArrayLLM::Compare.stats(XM::SFloat[1.0, 2.0, 3.0])
    assert_true(ok[:finite])
    assert_in_delta(2.0, ok[:mean], 1e-6)
    assert_equal(3.0, ok[:max])
    assert_equal(1.0, ok[:min])

    nan = XM::SFloat[1.0, 0.0]
    nan[1] = Float::NAN
    assert_false(NArrayLLM::Compare.stats(nan)[:finite])

    inf = XM::SFloat[1.0, 0.0]
    inf[1] = Float::INFINITY
    assert_false(NArrayLLM::Compare.stats(inf)[:finite])
  end
end
