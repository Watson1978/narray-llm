# frozen_string_literal: true

require_relative 'test_helper'

class TestOps < Test::Unit::TestCase
  include TestHelper

  O = NArrayLLM::Ops

  # fp32 carries ~7 significant digits, so tolerances have to scale with the
  # magnitude of the value. 1e-6 relative is loose enough for rounding and tight
  # enough to catch a wrong formula.
  FP32_REL = 1e-6

  def fp32_delta(expected)
    FP32_REL * [1.0, host(expected).abs].max
  end

  # --- contiguous ---

  def test_contiguous_copies_a_column_slice_without_changing_values
    a = XM::SFloat.new(3, 4).seq
    view = a[true, 1...3]
    copy = O.contiguous(view)
    assert_equal([3, 2], copy.shape)
    assert_equal(view.to_a, copy.to_a)
    copy[0, 0] = -1.0
    assert_equal(1.0, host(a[0, 1]), 'the copy must not alias the original')
  end

  # --- linear ---

  def test_linear_matches_a_hand_computed_product
    x = XM::SFloat[[1.0, 2.0], [3.0, 4.0]]        # [N=2, C=2]
    weight_t = XM::SFloat[[10.0, 20.0, 30.0],     # [C=2, OC=3]
                          [40.0, 50.0, 60.0]]
    bias = XM::SFloat[1.0, 2.0, 3.0]
    # row0: [1*10+2*40, 1*20+2*50, 1*30+2*60] + bias = [90, 120, 150] + [1,2,3]
    # row1: [3*10+4*40, 3*20+4*50, 3*30+4*60] + bias = [190, 260, 330] + [1,2,3]
    assert_equal([[91.0, 122.0, 153.0], [191.0, 262.0, 333.0]],
                 O.linear(x, weight_t, bias).to_a)
    assert_equal([[90.0, 120.0, 150.0], [190.0, 260.0, 330.0]],
                 O.linear(x, weight_t).to_a)
  end

  # --- layernorm ---

  def test_layernorm_matches_a_hand_computed_row
    x = XM::SFloat[[1.0, 2.0, 3.0, 4.0]]
    mean = 2.5
    var = ((1 - mean)**2 + (2 - mean)**2 + (3 - mean)**2 + (4 - mean)**2) / 4.0  # 1.25
    rstd = 1.0 / Math.sqrt(var + O::LAYERNORM_EPS)
    expected = [1, 2, 3, 4].map { |v| (v - mean) * rstd }
    out = O.layernorm(x, XM::SFloat.ones(4), XM::SFloat.zeros(4))
    expected.each_with_index { |e, i| assert_in_delta(e, host(out[0, i]), fp32_delta(e)) }
  end

  def test_layernorm_output_rows_have_zero_mean_and_unit_variance
    x = XM::SFloat.new(5, 64).seq * 0.37 - 3.0
    out = O.layernorm(x, XM::SFloat.ones(64), XM::SFloat.zeros(64))
    5.times do |r|
      row = out[r, true]
      assert_in_delta(0.0, NArrayLLM.scalar(row.mean), 1e-5, "row #{r} mean")
      # eps inside the rstd makes the variance a hair under 1, not over.
      assert_in_delta(1.0, NArrayLLM.scalar((row * row).mean), 1e-4, "row #{r} variance")
    end
  end

  def test_layernorm_applies_weight_and_bias
    x = XM::SFloat[[1.0, 2.0, 3.0, 4.0]]
    plain = O.layernorm(x, XM::SFloat.ones(4), XM::SFloat.zeros(4))
    weight = XM::SFloat[2.0, 3.0, 4.0, 5.0]
    bias = XM::SFloat[10.0, 20.0, 30.0, 40.0]
    scaled = O.layernorm(x, weight, bias)
    4.times do |i|
      expected = host(plain[0, i]) * host(weight[i]) + host(bias[i])
      assert_in_delta(expected, host(scaled[0, i]), fp32_delta(expected))
    end
  end

  def test_layernorm_of_a_constant_row_is_finite
    out = O.layernorm(XM::SFloat[[7.0] * 8], XM::SFloat.ones(8), XM::SFloat.zeros(8))
    assert_true(NArrayLLM::Compare.stats(out)[:finite])
    assert_equal([[0.0] * 8], out.to_a)
  end

  # --- gelu ---

  def test_gelu_matches_the_tanh_approximation
    xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]
    out = O.gelu(XM::SFloat.cast(xs))
    xs.each_with_index do |x, i|
      expected = 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math::PI) * (x + 0.044715 * x**3)))
      assert_in_delta(expected, host(out[i]), 1e-6, "gelu(#{x})")
    end
  end

  # GPT-2 uses the tanh approximation, not the exact erf form. At x=1 they
  # differ by ~1.5e-4, so this fails loudly if the two are ever swapped.
  def test_gelu_is_not_the_erf_formulation
    erf_at_one = 0.5 * (1.0 + Math.erf(1.0 / Math.sqrt(2.0)))
    got = host(O.gelu(XM::SFloat[1.0])[0])
    assert_operator((got - erf_at_one).abs, :>, 1e-5,
                    'gelu(1) landed on the erf value; the tanh approximation is required')
    assert_in_delta(0.841_191_99, got, 1e-6)
  end

  def test_gelu_saturates
    assert_equal(0.0, host(O.gelu(XM::SFloat[0.0])[0]))
    assert_in_delta(10.0, host(O.gelu(XM::SFloat[10.0])[0]), 1e-5)
    assert_in_delta(0.0, host(O.gelu(XM::SFloat[-10.0])[0]), 1e-5)
  end

  # --- softmax ---

  def test_softmax_rows_sum_to_one
    x = XM::SFloat.new(4, 7).seq * 0.9 - 3.0
    p = O.softmax_rows(x)
    4.times { |r| assert_in_delta(1.0, NArrayLLM.scalar(p[r, true].sum), 1e-6, "row #{r}") }
  end

  def test_softmax_of_equal_logits_is_uniform
    assert_equal([[0.25] * 4], O.softmax_rows(XM::SFloat[[2.0] * 4]).to_a)
  end

  def test_softmax_is_shift_invariant
    x = XM::SFloat[[1.0, 2.0, 3.0]]
    a = O.softmax_rows(x)
    b = O.softmax_rows(x + 1000.0)
    3.times { |i| assert_in_delta(host(a[0, i]), host(b[0, i]), 1e-6) }
  end

  # Without subtracting the row max, exp(1000) overflows to Inf in fp32 and the
  # division yields NaN.
  def test_softmax_survives_large_logits
    p = O.softmax_rows(XM::SFloat[[1000.0, 1001.0, 999.0]])
    assert_true(NArrayLLM::Compare.stats(p)[:finite])
    assert_in_delta(1.0, NArrayLLM.scalar(p.sum), 1e-6)
    assert_operator(host(p[0, 1]), :>, host(p[0, 0]))
  end

  def test_softmax_ignores_fully_masked_positions
    # A row of a causal mask: only the first two entries are allowed.
    x = XM::SFloat[[0.0, 0.0, O::MASK_VALUE, O::MASK_VALUE]]
    p = O.softmax_rows(x)
    assert_equal([[0.5, 0.5, 0.0, 0.0]], p.to_a)
  end

  # --- causal mask ---

  def test_causal_mask_blocks_only_the_strict_future
    m = O.causal_mask(4)
    assert_equal([4, 4], m.shape)
    4.times do |i|
      4.times do |j|
        expected = j > i ? O::MASK_VALUE : 0.0
        assert_equal(expected, host(m[i, j]), "mask[#{i},#{j}]")
      end
    end
  end

  def test_causal_mask_of_length_one_is_zero
    assert_equal([[0.0]], O.causal_mask(1).to_a)
  end

  # --- one-hot ---

  def test_one_hot_has_a_single_one_per_row
    oh = O.one_hot([0, 3, 1], 5)
    assert_equal([3, 5], oh.shape)
    assert_equal([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]].map { |r| r.map(&:to_f) },
                 oh.to_a)
  end

  def test_one_hot_works_at_the_top_of_the_gpt2_vocabulary
    # 50256 is exactly representable in fp32 (< 2^24), which is what the
    # arithmetic construction relies on.
    oh = O.one_hot([50_256], 50_257)
    assert_equal(1.0, host(oh[0, 50_256]))
    assert_equal(1.0, NArrayLLM.scalar(oh.sum))
  end

  def test_one_hot_times_a_table_selects_rows
    table = XM::SFloat.new(5, 3).seq
    picked = O.one_hot([2, 0], 5).dot(table)
    assert_equal([table[2, true].to_a, table[0, true].to_a], picked.to_a)
  end

  # --- attention ---

  # The per-head loop the batched implementation replaces. Kept here as an
  # independent oracle rather than in lib, where nothing calls it any more.
  def reference_head(q, k, v, mask, scale)
    O.softmax_rows(q.dot(k.transpose) * scale + mask).dot(v)
  end

  def reference_attention(qkv, batch_size, seq_len, num_heads, mask)
    channels = qkv.shape[1] / 3
    head_size = channels / num_heads
    scale = 1.0 / Math.sqrt(head_size)
    packed = qkv.reshape(batch_size, seq_len, 3 * channels)
    out = XM::SFloat.zeros(batch_size, seq_len, channels)
    batch_size.times do |b|
      num_heads.times do |h|
        lo = h * head_size
        slice = lo...(lo + head_size)
        out[b, true, slice] = reference_head(
          O.contiguous(packed[b, true, slice]),
          O.contiguous(packed[b, true, (channels + lo)...(channels + lo + head_size)]),
          O.contiguous(packed[b, true, (2 * channels + lo)...(2 * channels + lo + head_size)]),
          mask, scale
        )
      end
    end
    out.reshape(batch_size * seq_len, channels)
  end

  # The batched implementation must reproduce the loop it replaced, at the real
  # forward-pass shape as well as the small ones.
  data('B=1 T=1', [1, 1, 2, 8])
  data('B=1 T=5', [1, 5, 2, 8])
  data('B=2 T=4', [2, 4, 2, 8])
  data('B=4 T=64 GPT-2 shape', [4, 64, 12, 768])
  def test_batched_attention_matches_the_per_head_loop(shape)
    batch_size, seq_len, heads, channels = shape
    qkv = XM::NMath.sin(XM::SFloat.new(batch_size * seq_len, 3 * channels).seq * 0.013) * 0.9
    mask = O.causal_mask(seq_len)
    expected = reference_attention(qkv, batch_size, seq_len, heads, mask)
    got = O.attention(qkv, batch_size: batch_size, seq_len: seq_len, num_heads: heads, mask: mask)
    # 1e-5 is ~80 fp32 ULPs at these magnitudes. Measured: 0 on Numo, 4.5e-7 on
    # Cumo, where a batched cuBLAS call and NH separate ones round differently.
    d = NArrayLLM::Compare.diff(expected, got, tolerance: 1e-5, label: 'batched vs looped')
    assert_true(d.ok?, d.to_s)
    assert_equal(0.0, d.max_abs, d.to_s) unless NArrayLLM.gpu?
  end

  def test_attention_head_with_flat_scores_averages_the_visible_values
    t, hs = 4, 2
    q = XM::SFloat.zeros(t, hs)
    k = XM::SFloat.zeros(t, hs)
    v = XM::SFloat.new(t, hs).seq
    out = reference_head(q, k, v, O.causal_mask(t), 1.0)
    # All scores are 0, so each row attends uniformly over positions 0..t.
    t.times do |i|
      hs.times do |c|
        expected = (0..i).sum { |j| host(v[j, c]) } / (i + 1.0)
        assert_in_delta(expected, host(out[i, c]), 1e-5, "out[#{i},#{c}]")
      end
    end
  end

  def test_attention_head_first_position_sees_only_itself
    t, hs = 3, 2
    q = XM::SFloat.new(t, hs).seq * 0.1
    k = XM::SFloat.new(t, hs).seq * 0.2
    v = XM::SFloat[[1.0, 2.0], [30.0, 40.0], [500.0, 600.0]]
    out = reference_head(q, k, v, O.causal_mask(t), 1.0)
    assert_in_delta(1.0, host(out[0, 0]), 1e-5)
    assert_in_delta(2.0, host(out[0, 1]), 1e-5)
  end

  def test_attention_is_causal
    b, t, c, nh = 2, 5, 4, 2
    qkv = XM::SFloat.new(b * t, 3 * c).seq * 0.013 - 0.4
    mask = O.causal_mask(t)
    base = O.attention(qkv, batch_size: b, seq_len: t, num_heads: nh, mask: mask)

    # Perturb the value vector of the last position of batch 0 only.
    perturbed = qkv.clone
    perturbed[t - 1, (2 * c)...(3 * c)] = XM::SFloat.new(c).seq + 99.0
    after = O.attention(perturbed, batch_size: b, seq_len: t, num_heads: nh, mask: mask)

    earlier = NArrayLLM::Compare.diff(base[0...(t - 1), true], after[0...(t - 1), true],
                                       tolerance: 0.0, label: 'earlier positions')
    assert_equal(0.0, earlier.max_abs, 'a later token must not affect earlier outputs')
    last = NArrayLLM::Compare.diff(base[t - 1, true], after[t - 1, true], tolerance: 0.0)
    assert_operator(last.max_abs, :>, 1.0, 'the perturbed position itself must change')
  end

  def test_single_head_attention_equals_the_head_helper
    t, c = 6, 4
    qkv = XM::SFloat.new(t, 3 * c).seq * 0.07 - 0.5
    mask = O.causal_mask(t)
    packed = O.attention(qkv, batch_size: 1, seq_len: t, num_heads: 1, mask: mask)
    manual = reference_head(O.contiguous(qkv[true, 0...c]),
                            O.contiguous(qkv[true, c...(2 * c)]),
                            O.contiguous(qkv[true, (2 * c)...(3 * c)]),
                            mask, 1.0 / Math.sqrt(c))
    d = NArrayLLM::Compare.diff(manual, packed, tolerance: 1e-5)
    assert_true(d.ok?, d.to_s)
  end

  def test_attention_heads_are_independent
    # Two heads over identical column blocks must produce identical outputs.
    t, hs, nh = 4, 3, 2
    c = hs * nh
    one = XM::SFloat.new(t, 3 * hs).seq * 0.11 - 0.3
    qkv = XM::SFloat.zeros(t, 3 * c)
    3.times do |block|
      nh.times do |h|
        qkv[true, (block * c + h * hs)...(block * c + (h + 1) * hs)] = one[true, (block * hs)...((block + 1) * hs)]
      end
    end
    out = O.attention(qkv, batch_size: 1, seq_len: t, num_heads: nh, mask: O.causal_mask(t))
    assert_equal(0.0, NArrayLLM::Compare.diff(out[true, 0...hs], out[true, hs...c],
                                               tolerance: 0.0).max_abs)
  end

  def test_attention_degenerate_single_position
    out = O.attention(XM::SFloat.new(1, 12).seq, batch_size: 1, seq_len: 1, num_heads: 2,
                      mask: O.causal_mask(1))
    assert_equal([1, 4], out.shape)
    # With T=1 attention is the identity on V.
    assert_equal([[8.0, 9.0, 10.0, 11.0]], out.to_a)
  end

  # --- transformer block ---

  def block_weights(channels, zero_projections: false)
    hidden = 4 * channels
    zero = ->(*shape) { XM::SFloat.zeros(*shape) }
    ramp = ->(*shape) { XM::SFloat.new(*shape).seq * 0.01 - 0.2 }
    {
      ln1w: XM::SFloat.ones(channels) * 1.1, ln1b: ramp.call(channels),
      qkvw_t: ramp.call(channels, 3 * channels), qkvb: ramp.call(3 * channels),
      attprojw_t: zero_projections ? zero.call(channels, channels) : ramp.call(channels, channels),
      attprojb: zero_projections ? zero.call(channels) : ramp.call(channels),
      ln2w: XM::SFloat.ones(channels) * 0.9, ln2b: ramp.call(channels),
      fcw_t: ramp.call(channels, hidden), fcb: ramp.call(hidden),
      fcprojw_t: zero_projections ? zero.call(hidden, channels) : ramp.call(hidden, channels),
      fcprojb: zero_projections ? zero.call(channels) : ramp.call(channels)
    }
  end

  # If both projections output zero, the block is exactly the identity. That can
  # only hold if both residual connections are wired up.
  def test_transformer_block_with_zero_projections_is_the_identity
    b, t, c, nh = 2, 4, 8, 2
    x = XM::SFloat.new(b * t, c).seq * 0.03 - 0.5
    out = O.transformer_block(x, block_weights(c, zero_projections: true),
                              batch_size: b, seq_len: t, num_heads: nh,
                              mask: O.causal_mask(t))
    assert_equal(0.0, NArrayLLM::Compare.diff(x, out, tolerance: 0.0).max_abs,
                 'a missing residual connection would show up here')
  end

  def test_transformer_block_preserves_shape_and_changes_the_input
    b, t, c, nh = 2, 4, 8, 2
    x = XM::SFloat.new(b * t, c).seq * 0.03 - 0.5
    out = O.transformer_block(x, block_weights(c), batch_size: b, seq_len: t,
                              num_heads: nh, mask: O.causal_mask(t))
    assert_equal([b * t, c], out.shape)
    assert_true(NArrayLLM::Compare.stats(out)[:finite])
    assert_operator(NArrayLLM::Compare.diff(x, out, tolerance: 0.0).max_abs, :>, 1e-3)
  end

  def test_transformer_block_is_causal
    b, t, c, nh = 1, 5, 8, 2
    w = block_weights(c)
    mask = O.causal_mask(t)
    x = XM::SFloat.new(b * t, c).seq * 0.03 - 0.5
    base = O.transformer_block(x, w, batch_size: b, seq_len: t, num_heads: nh, mask: mask)

    perturbed = x.clone
    perturbed[t - 1, true] = XM::SFloat.new(c).seq + 7.0
    after = O.transformer_block(perturbed, w, batch_size: b, seq_len: t, num_heads: nh, mask: mask)

    d = NArrayLLM::Compare.diff(base[0...(t - 1), true], after[0...(t - 1), true], tolerance: 0.0)
    assert_equal(0.0, d.max_abs, 'a later token must not affect earlier positions')
  end

  def test_transformer_block_degenerate_single_token
    c, nh = 8, 2
    x = XM::SFloat.new(1, c).seq * 0.1
    out = O.transformer_block(x, block_weights(c), batch_size: 1, seq_len: 1,
                              num_heads: nh, mask: O.causal_mask(1))
    assert_equal([1, c], out.shape)
    assert_true(NArrayLLM::Compare.stats(out)[:finite])
  end

  def test_transformer_block_trace_records_every_intermediate
    c, nh = 8, 2
    trace = {}
    O.transformer_block(XM::SFloat.new(4, c).seq * 0.1, block_weights(c),
                        batch_size: 1, seq_len: 4, num_heads: nh,
                        mask: O.causal_mask(4), trace: trace, prefix: 'L0/')
    assert_equal(%w[L0/ln1 L0/qkv L0/atty L0/attproj L0/residual2
                    L0/ln2 L0/fch L0/fch_gelu L0/fcproj L0/residual3], trace.keys)
    assert_equal([4, 3 * c], trace['L0/qkv'].shape)
    assert_equal([4, 4 * c], trace['L0/fch_gelu'].shape)
    assert_equal([4, c], trace['L0/residual3'].shape)
  end

  def test_profiler_records_only_when_enabled
    c = 8
    off = NArrayLLM::Profiler.new(enabled: false)
    O.transformer_block(XM::SFloat.new(4, c).seq * 0.1, block_weights(c), batch_size: 1,
                        seq_len: 4, num_heads: 2, mask: O.causal_mask(4), prof: off)
    assert_empty(off.totals)

    on = NArrayLLM::Profiler.new(enabled: true)
    O.transformer_block(XM::SFloat.new(4, c).seq * 0.1, block_weights(c), batch_size: 1,
                        seq_len: 4, num_heads: 2, mask: O.causal_mask(4), prof: on)
    assert_equal(%i[attention gelu gemm layernorm residual softmax].sort,
                 on.totals.keys.sort)
    assert_equal(4, on.counts[:gemm])
    assert_equal(2, on.counts[:layernorm])
    assert_equal(2, on.counts[:residual])
    assert_operator(on.total, :>, 0.0)
  end

  # --- decode attention (KV キャッシュ経路) ---

  # Deterministic but varied: Numo and Cumo have different RNGs, so a seq-based
  # expression is the only input both backends can be given (AGENTS.md).
  def varied(*shape)
    XM::NMath.sin(XM::SFloat.new(*shape).seq * 1.7) * 0.9
  end

  # A single query against the cache has to reproduce the last row of a full
  # recompute. Verified on the part before trusting it in the whole model.
  data('t=1', 1)
  data('t=7', 7)
  data('t=64', 64)
  def test_decode_attention_matches_the_last_position_of_full_attention(seq_len)
    channels, heads = 8, 2
    qkv = varied(seq_len, 3 * channels)
    full = O.attention(qkv, batch_size: 1, seq_len: seq_len, num_heads: heads,
                       mask: O.causal_mask(seq_len))
    query = O.contiguous(qkv[(seq_len - 1)...seq_len, 0...channels])
    keys = O.contiguous(qkv[true, channels...(2 * channels)])
    values = O.contiguous(qkv[true, (2 * channels)...(3 * channels)])

    decoded = O.decode_attention(query, keys, values, num_heads: heads)
    assert_equal([1, channels], decoded.shape)

    # The batched decode sums each head's channel block with a reduction where
    # the full path uses a GEMM, so the two round differently and neither
    # backend is bit-exact here. 1e-6 is roughly ten fp32 ULPs at these
    # magnitudes; measured 3e-8 on both.
    d = NArrayLLM::Compare.diff(full[(seq_len - 1)...seq_len, true], decoded,
                                 tolerance: 1e-6, label: "decode vs full t=#{seq_len}")
    assert_true(d.ok?, d.to_s)
  end

  def test_decode_attention_over_a_single_cached_row_returns_that_value
    # With one key there is nothing to weigh: softmax of a single score is 1.
    values = XM::SFloat[[3.0, 4.0, 5.0, 6.0]]
    out = O.decode_attention(XM::SFloat[[1.0, 2.0, 3.0, 4.0]], XM::SFloat[[9.0, 9.0, 9.0, 9.0]],
                             values, num_heads: 2)
    assert_equal(values.to_a, out.to_a)
  end

  def test_decode_attention_heads_do_not_mix
    channels, heads = 8, 2
    head_size = channels / heads
    query = varied(1, channels)
    keys = varied(5, channels)
    values = varied(5, channels)
    base = O.decode_attention(query, keys, values, num_heads: heads)

    perturbed = values.clone
    perturbed[true, head_size...channels] = varied(5, head_size) + 10.0
    after = O.decode_attention(query, keys, perturbed, num_heads: heads)

    d = NArrayLLM::Compare.diff(base[true, 0...head_size], after[true, 0...head_size],
                                 tolerance: 0.0)
    assert_equal(0.0, d.max_abs, 'head 0 must not see head 1 values')
  end

  def test_transformer_block_kv_sink_receives_the_key_and_value_blocks
    channels, heads, seq_len = 8, 2, 4
    x = XM::SFloat.new(seq_len, channels).seq * 0.03 - 0.5
    weights = block_weights(channels)
    captured = []
    trace = {}
    O.transformer_block(x, weights, batch_size: 1, seq_len: seq_len, num_heads: heads,
                        mask: O.causal_mask(seq_len), trace: trace,
                        kv_sink: ->(k, v) { captured << [k, v] })

    assert_equal(1, captured.size)
    keys, values = captured.first
    assert_equal([seq_len, channels], keys.shape)
    assert_equal([seq_len, channels], values.shape)
    qkv = trace['qkv']
    assert_equal(qkv[true, channels...(2 * channels)].to_a, keys.to_a)
    assert_equal(qkv[true, (2 * channels)...(3 * channels)].to_a, values.to_a)
  end
end
