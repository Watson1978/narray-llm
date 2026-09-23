# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-training.md).
#
# Every backward is checked against a central difference of its own forward.
# The check runs in fp64 whatever the backend is: with h of 1e-6 the difference
# divides by 2e-6, so fp32 rounding in the forward would come back multiplied by
# five hundred thousand. That is not a hypothetical -- it is what the first run
# of these did, and it made a correct attention backward look 3.6e-2 wrong.
class TestBackward < Test::Unit::TestCase
  include TestHelper

  # The backend's own fp64, not Numo's: under GPU=1 the fused kernels and the
  # native batched dot are chosen from XF, and handing those a Numo array from
  # a Cumo build breaks before any gradient is compared.
  D = XM::DFloat
  BATCH = 2
  SEQ = 3
  CHANNELS = 4
  HEADS = 2
  VOCAB = 6
  ROWS = BATCH * SEQ
  H = 1e-6

  # Observed across all seven: at most 1.8e-9 relative. A wrong answer in this
  # same harness came back at 3.6e-2. This sits 55x above the noise and five
  # orders below the one mistake that has actually happened here.
  TOLERANCE = 1e-7

  def wave(*shape, scale: 1.0, shift: 0.0)
    count = shape.inject(:*)
    (D::Math.sin((D.new(count).seq * 1.7) + shift) * scale).reshape(*shape)
  end

  # reshape answers a copy in both backends (AGENTS.md), so the perturbation
  # goes through a flat subscript on the array itself.
  #
  # The saved value is read back to the host first. Cumo answers a subscript as
  # a zero dimensional NArray and that one is a *view*: keeping it and writing
  # through the same subscript changes what it reads, so the second perturbation
  # would start from the first one and the difference would come back exactly
  # half. Numo hands back a Float and never shows this.
  def numeric_grad(x)
    grad = x.class.zeros(*x.shape)
    x.size.times do |i|
      old = host(x[i]).to_f
      x[i] = old + H
      plus = host(yield).to_f
      x[i] = old - H
      minus = host(yield).to_f
      x[i] = old
      grad[i] = (plus - minus) / (2 * H)
    end
    grad
  end

  # Cumo answers a reduction as a zero dimensional NArray, not a Ruby Float
  # (AGENTS.md), so both ends come back to the host before they are compared.
  def assert_matches_numeric(analytic, numeric, label)
    scale = host(numeric.abs.max).to_f
    worst = host((analytic - numeric).abs.max).to_f
    relative = scale.zero? ? worst : worst / scale
    assert_operator(relative, :<, TOLERANCE, "#{label}: max|d| #{worst}, 値域 #{scale}")
  end

  def test_gelu_matches_a_central_difference
    x = wave(ROWS, CHANNELS, scale: 2.0)
    dout = wave(ROWS, CHANNELS, shift: 0.5)
    analytic = NArrayLLM::Backward.gelu(dout, x)
    numeric = numeric_grad(x) { (NArrayLLM::Ops.gelu(x) * dout).sum }
    assert_matches_numeric(analytic, numeric, 'gelu dinp')
  end

  def test_layernorm_matches_a_central_difference
    x = wave(ROWS, CHANNELS, scale: 1.5, shift: 3.0)
    weight = wave(CHANNELS, shift: 1.0) + 1.0
    bias = wave(CHANNELS, scale: 0.5, shift: 2.0)
    dout = wave(ROWS, CHANNELS, shift: 0.5)
    loss = -> { (NArrayLLM::Ops.layernorm(x, weight, bias) * dout).sum }

    dinp, dweight, dbias = NArrayLLM::Backward.layernorm(dout, x, weight)
    assert_matches_numeric(dinp, numeric_grad(x) { loss.call }, 'layernorm dinp')
    assert_matches_numeric(dweight, numeric_grad(weight) { loss.call }, 'layernorm dweight')
    assert_matches_numeric(dbias, numeric_grad(bias) { loss.call }, 'layernorm dbias')
  end

  def test_matmul_matches_a_central_difference
    out_channels = 5
    x = wave(ROWS, CHANNELS, scale: 1.1, shift: 6.0)
    weight_t = wave(CHANNELS, out_channels, scale: 0.7, shift: 4.0)
    bias = wave(out_channels, scale: 0.4, shift: 5.0)
    dout = wave(ROWS, out_channels, shift: 7.0)
    loss = -> { (NArrayLLM::Ops.linear(x, weight_t, bias) * dout).sum }

    dinp, dweight_t, dbias = NArrayLLM::Backward.matmul(dout, x, weight_t)
    assert_matches_numeric(dinp, numeric_grad(x) { loss.call }, 'matmul dinp')
    assert_matches_numeric(dweight_t, numeric_grad(weight_t) { loss.call }, 'matmul dweight_t')
    assert_matches_numeric(dbias, numeric_grad(bias) { loss.call }, 'matmul dbias')
  end

  def causal_mask
    mask = D.zeros(SEQ, SEQ)
    SEQ.times { |i| SEQ.times { |j| mask[i, j] = j > i ? -1.0e9 : 0.0 } }
    mask
  end

  def test_attention_matches_a_central_difference
    qkv = wave(ROWS, 3 * CHANNELS, scale: 0.8, shift: 8.0)
    dout = wave(ROWS, CHANNELS, shift: 9.0)
    mask = causal_mask
    heads = { batch_size: BATCH, seq_len: SEQ, num_heads: HEADS }

    _, weights = NArrayLLM::Ops.attention_with_weights(qkv, mask: mask, **heads)
    analytic = NArrayLLM::Backward.attention(dout, qkv, weights, **heads)
    numeric = numeric_grad(qkv) do
      (NArrayLLM::Ops.attention(qkv, mask: mask, **heads) * dout).sum
    end
    assert_matches_numeric(analytic, numeric, 'attention dqkv')
  end

  def test_crossentropy_softmax_matches_a_central_difference
    logits = wave(ROWS, VOCAB, scale: 2.0, shift: 10.0)
    targets = [0, 3, 5, 1, 4, 2]
    loss = lambda do
      probs = NArrayLLM::Ops.softmax_rows(logits)
      hot = NArrayLLM::Backward.one_hot(D, targets, VOCAB)
      -(D::Math.log((probs * hot).sum(axis: 1))).sum / ROWS
    end

    analytic = NArrayLLM::Backward.crossentropy_softmax(
      NArrayLLM::Ops.softmax_rows(logits), targets
    )
    assert_matches_numeric(analytic, numeric_grad(logits) { loss.call }, 'crossentropy dlogits')
  end

  def test_encoder_matches_a_central_difference
    wte = wave(VOCAB, CHANNELS, scale: 0.9, shift: 11.0)
    wpe = wave(SEQ, CHANNELS, scale: 0.6, shift: 12.0)
    ids = [2, 0, 5, 3, 5, 1]
    dout = wave(ROWS, CHANNELS, shift: 13.0)
    embed = lambda do
      rows = NArrayLLM::Backward.one_hot(D, ids, VOCAB).dot(wte)
      (rows.reshape(BATCH, SEQ, CHANNELS) + wpe).reshape(ROWS, CHANNELS)
    end

    dwte, dwpe = NArrayLLM::Backward.encoder(dout, ids, vocab_size: VOCAB,
                                             batch_size: BATCH, seq_len: SEQ)
    assert_matches_numeric(dwte, numeric_grad(wte) { (embed.call * dout).sum }, 'encoder dwte')
    assert_matches_numeric(dwpe, numeric_grad(wpe) { (embed.call * dout).sum }, 'encoder dwpe')
    # Not vacuous: one id repeats, so dwte has to accumulate two rows into one.
    assert_operator(ids.size, :>, ids.uniq.size)
  end

  def test_residual_sends_the_whole_gradient_both_ways
    dout = wave(ROWS, CHANNELS)
    one, two = NArrayLLM::Backward.residual(dout)
    assert_equal(dout.to_a, one.to_a)
    assert_equal(dout.to_a, two.to_a)
  end
end
