# frozen_string_literal: true

require_relative 'test_helper'

# Stage 1 acceptance tests (docs/plans/PLAN-switch.md).
#
# The reference is transformers, taken with forward hooks on the real model by
# python/switch_dump.py. There is no C implementation of this architecture to
# compare against, so the token-exact gate the other models use is replaced by
# two checks: the routing decision has to match exactly, and the activations
# have to land inside a bound.
class TestSwitchEncoder < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/switch-base-8.safetensors', __dir__)
  DUMPS = {
    6 => File.expand_path('../data/switch-base-8_encoder_state.safetensors', __dir__),
    128 => File.expand_path('../data/switch-base-8_encoder_state_128.safetensors', __dir__)
  }.freeze

  # Twelve blocks of fp32 attention and feed forward over a range of about 3.
  # The worst seen is 3.7e-05 at 128 tokens on Numo; 5e-05 leaves room for the
  # backends to disagree with each other without hiding a real drift.
  TOLERANCE = 5.0e-5

  def setup
    # Before the omit: teardown runs either way, and an omitted test would
    # otherwise fail there instead of being skipped.
    @dumps = {}
    omit("#{MODEL} not found; run `rake prepare:switch`") unless File.exist?(MODEL)

    @checkpoint = NArrayLLM::Switch::Checkpoint.load(MODEL)
  end

  def teardown
    @dumps.each_value(&:close)
    @checkpoint&.close
  end

  def dump(length)
    path = DUMPS.fetch(length)
    omit("#{path} not found; run `rake prepare:switch`") unless File.exist?(path)

    @dumps[length] ||= NArrayLLM::Safetensors.new(path)
  end

  def ids(length)
    dump(length)['input_ids'].to_a.map(&:to_i)
  end

  data('6 トークン', 6)
  data('128 トークン', 128)
  def test_the_encoder_matches_transformers(length)
    reference = dump(length)['encoder_last_hidden_state']
    NArrayLLM::Switch::Encoder::ROUTERS.each do |router|
      encoder = NArrayLLM::Switch::Encoder.new(@checkpoint, router: router)
      worst = host((reference - encoder.forward(ids(length))).abs.max)
      assert_operator(worst, :<, TOLERANCE, "#{router}, #{length} トークン")
      notify(format('%s %d トークン: 最大 max|d| = %.3e (閾値 %.1e)',
                    router, length, worst, TOLERANCE))
    end
  end

  # The buckets are integer arithmetic over the positions, so nothing here is
  # allowed to be approximate. It caught a real difference: the reference
  # divides and then scales, and folding the two into one constant moves a
  # distance of exactly 64 one bucket down.
  data('6 トークン', 6)
  data('128 トークン', 128)
  def test_the_relative_bias_is_bit_exact(length)
    encoder = NArrayLLM::Switch::Encoder.new(@checkpoint)
    assert_equal(0.0, host((dump(length)['relative_bias'] - encoder.relative_bias(length)).abs.max))
  end

  # Which token went to which expert is an integer decision, so it can be
  # required to match exactly rather than within a bound.
  data('6 トークン', 6)
  data('128 トークン', 128)
  def test_the_routing_decision_matches_exactly(length)
    tokens = ids(length)
    NArrayLLM::Switch::Encoder::ROUTERS.each do |router|
      trace = {}
      NArrayLLM::Switch::Encoder.new(@checkpoint, router: router).forward(tokens, trace: trace)
      assert_equal(6, trace.size, 'sparse な層は 6 つ')
      trace.each do |name, got|
        want = dump(length)[name].reshape(tokens.size, @checkpoint.config.num_experts)
        assert_equal(0.0, host((got - got.class.cast(want)).abs.max), "#{router} #{name}")
      end
    end
  end

  def test_the_two_routers_answer_the_same_thing
    tokens = ids(6)
    dispatch = NArrayLLM::Switch::Encoder.new(@checkpoint, router: :dispatch).forward(tokens)
    dense = NArrayLLM::Switch::Encoder.new(@checkpoint, router: :dense).forward(tokens)
    worst = host((dispatch - dense).abs.max)
    assert_operator(worst, :<, TOLERANCE)
    notify(format('振り分ける版と全部走らせる版の差: %.3e', worst))
  end

  # config.json asks for a capacity of 64 and the paper drops what does not
  # fit, but transformers takes its cumsum over the axis of size one, so its
  # mask never fires. Matching the reference means not dropping either.
  def test_the_reference_never_drops_a_token
    tokens = ids(128)
    trace = {}
    NArrayLLM::Switch::Encoder.new(@checkpoint).forward(tokens, trace: trace)
    loads = trace.values.map { |one_hot| host(one_hot.sum(axis: 0).max) }
    assert_operator(loads.max, :>, @checkpoint.config.expert_capacity,
                    '容量を超えて 1 つの expert に集まる入力であること')
    trace.each_value { |one_hot| assert_equal(tokens.size.to_f, host(one_hot.sum)) }
    notify(format('1 つの expert に集まった最大 %d (容量 %d)、落ちたトークン 0',
                  loads.max, @checkpoint.config.expert_capacity))
  end

  def test_an_explicit_capacity_drops_what_does_not_fit
    tokens = ids(128)
    NArrayLLM::Switch::Encoder::ROUTERS.each do |router|
      trace = {}
      NArrayLLM::Switch::Encoder.new(@checkpoint, router: router, expert_capacity: 64)
                                .forward(tokens, trace: trace)
      dropped = trace.values.sum { |one_hot| tokens.size - host(one_hot.sum).round }
      assert_equal(16, dropped, router.to_s)
      trace.each_value do |one_hot|
        assert_operator(host(one_hot.sum(axis: 0).max), :<=, 64.0)
      end
    end
  end

  # max_index takes the first maximum, so a mask that kept every tie would
  # send a token to more than one expert and disagree with the dispatching
  # spelling. Ties do not arise with these weights, which is why the two
  # spellings agreed before this was checked directly.
  def test_a_tie_sends_the_token_to_one_expert_only
    experts = @checkpoint.config.num_experts
    probs = XF.zeros(3, experts)
    probs[0, 0..1] = XF[0.5, 0.5]
    probs[1, 1] = 0.7
    probs[2, true] = XF.ones(experts) / experts
    encoder = NArrayLLM::Switch::Encoder.new(@checkpoint)
    first = encoder.send(:first_maximum, probs, probs.max(axis: 1, keepdims: true))
    assert_equal([1.0] * 3, first.sum(axis: 1).to_a, '行ごとに 1 つだけ')
    assert_equal(1.0, host(first[0, 0]), '同点なら先頭')
    assert_equal(0.0, host(first[0, 1]))
    assert_equal(1.0, host(first[1, 1]), '同点でなければ最大')
    assert_equal(1.0, host(first[2, 0]), '全部同点でも先頭 1 つ')
  end

  # The file is F32 whatever DTYPE asks for. Landing its bytes in a two byte
  # type would read the first half of the buffer and answer without
  # complaining, which is a different number rather than a rounded one.
  def test_a_tensor_is_read_in_the_file_s_dtype
    store = NArrayLLM::Safetensors.new(MODEL)
    begin
      weight = store['decoder.final_layer_norm.weight']
      assert_equal(XM::SFloat, weight.class)
      assert_equal([768], weight.shape)
      assert_equal(XF, NArrayLLM::Ops.contiguous(weight).class)
    ensure
      store.close
    end
  end

  # cumsum synchronizes on cumo, and the dense spelling exists to avoid host
  # round trips, so the running count comes from a matmul instead.
  def test_the_running_count_block_is_strictly_lower
    block = NArrayLLM::Switch::Stack.strictly_lower(4)
    assert_equal([[0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0],
                  [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], block.to_a)
  end

  def test_an_unknown_router_is_rejected
    assert_raise_kind_of(NArrayLLM::Error) do
      NArrayLLM::Switch::Encoder.new(@checkpoint, router: :whatever)
    end
  end
end
