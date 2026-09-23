# frozen_string_literal: true

module NArrayLLM
  module Switch
    # What the encoder and the decoder share. T5 lineage, so: no bias
    # anywhere, a layer norm that does not centre, a relative position bias
    # added to the scores, and no scaling of the query-key product
    # (SwitchTransformersAttention sets scaling to 1.0).
    #
    # A stack computes its relative bias once, on its first block, and every
    # block after that uses the same one.
    class Stack
      # Two ways to spend the sparse layers. dispatch follows transformers and
      # sends each token to its one expert, which needs the expert numbers on
      # the host. dense runs all eight experts over every token and multiplies
      # by a mask, which never leaves the device and costs eight times the
      # work. Which one wins is the question this model is here to answer.
      ROUTERS = %i[dispatch dense].freeze

      attr_reader :config, :router, :expert_capacity

      # expert_capacity is nil by default, meaning no limit, because that is
      # what the reference does. config.json asks for 64 and the paper has the
      # limit, but transformers 5.16.1 takes its cumsum over the axis of size
      # one (`torch.cumsum(expert_index, dim=-2)` on a [tokens, 1, experts]
      # one hot), so token_priority never exceeds 1 and the mask never drops
      # anything. Reproduced on the real model: 600 tokens, 222 of them on one
      # expert, nothing dropped. Pass an integer here to apply the limit and
      # step away from the reference on purpose.
      # T5's residual stream is large: with these weights it passes 1.9e+04 by
      # the fifth encoder block and reaches Inf in the sixth. Nothing in the
      # order of operations fixes that, so fp16 is refused rather than
      # answered with a sequence of zeros. bf16 keeps fp32's exponent and runs.
      NEEDS_EXPONENT_RANGE = 'Switch needs fp32 or bf16, not %<dtype>s: the ' \
                             'residual passes 1.9e+04 by the fifth block.'

      def initialize(checkpoint, router: :dispatch, expert_capacity: nil)
        raise Error, "unknown router #{router.inspect}" unless ROUTERS.include?(router)
        raise Error, format(NEEDS_EXPONENT_RANGE, dtype: NArrayLLM.dtype_name) if NArrayLLM.fp16?

        @config = checkpoint.config
        @router = router
        @expert_capacity = expert_capacity
        @checkpoint = checkpoint
        prepare(checkpoint)
        # Cross attention has no position bias at all, so a block of zeros
        # stands in for one (SwitchTransformersAttention builds the same).
        @zero_bias = XF.zeros(@config.num_heads, 1, 1)
        @before = Stack.strictly_lower(@config.num_experts)
      end

      # before[k, j] is 1 when k < j, so x.dot(before) is the running count of
      # everything ahead of each column without a scan.
      def self.strictly_lower(size)
        i = XF.new(size, 1).seq
        j = XF.new(1, size).seq
        (j - i).clip(0.0, 1.0).ceil
      end

      # _relative_position_bucket. Bidirectional halves the buckets and keeps
      # the sign in the top half; one directional folds everything that lies
      # ahead onto zero.
      #
      # The three factors are multiplied in the reference's order, divide then
      # scale, and not folded into one constant. At a distance of 64 the exact
      # answer is 6, and `log(d / 8) * (8 / log(16))` lands on
      # 5.999999999999999, which truncates to 5 and moves the pair one bucket
      # down. With 128 tokens that is the 64 pairs on one diagonal.
      def self.buckets(query_length, key_length, config, bidirectional:, offset: 0)
        total = config.relative_attention_num_buckets
        half = bidirectional ? total / 2 : total
        exact = half / 2
        span = Math.log(config.relative_attention_max_distance.fdiv(exact))
        out = Array.new(query_length * key_length)
        query_length.times do |q|
          key_length.times do |k|
            distance = k - (q + offset)
            bucket = 0
            if bidirectional
              bucket = half if distance.positive?
              distance = distance.abs
            else
              distance = -[distance, 0].min
            end
            bucket += if distance < exact
                        distance
                      else
                        large = (Math.log(distance.fdiv(exact)) / span * (half - exact)).to_i
                        [exact + large, half - 1].min
                      end
            out[(q * key_length) + k] = bucket
          end
        end
        out
      end

      private

      # The bucket table depends only on the positions, so it is built on the
      # host and the rows come out with a Ruby Array subscript, which does not
      # synchronize the way an NArray subscript would (AGENTS.md).
      def bias_from(table, query_length, key_length, bidirectional:, offset: 0)
        rows = table[Stack.buckets(query_length, key_length, @config,
                                   bidirectional: bidirectional, offset: offset), true]
        Ops.contiguous(rows.reshape(query_length, key_length, @config.num_heads).transpose(2, 0, 1))
      end

      # mean of squares, no centring, no bias (SwitchTransformersLayerNorm).
      def norm(x, weight)
        Ops.rmsnorm(x, weight, eps: @config.layer_norm_epsilon)
      end

      # h: [queries, d_model], keys and values already laid out head major as
      # [heads, keys, d_kv]. bias is [heads, queries, keys] and mask, when
      # given, is added to the scores before the softmax.
      #
      # Head major is what keeps the copies down. A column span of a
      # [tokens, inner] array is a non-contiguous view, so the old spelling
      # copied Q, K and V once per head; here one copy lays out the whole
      # tensor and each head is a contiguous slice of it. For the decoder that
      # is 72 copies a block a token down to 4.
      def attend(h, w, prefix, keys, values, bias, mask = nil)
        q = heads_major(h.dot(w[:"#{prefix}q_t"]))
        parts = Array.new(@config.num_heads) do |head|
          scores = q[head, true, true].dot(keys[head, true, true].transpose) + bias[head, true, true]
          scores += mask unless mask.nil?
          Ops.softmax_rows(scores).dot(values[head, true, true])
        end
        Ops.contiguous(XF.hstack(parts)).dot(w[:"#{prefix}o_t"])
      end

      # [tokens, inner] to [heads, tokens, d_kv], contiguous. reshape! rather
      # than reshape because reshape copies and the caller owns this array.
      def heads_major(x)
        rows = x.shape[0]
        Ops.contiguous(x.reshape!(rows, @config.num_heads, @config.d_kv).transpose(1, 0, 2))
      end

      def project_kv(x, w, prefix)
        [x.dot(w[:"#{prefix}k_t"]), x.dot(w[:"#{prefix}v_t"])]
      end

      def project_kv_heads(x, w, prefix)
        project_kv(x, w, prefix).map { |a| heads_major(a) }
      end

      def feed_forward(h, w, prof, trace, index)
        return dense(h, w) unless w[:experts]

        mixture(h, w, prof, trace, index)
      end

      def dense(h, w)
        relu(h.dot(w[:wi_t])).dot(w[:wo_t])
      end

      def relu(x)
        x.clip(0.0, nil)
      end

      # The router is computed in fp32 whatever the forward pass runs in, which
      # is what config.json's router_dtype asks for and what the reference
      # does (`hidden_states.to(self.dtype)` before the classifier). Its weight
      # is kept in fp32 for the same reason. Under DTYPE=fp32 both casts are
      # the identity; under bf16 they are the difference between following the
      # reference and not.
      def mixture(h, w, prof, trace = nil, index = nil)
        probs = prof.section(:router) do
          XF.cast(Ops.softmax_rows(XM::SFloat.cast(h).dot(w[:router_t])))
        end
        if @router == :dense
          mixture_dense(h, w, probs, prof, trace, index)
        else
          mixture_dispatch(h, w, probs, prof, trace, index)
        end
      end

      # transformers' shape: one expert per token, and a token past the
      # expert's capacity goes nowhere and reaches the output through the
      # residual alone.
      def mixture_dispatch(h, w, probs, prof, trace = nil, index = nil)
        tokens = h.shape[0]
        gate = probs.max(axis: 1, keepdims: true)
        chosen = prof.section(:readback) { Stack.argmax_rows(probs, tokens) }
        out = XF.zeros(tokens, @config.d_model)
        buckets = Array.new(@config.num_experts) { [] }
        chosen.each_with_index do |expert, token|
          seats = buckets[expert]
          seats << token if @expert_capacity.nil? || seats.size < @expert_capacity
        end
        buckets.each_with_index do |seats, expert|
          next if seats.empty?

          rows = Ops.contiguous(h[seats, true])
          y = prof.section(:expert) { dense(rows, w[:experts][expert]) }
          out[seats, true] = y * Ops.contiguous(gate[seats, true])
        end
        record(trace, index) { one_hot_from(buckets, tokens) }
        out
      end

      # The same answer with no host round trip: every expert sees every
      # token, and a mask built with arithmetic keeps the one that was chosen.
      def mixture_dense(h, w, probs, prof, trace = nil, index = nil)
        top = probs.max(axis: 1, keepdims: true)
        selected = first_maximum(probs, top)
        within = capacity_mask(selected)
        weights = selected * within * top
        out = XF.zeros(*h.shape)
        @config.num_experts.times do |expert|
          y = prof.section(:expert) { dense(h, w[:experts][expert]) }
          out += y * Ops.contiguous(weights[true, expert...(expert + 1)])
        end
        record(trace, index) { selected * within }
        out
      end

      # 1 on the expert holding the row maximum, and on the first one when
      # several tie. max_index takes the first, so a mask that keeps every tie
      # would send a token to more than one expert and disagree with the
      # dispatching spelling.
      #
      # The running count comes from a matmul with a strictly lower triangular
      # block of ones rather than from cumsum, which synchronizes on cumo (its
      # own source says FIXME). cumsum here took the dense spelling from one
      # host round trip a token to seven, which is the whole point of it.
      def first_maximum(probs, top)
        hit = 1.0 - (probs - top).abs.clip(0.0, 1.0).ceil
        hit * (1.0 - hit.dot(@before).clip(0.0, 1.0).ceil)
      end

      # 1 where this token is inside its expert's capacity. Without a limit
      # every seat is inside, which is what the reference computes.
      def capacity_mask(selected)
        return XF.ones(*selected.shape) if @expert_capacity.nil?

        # seat <= capacity by arithmetic: past the capacity the ceil is 1.
        1.0 - (selected.cumsum(axis: 0) - @expert_capacity).clip(0.0, 1.0).ceil
      end

      def record(trace, index)
        return if trace.nil? || index.nil?

        trace["block.#{index}.router.expert_index"] = yield
      end

      def one_hot_from(buckets, tokens)
        out = XF.zeros(tokens, @config.num_experts)
        buckets.each_with_index do |seats, expert|
          next if seats.empty?

          out[seats, expert] = XF.ones(seats.size)
        end
        out
      end

      # max_index answers a flattened running number (AGENTS.md), so the row
      # offsets come off before this reaches the host.
      def self.argmax_rows(probs, tokens)
        flat = probs.max_index(axis: 1)
        experts = probs.shape[1]
        offsets = flat.class.new(tokens).seq * experts
        (flat - offsets).to_a
      end

      # Stored [out, in]; every matmul here wants [in, out].
      def attention_weights(checkpoint, name, prefix, kind)
        t = ->(part) { Ops.contiguous(checkpoint["#{name}.#{kind}.#{part}.weight"].transpose) }
        { :"#{prefix}q_t" => t.call('q'), :"#{prefix}k_t" => t.call('k'),
          :"#{prefix}v_t" => t.call('v'), :"#{prefix}o_t" => t.call('o') }
      end

      def feed_forward_weights(checkpoint, prefix, sparse)
        t = ->(name) { Ops.contiguous(checkpoint["#{prefix}.#{name}"].transpose) }
        return { wi_t: t.call('wi.weight'), wo_t: t.call('wo.weight') } unless sparse

        # fp32 even under DTYPE=bf16: see mixture.
        router = checkpoint["#{prefix}.router.classifier.weight"].transpose
        { router_t: XM::SFloat.cast(router).dup,
          experts: Array.new(@config.num_experts) do |expert|
            { wi_t: t.call("experts.expert_#{expert}.wi.weight"),
              wo_t: t.call("experts.expert_#{expert}.wo.weight") }
          end }
      end
    end
  end
end
