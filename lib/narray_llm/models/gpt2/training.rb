# frozen_string_literal: true

module NArrayLLM
  module GPT2
    # Training reopens the model rather than wrapping it: the backward pass
    # needs the same weights the forward holds, and they are not public.
    #
    # The order of operations follows gpt2_forward and gpt2_backward in llm.c's
    # train_gpt2.c. Every activation the backward reads is kept; the two that
    # llm.c also stores and this does not are the layernorm means and inverse
    # deviations, which Backward.layernorm recomputes, and preatt, which nothing
    # reads.
    class Model
      # Answers [mean_loss, activations]. targets is [B, T] of ids.
      def forward_train(tokens, targets)
        ids = self.class.normalize_tokens(tokens)
        batch_size, seq_len = ids.shape
        acts = { ids: ids.to_a.flatten, targets: self.class.normalize_tokens(targets).to_a.flatten,
                 batch_size: batch_size, seq_len: seq_len, layers: [] }

        x = embed(ids, batch_size, seq_len)
        mask = causal_mask(seq_len)
        @layers.each do |weights|
          x, saved = train_block(x, weights, batch_size, seq_len, mask)
          acts[:layers] << saved
        end

        acts[:final] = x
        acts[:lnf] = Ops.layernorm(x, @lnfw, @lnfb)
        acts[:probs] = Ops.softmax_rows(acts[:lnf].dot(@wte_t))
        [mean_loss(acts[:probs], acts[:targets]), acts]
      end

      # Answers a Hash keyed like Checkpoint::TENSOR_NAMES, each in the layout
      # the checkpoint uses so it can be compared with llm.c's expected_grads.
      def backward(acts)
        grads = zero_grads
        rows = acts[:batch_size] * acts[:seq_len]

        dlogits = Backward.crossentropy_softmax(acts[:probs], acts[:targets])
        dlnf, dwte_t, = Backward.matmul(dlogits, acts[:lnf], @wte_t, bias: false)
        grads[:wte] = Ops.contiguous(dwte_t.transpose)
        dx, grads[:lnfw], grads[:lnfb] = Backward.layernorm(dlnf, acts[:final], @lnfw)

        (@layers.size - 1).downto(0) do |layer|
          dx = backward_block(dx, acts[:layers][layer], @layers[layer], grads, layer,
                              acts[:batch_size], acts[:seq_len])
        end

        dwte, dwpe = Backward.encoder(dx, acts[:ids], vocab_size: @config.vocab_size,
                                      batch_size: acts[:batch_size], seq_len: acts[:seq_len])
        grads[:wte] = grads[:wte] + dwte
        grads[:wpe][0...acts[:seq_len], true] = grads[:wpe][0...acts[:seq_len], true] + dwpe
        raise Error, "backward saw #{rows} rows and no targets" if acts[:targets].empty?

        grads
      end

      # Yields [key, gradient, parameter] for every parameter, with the gradient
      # turned into the orientation the parameter is stored in. The weights are
      # kept transposed here and llm.c keeps them the other way round, so four
      # matrices a layer are transposed back on the way in.
      def each_parameter(grads)
        yield [:wte], grads[:wte], @wte
        yield [:wpe], grads[:wpe], @wpe
        @layers.each_with_index do |weights, layer|
          %i[ln1w ln1b qkvb attprojb ln2w ln2b fcb fcprojb].each do |name|
            yield [name, layer], grads[name][layer, true], weights[name]
          end
          { qkvw: :qkvw_t, attprojw: :attprojw_t, fcw: :fcw_t, fcprojw: :fcprojw_t }
            .each do |name, stored|
            yield [name, layer], Ops.contiguous(grads[name][layer, true, true].transpose),
                  weights[stored]
          end
        end
        yield [:lnfw], grads[:lnfw], @lnfw
        yield [:lnfb], grads[:lnfb], @lnfb
      end

      # The token table is held in both orientations and only one of them is a
      # parameter, so the other is rebuilt after an update. The decode path
      # keeps its own copy of every bias, which goes stale the same way.
      def refresh_derived
        @wte_t.store(@wte.transpose)
        invalidate_bias_master
      end

      private

      def train_block(x, weights, batch_size, seq_len, mask)
        saved = { inp: x }
        saved[:ln1] = Ops.layernorm(x, weights[:ln1w], weights[:ln1b])
        saved[:qkv] = Ops.linear(saved[:ln1], weights[:qkvw_t], weights[:qkvb])
        saved[:atty], saved[:att] =
          Ops.attention_with_weights(saved[:qkv], batch_size: batch_size, seq_len: seq_len,
                                     num_heads: @config.num_heads, mask: mask)
        attproj = Ops.linear(saved[:atty], weights[:attprojw_t], weights[:attprojb])
        saved[:res2] = x + attproj

        saved[:ln2] = Ops.layernorm(saved[:res2], weights[:ln2w], weights[:ln2b])
        saved[:fch] = Ops.linear(saved[:ln2], weights[:fcw_t], weights[:fcb])
        saved[:gelu] = Ops.gelu(saved[:fch])
        fcproj = Ops.linear(saved[:gelu], weights[:fcprojw_t], weights[:fcprojb])
        [saved[:res2] + fcproj, saved]
      end

      # The mirror image of train_block, read bottom to top. Both residuals send
      # the whole gradient two ways, which is why dres2 and the answer are sums.
      def backward_block(dout, saved, weights, grads, layer, batch_size, seq_len)
        dgelu, dfcprojw_t, dfcprojb = Backward.matmul(dout, saved[:gelu], weights[:fcprojw_t])
        accumulate(grads, :fcprojw, layer, Ops.contiguous(dfcprojw_t.transpose))
        accumulate(grads, :fcprojb, layer, dfcprojb)

        dfch = Backward.gelu(dgelu, saved[:fch])
        dln2, dfcw_t, dfcb = Backward.matmul(dfch, saved[:ln2], weights[:fcw_t])
        accumulate(grads, :fcw, layer, Ops.contiguous(dfcw_t.transpose))
        accumulate(grads, :fcb, layer, dfcb)

        dres2, dln2w, dln2b = Backward.layernorm(dln2, saved[:res2], weights[:ln2w])
        accumulate(grads, :ln2w, layer, dln2w)
        accumulate(grads, :ln2b, layer, dln2b)
        dres2 = dres2 + dout

        datty, dattprojw_t, dattprojb = Backward.matmul(dres2, saved[:atty], weights[:attprojw_t])
        accumulate(grads, :attprojw, layer, Ops.contiguous(dattprojw_t.transpose))
        accumulate(grads, :attprojb, layer, dattprojb)

        dqkv = Backward.attention(datty, saved[:qkv], saved[:att], batch_size: batch_size,
                                  seq_len: seq_len, num_heads: @config.num_heads)
        dln1, dqkvw_t, dqkvb = Backward.matmul(dqkv, saved[:ln1], weights[:qkvw_t])
        accumulate(grads, :qkvw, layer, Ops.contiguous(dqkvw_t.transpose))
        accumulate(grads, :qkvb, layer, dqkvb)

        dinp, dln1w, dln1b = Backward.layernorm(dln1, saved[:inp], weights[:ln1w])
        accumulate(grads, :ln1w, layer, dln1w)
        accumulate(grads, :ln1b, layer, dln1b)
        dres2 + dinp
      end

      def accumulate(grads, name, layer, value)
        target = grads[name]
        target[layer, false] = target[layer, false] + value
      end

      def mean_loss(probs, targets)
        hot = Backward.one_hot(probs.class, targets, @config.vocab_size)
        NArrayLLM.scalar(-XM::NMath.log((probs * hot).sum(axis: 1)).mean).to_f
      end

      def zero_grads
        shapes = Checkpoint.tensor_shapes(@config)
        Checkpoint::TENSOR_NAMES.each_with_object({}) do |name, acc|
          shape = shapes.fetch(name)
          # wte is stored at the real vocabulary size here: the model drops the
          # padded rows at load, and llm.c never writes a gradient into them.
          shape = [@config.vocab_size, shape[1]] if name == :wte
          acc[name] = XF.zeros(*shape)
        end
      end
    end
  end
end
