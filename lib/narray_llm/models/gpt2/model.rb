# frozen_string_literal: true

module NArrayLLM
  # GPT-2 固有。共有するものは lib/narray_llm/ の直下にある。
  module GPT2
    # GPT-2 forward pass, following gpt2_forward in llm.c's train_gpt2.c.
    class Model
      # The four biases a decode block adds, in the order they are laid out in
      # the buffer decode_linear slices.
      BIAS_KEYS = %i[qkvb attprojb fcb fcprojb].freeze

      attr_reader :config

      def self.load(path)
        new(Checkpoint.load(path))
      end

      def initialize(checkpoint)
        @config = checkpoint.config
        prepare(checkpoint)
      end

      # tokens: [B, T] of ids (NArray or nested Array). Returns logits [B, T, V],
      # or [B, 1, V] for the last position only, which is all greedy decoding
      # needs and turns the V-wide unembedding from T rows into one.
      def forward(tokens, trace: nil, prof: Profiler::NULL, last_only: false, cache: nil)
        ids = self.class.normalize_tokens(tokens)
        batch_size, seq_len = ids.shape
        if seq_len > @config.max_seq_len
          raise Error, "sequence length #{seq_len} exceeds max_seq_len #{@config.max_seq_len}"
        end
        if cache && cache.batch_size != batch_size
          raise Error, "a cache of batch size #{cache.batch_size} cannot take #{batch_size} rows"
        end

        x = prof.section(:embed) { embed(ids, batch_size, seq_len) }
        trace['encoded'] = x if trace
        mask = causal_mask(seq_len)

        @layers.each_with_index do |weights, layer|
          sink = cache && lambda do |k, v|
            cache.append(layer, cache_rows(k, batch_size, seq_len),
                         cache_rows(v, batch_size, seq_len))
          end
          x = prof.section(:block) do
            Ops.transformer_block(x, weights, batch_size: batch_size, seq_len: seq_len,
                                  num_heads: @config.num_heads, mask: mask,
                                  trace: trace, prefix: "L#{layer}/", prof: prof, kv_sink: sink)
          end
        end

        x = prof.section(:final_norm) { Ops.layernorm(x, @lnfw, @lnfb) }
        trace['lnf'] = x if trace
        x = prof.section(:unembed) { last_position(x, batch_size, seq_len) } if last_only
        # Unembedding shares the token embedding table. wte is [V, C] and this
        # needs [C, V], so the transpose was taken once at load time.
        logits = prof.section(:unembed) { x.dot(@wte_t) }
        trace['logits'] = logits if trace

        logits.reshape!(batch_size, last_only ? 1 : seq_len, @config.vocab_size)
      end

      # Bytes held by the prepared weights. Not the checkpoint size: the padded
      # vocabulary rows are dropped and wte is kept in both orientations.
      def parameter_bytes
        tensors = [@wte, @wte_t, @wpe, @lnfw, @lnfb] + @layers.flat_map(&:values)
        XF::ELEMENT_BYTE_SIZE * tensors.sum(&:size)
      end

      def new_cache(batch_size: 1)
        KVCache.new(num_layers: @config.num_layers, max_seq_len: @config.max_seq_len,
                    channels: @config.channels, batch_size: batch_size)
      end

      # Phase 1 of cached generation: run the prompt through the ordinary forward
      # pass, filling every layer's K and V. Returns the last position's logits.
      def prefill(tokens, cache:, prof: Profiler::NULL)
        cache.reset
        forward(tokens, last_only: true, prof: prof, cache: cache)
      end

      # Phase 2: one new token against the cache. token_id is a Ruby Integer that
      # argmax already read back, so indexing wte with it costs no device readback
      # (AGENTS.md: only an NArray index forces a synchronizing gather).
      #
      # An Array of ids decodes one step for each sequence in a batched cache.
      # Every sequence sits at the same position, so there is one position and
      # not one per sequence (docs/plans/PLAN-batch.md).
      def decode(token_id, position, cache:, prof: Profiler::NULL)
        if position >= @config.max_seq_len
          raise Error, "position #{position} exceeds max_seq_len #{@config.max_seq_len}"
        end

        ids = token_id.is_a?(Array) ? token_id : [token_id]
        unless ids.size == cache.batch_size
          raise Error, "got #{ids.size} token ids for a cache of batch size #{cache.batch_size}"
        end

        refill_biases(ids.size)
        x = prof.section(:embed) { decode_embed(ids, position) }
        @layers.each_with_index do |weights, layer|
          x = prof.section(:block) { decode_block(x, weights, layer, cache, prof) }
        end
        x = prof.section(:final_norm) { Ops.layernorm(x, @lnfw, @lnfb) }
        logits = prof.section(:unembed) { x.dot(@wte_t) }
        logits.reshape!(ids.size, 1, @config.vocab_size)
      end

      # Mean cross-entropy over every position (train_gpt2.c:486 crossentropy_forward,
      # averaged into mean_loss in gpt2_forward). logsumexp - logit[target] is the
      # same quantity as -log(softmax(logits)[target]) without forming probs.
      def loss(logits, targets)
        vocab = @config.vocab_size
        flat = logits.reshape(logits.size / vocab, vocab)
        ids = self.class.normalize_tokens(targets)

        shift = flat.max(axis: 1, keepdims: true)
        logsumexp = shift + XM::NMath.log(XM::NMath.exp(flat - shift).sum(axis: 1, keepdims: true))
        picked = (flat * Ops.one_hot(ids.to_a, vocab)).sum(axis: 1, keepdims: true)
        NArrayLLM.scalar((logsumexp - picked).mean)
      end

      def self.normalize_tokens(tokens)
        return tokens if tokens.is_a?(HM::Int32) && tokens.ndim == 2

        rows = tokens.to_a
        rows = [rows] unless rows.first.is_a?(Array)
        HM::Int32.cast(rows)
      end

      private

      # One row of wte plus one row of wpe, per sequence. No one-hot matrix: with
      # a single token the GEMM would be a [1, V] x [V, C] product to pick one
      # row. The ids are a Ruby Array, so the gather does not synchronize
      # (AGENTS.md).
      def decode_embed(ids, position)
        ids.each do |id|
          unless id.is_a?(Integer) && id >= 0 && id < @config.vocab_size
            raise Error, "invalid token id #{id.inspect}"
          end
        end
        return (@wte[ids.first, true] + @wpe[position, true])
               .reshape!(1, @config.channels) if ids.size == 1

        Ops.gather_rows(@wte, ids) + @wpe[position, true]
      end

      def decode_block(x, weights, layer, cache, prof)
        channels = @config.channels
        ln1 = prof.section(:layernorm) { Ops.layernorm(x, weights[:ln1w], weights[:ln1b]) }
        qkv = prof.section(:gemm) { decode_linear(ln1, weights, layer, :qkvw_t, :qkvb) }
        prof.section(:cache) do
          cache.append(layer, qkv[true, channels...(2 * channels)],
                       qkv[true, (2 * channels)...(3 * channels)])
        end
        keys, values = cache.view(layer)
        attn = Ops.decode_attention(Ops.row_slice(qkv, 0...channels), keys, values,
                                    num_heads: @config.num_heads, prof: prof)
        attproj = prof.section(:gemm) { decode_linear(attn, weights, layer, :attprojw_t, :attprojb) }
        residual2 = prof.section(:residual) { x + attproj }

        ln2 = prof.section(:layernorm) { Ops.layernorm(residual2, weights[:ln2w], weights[:ln2b]) }
        fch = prof.section(:gemm) { decode_linear(ln2, weights, layer, :fcw_t, :fcb) }
        activated = prof.section(:gelu) { Ops.gelu(fch) }
        fcproj = prof.section(:gemm) { decode_linear(activated, weights, layer, :fcprojw_t, :fcprojb) }
        prof.section(:residual) { residual2 + fcproj }
      end

      # The bias broadcast to one row per sequence is exactly the shape the GEMM
      # wants for C. The product overwrites it, which is why every bias in the
      # model sits in one buffer that refill_biases restores once per token
      # rather than once per call. Numo has no gemm, so that backend keeps the
      # separate add.
      def decode_linear(x, weights, layer, weight_key, bias_key)
        return Ops.linear(x, weights[weight_key], weights[bias_key]) if @bias_master.nil?

        Ops.linear_into(x, weights[weight_key], @bias_views[[layer, bias_key]])
      end

      # The scratch is the GEMM's C, so it needs one row per sequence. Laid out
      # so that each bias occupies a contiguous [rows, size] block: cumo's gemm
      # refuses a strided C, and slicing columns out of a [rows, total] buffer
      # gives exactly that past one row. Sized on first use and rebuilt only
      # when the batch changes.
      def refill_biases(rows)
        return if @bias_layout.nil?

        if @bias_rows == rows
          fill_bias_master if @bias_stale
        else
          build_bias_buffers(rows)
        end
        @bias_scratch.store(@bias_master)
      end

      def build_bias_buffers(rows)
        total = @bias_layout.sum { |(_, _), size| size }
        @bias_master = XF.zeros(rows * total)
        @bias_scratch = XF.zeros(rows * total)
        @bias_views = {}
        at = 0
        @bias_layout.each do |key, size|
          span = at...(at + (rows * size))
          @bias_views[key] = @bias_scratch[span].reshape!(rows, size)
          at += rows * size
        end
        @bias_rows = rows
        fill_bias_master
      end

      # The master holds a copy of every bias, so an update to the parameters
      # has to be brought over. Training marks it and the next decode pays it.
      def fill_bias_master
        at = 0
        @bias_layout.each do |key, size|
          span = at...(at + (@bias_rows * size))
          @bias_master[span].reshape!(@bias_rows, size).store(@weights_for_bias[key])
          at += @bias_rows * size
        end
        @bias_stale = false
      end

      def invalidate_bias_master
        @bias_stale = true
      end

      def prepare_biases
        return unless Ops::GEMM_WITH_C

        @bias_layout = []
        @weights_for_bias = {}
        @layers.each_with_index do |weights, layer|
          BIAS_KEYS.each do |key|
            @bias_layout << [[layer, key], weights[key].size]
            @weights_for_bias[[layer, key]] = weights[key]
          end
        end
      end

      # The block hands over [B*T, C] laid out batch major, and the cache wants
      # [T, B, C]. With one sequence the two are the same rows in the same
      # order, so nothing is moved.
      def cache_rows(rows, batch_size, seq_len)
        return rows if batch_size == 1

        Ops.contiguous(rows.reshape(batch_size, seq_len, rows.shape[1]).transpose(1, 0, 2))
      end

      def last_position(x, batch_size, seq_len)
        Ops.contiguous(x.reshape(batch_size, seq_len, @config.channels)[true, seq_len - 1, true])
      end

      # Built once at max_seq_len and sliced, so a generation loop does not rebuild
      # it every step. The slice is a strided view, so it is copied before use.
      def causal_mask(seq_len)
        @full_mask ||= Ops.causal_mask(@config.max_seq_len)
        Ops.contiguous(@full_mask[0...seq_len, 0...seq_len])
      end

      def embed(ids, batch_size, seq_len)
        channels = @config.channels
        tokens = Ops.gather_rows(@wte, ids.to_a)
        positions = @wpe[0...seq_len, true]
        (tokens.reshape!(batch_size, seq_len, channels) + positions)
          .reshape!(batch_size * seq_len, channels)
      end

      def prepare(checkpoint)
        # llm.c keeps Vp rows in wte but never uses the padding (softmax and
        # crossentropy both stop at V), so drop it once here.
        @wte = Ops.contiguous(checkpoint[:wte][0...@config.vocab_size, true])
        @wte_t = Ops.contiguous(@wte.transpose)
        @wpe = Ops.contiguous(checkpoint[:wpe])
        @lnfw = Ops.contiguous(checkpoint[:lnfw])
        @lnfb = Ops.contiguous(checkpoint[:lnfb])
        @layers = Array.new(@config.num_layers) { |l| layer_weights(checkpoint, l) }
        prepare_biases
      end

      # Weights are stored [out, in]; every GEMM here wants [in, out], so transpose
      # once at load time rather than handing dot() a non-contiguous view per call.
      def layer_weights(checkpoint, layer)
        {
          ln1w: Ops.contiguous(checkpoint[:ln1w][layer, true]),
          ln1b: Ops.contiguous(checkpoint[:ln1b][layer, true]),
          qkvw_t: Ops.contiguous(checkpoint[:qkvw][layer, true, true].transpose),
          qkvb: Ops.contiguous(checkpoint[:qkvb][layer, true]),
          attprojw_t: Ops.contiguous(checkpoint[:attprojw][layer, true, true].transpose),
          attprojb: Ops.contiguous(checkpoint[:attprojb][layer, true]),
          ln2w: Ops.contiguous(checkpoint[:ln2w][layer, true]),
          ln2b: Ops.contiguous(checkpoint[:ln2b][layer, true]),
          fcw_t: Ops.contiguous(checkpoint[:fcw][layer, true, true].transpose),
          fcb: Ops.contiguous(checkpoint[:fcb][layer, true]),
          fcprojw_t: Ops.contiguous(checkpoint[:fcprojw][layer, true, true].transpose),
          fcprojb: Ops.contiguous(checkpoint[:fcprojb][layer, true])
        }
      end
    end
  end
end
