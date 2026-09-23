# frozen_string_literal: true

module NArrayLLM
  # Per-layer K and V, preallocated at [max_seq_len, B, C] so a decode step
  # writes one row per sequence instead of recomputing the whole sequence.
  #
  # Time is the outer axis, not the batch. A growing slice of it is contiguous
  # whatever B is, which is what lets decode_attention reshape the view in
  # place; [B, max_seq_len, C] would hand out a strided view and reshape!
  # refuses those outright. The reduction axis stays 0 either way.
  #
  # With the default batch_size of 1 the batch axis is dropped on the way out,
  # so view answers [t, C] and append takes [C] or [n, C]. Five models hold a
  # cache this way and none of them batches yet.
  #
  # This is the path AGENTS.md warns about: writing into a row of a preallocated
  # buffer and reading back a growing slice are exactly the non-contiguous view
  # operations that cumo #245 and later fixed.
  class KVCache
    attr_reader :num_layers, :max_seq_len, :channels, :batch_size

    def self.bytes_for(num_layers:, max_seq_len:, channels:, batch_size: 1)
      # K and V, fp32.
      2 * batch_size * num_layers * max_seq_len * channels * XF::ELEMENT_BYTE_SIZE
    end

    def initialize(num_layers:, max_seq_len:, channels:, batch_size: 1)
      raise Error, "batch size must be positive, got #{batch_size}" unless batch_size.positive?

      @num_layers = num_layers
      @max_seq_len = max_seq_len
      @channels = channels
      @batch_size = batch_size
      @keys = Array.new(num_layers) { XF.zeros(max_seq_len, batch_size, channels) }
      @values = Array.new(num_layers) { XF.zeros(max_seq_len, batch_size, channels) }
      @positions = Array.new(num_layers, 0)
    end

    def bytes
      self.class.bytes_for(num_layers: @num_layers, max_seq_len: @max_seq_len,
                           channels: @channels, batch_size: @batch_size)
    end

    def batched?
      @batch_size > 1
    end

    # Rows already written for this layer. Layers advance in lockstep, so with no
    # argument this is the length of the cached sequence. Every sequence in the
    # batch advances together, so this is one number and not one per sequence.
    def length(layer = 0)
      @positions[layer]
    end

    def reset
      @positions.fill(0)
      self
    end

    # Unbatched: keys/values are [C] for a single row or [n, C] for a prefill.
    # Batched: [B, C] for one row per sequence or [n, B, C] for a prefill.
    def append(layer, keys, values)
      check_layer(layer)
      k = as_block(keys)
      v = as_block(values)
      rows = k.shape[0]
      unless v.shape == k.shape
        raise Error, "key/value shape mismatch: #{k.shape.inspect} vs #{v.shape.inspect}"
      end
      unless k.shape[1] == @batch_size && k.shape[2] == @channels
        raise Error, "expected [n, #{@batch_size}, #{@channels}], got #{k.shape.inspect}"
      end

      position = @positions[layer]
      if position + rows > @max_seq_len
        raise Error, "kv cache overflow on layer #{layer}: #{position} + #{rows} rows " \
                     "exceeds max_seq_len #{@max_seq_len}"
      end

      span = position...(position + rows)
      @keys[layer][span, true, true] = k
      @values[layer][span, true, true] = v
      @positions[layer] = position + rows
      self
    end

    # The rows written so far, as [t, B, C] slices, or [t, C] when unbatched.
    # Views into the preallocated buffers, not copies, and contiguous.
    def view(layer)
      check_layer(layer)
      length = @positions[layer]
      # Both Numo and Cumo raise on an empty range slice ("0...0 is out of
      # range"), so an empty cache has to be spelled out.
      return [empty_rows, empty_rows] if length.zero?

      span = 0...length
      return [@keys[layer][span, 0, true], @values[layer][span, 0, true]] unless batched?

      [@keys[layer][span, true, true], @values[layer][span, true, true]]
    end

    private

    def empty_rows
      batched? ? XF.zeros(0, @batch_size, @channels) : XF.zeros(0, @channels)
    end

    # Answers [n, B, C]. Unbatched input carries no batch axis, so one is added.
    def as_block(tensor)
      return tensor if tensor.ndim == 3

      if batched?
        # [B, C] is one position for each sequence. The batch is checked here
        # because reshape would raise ArgumentError before append sees it.
        if tensor.ndim == 2 && tensor.shape[0] == @batch_size
          return tensor.reshape(1, @batch_size, tensor.shape[1])
        end

        raise Error, "batched append needs [#{@batch_size}, C] or [n, #{@batch_size}, C], " \
                     "got #{tensor.shape.inspect}"
      end

      rows = tensor.ndim == 1 ? tensor.reshape(1, tensor.shape[0]) : tensor
      rows.reshape(rows.shape[0], 1, rows.shape[1])
    end

    def check_layer(layer)
      return if layer.is_a?(Integer) && layer >= 0 && layer < @num_layers

      raise Error, "no such layer: #{layer.inspect} (num_layers=#{@num_layers})"
    end
  end
end
