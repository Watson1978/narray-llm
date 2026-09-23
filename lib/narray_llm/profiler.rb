# frozen_string_literal: true

module NArrayLLM
  # Per-section timing. Disabled by default: on Cumo a section boundary has to
  # synchronize, and synchronizing between every op makes the whole pass slower
  # than it really is. Report totals from a run with this off (AGENTS.md).
  class Profiler
    # Coarse granularity: one section per block plus the three things outside
    # them. A synchronize costs the same whatever it wraps, so the per-op
    # breakdown charges the frequently called sections far more than the rare
    # ones -- gemm is entered 48 times per token and unembed once. Timing whole
    # blocks cuts the number of synchronizes by about ten and with it the
    # distortion, at the cost of detail.
    COARSE_SECTIONS = %i[embed block final_norm unembed].freeze

    attr_reader :totals, :counts

    def initialize(enabled: false, only: nil)
      @enabled = enabled
      @only = only
      @totals = Hash.new(0.0)
      @counts = Hash.new(0)
    end

    def self.coarse
      new(enabled: true, only: COARSE_SECTIONS)
    end

    def enabled?
      @enabled
    end

    def section(name)
      return yield unless @enabled && (@only.nil? || @only.include?(name))

      synchronize
      started = clock
      result = yield
      synchronize
      @totals[name] += clock - started
      @counts[name] += 1
      result
    end

    def synchronize
      XM::CUDA::Runtime.cudaDeviceSynchronize if NArrayLLM.gpu?
    end

    def total
      @totals.values.sum
    end

    def reset
      @totals.clear
      @counts.clear
      self
    end

    # [[name, seconds, calls, share], ...] sorted by time, slowest first.
    def rows
      sum = total
      @totals.sort_by { |_, seconds| -seconds }
             .map { |name, seconds| [name, seconds, @counts[name], sum.zero? ? 0.0 : seconds / sum] }
    end

    def clock
      Process.clock_gettime(Process::CLOCK_MONOTONIC)
    end

    NULL = new(enabled: false)
  end
end
