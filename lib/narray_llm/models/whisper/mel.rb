# frozen_string_literal: true

module NArrayLLM
  module Whisper
    # The log-mel spectrogram Whisper takes as input, built out of the array
    # operations this repository already has.
    #
    # There is no FFT here. Numo and Cumo both carry a complex type but no
    # transform, so the DFT is written as two real matmuls against cosine and
    # sine tables. That is 400 x 201 multiplications a frame where an FFT
    # would be about 3500 in total, but it is two [3000, 400] x [400, 201]
    # products, which is 0.48 GFLOP for the whole 30 seconds.
    #
    # The mel filterbank is not recomputed: preprocessor_config.json publishes
    # it as an [80, 201] table and that is read as data.
    class Mel
      # feature_extraction_whisper.py and audio_utils.spectrogram's defaults:
      # a periodic Hann window, centred frames with reflected padding, one
      # sided, power 2, then log10 and the two clamps.
      FLOOR = 1.0e-10
      DYNAMIC_RANGE = 8.0
      # The whole front end runs in fp32 whatever XF is, and the result is cast
      # on the way out. The phase table needs the index product n_fft * bins =
      # 79800, which is Infinity in fp16, and bf16 rounds it far enough to flip
      # the sign of the cosine (max|d| 2.0). FLOOR underflows to 0 in fp16.
      PRECISION = XM::SFloat

      attr_reader :n_fft, :hop_length, :n_samples, :frames, :bins, :mel_bins

      def self.load(path)
        new(JSON.parse(File.read(path)))
      end

      def initialize(preprocessor)
        @n_fft = preprocessor.fetch('n_fft')
        @hop_length = preprocessor.fetch('hop_length')
        @n_samples = preprocessor.fetch('n_samples')
        @frames = preprocessor.fetch('nb_max_frames')
        @mel_bins = preprocessor.fetch('feature_size')
        filters = preprocessor.fetch('mel_filters')
        @bins = (@n_fft / 2) + 1
        unless filters.size == @mel_bins && filters.first.size == @bins
          raise FormatError, "mel_filters is #{filters.size}x#{filters.first.size}, " \
                             "expected #{@mel_bins}x#{@bins}"
        end

        @filters = Ops.contiguous(PRECISION.cast(filters), PRECISION)
        prepare
      end

      # waveform: [samples], trimmed or padded to n_samples by the caller or
      # here. Answers [mel_bins, frames].
      def call(waveform, prof: Profiler::NULL)
        signal = fit(waveform)
        windows = prof.section(:frames) { frame(reflect(signal)) }
        power = prof.section(:dft) do
          real = windows.dot(@cos)
          imag = windows.dot(@sin)
          (real * real) + (imag * imag)
        end
        spec = prof.section(:mel) do
          logarithm(@filters.dot(Ops.contiguous(power.transpose, PRECISION)))
        end
        XF == PRECISION ? spec : Ops.contiguous(spec)
      end

      private

      def fit(waveform)
        samples = waveform.shape[0]
        return Ops.contiguous(waveform[0...@n_samples], PRECISION) if samples > @n_samples
        return Ops.contiguous(waveform, PRECISION) if samples == @n_samples

        out = PRECISION.zeros(@n_samples)
        out[0...samples] = waveform
        out
      end

      # np.pad(..., mode="reflect"): the edge sample itself is not repeated, so
      # the left side runs x[pad], x[pad - 1], ... x[1].
      def reflect(signal)
        pad = @n_fft / 2
        out = PRECISION.new(@n_samples + (2 * pad)).allocate
        out[pad...(pad + @n_samples)] = signal
        out[0...pad] = signal[(1..pad).to_a.reverse, false]
        tail = @n_samples - 2
        out[(pad + @n_samples)...(@n_samples + (2 * pad))] = signal[(tail - pad + 1)..tail, false].reverse
        out
      end

      # Frame t covers padded[t * hop, n_fft). With hop 160 and a frame of
      # 400 that is two and a half hop blocks, so the window matrix is three
      # slices of a [blocks, hop] reshape rather than n_fft strided copies.
      def frame(padded)
        blocks = @n_fft / @hop_length
        rest = @n_fft % @hop_length
        needed = @frames + blocks + (rest.zero? ? 0 : 1) - 1
        grid = Ops.contiguous(padded[0...(needed * @hop_length)], PRECISION)
               .reshape!(needed, @hop_length)
        out = PRECISION.new(@frames, @n_fft).allocate
        blocks.times do |b|
          out[true, (b * @hop_length)...((b + 1) * @hop_length)] = grid[b...(b + @frames), true]
        end
        unless rest.zero?
          out[true, (blocks * @hop_length)...@n_fft] = grid[blocks...(blocks + @frames), 0...rest]
        end
        out * @window
      end

      def logarithm(mel)
        spec = PRECISION::Math.log10(mel.clip(FLOOR, nil))
        floor = NArrayLLM.scalar(spec.max) - DYNAMIC_RANGE
        (spec.clip(floor, nil) + 4.0) / 4.0
      end

      def prepare
        # Periodic Hann: 0.5 - 0.5 cos(2 pi n / N), which is what
        # window_function(n_fft, "hann") builds with periodic true.
        n = PRECISION.new(1, @n_fft).seq
        @window = 0.5 - (0.5 * PRECISION::Math.cos(n * (2.0 * Math::PI / @n_fft)))
        # The one sided DFT as two real tables.
        rows = PRECISION.new(@n_fft, 1).seq
        cols = PRECISION.new(1, @bins).seq
        angle = rows * cols * (-2.0 * Math::PI / @n_fft)
        @cos = Ops.contiguous(PRECISION::Math.cos(angle), PRECISION)
        @sin = Ops.contiguous(PRECISION::Math.sin(angle), PRECISION)
      end
    end
  end
end
