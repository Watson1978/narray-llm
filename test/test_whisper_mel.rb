# frozen_string_literal: true

require_relative 'test_helper'

# Stage 3 acceptance tests (docs/plans/PLAN-whisper.md).
#
# The log-mel front end, checked against the one the published feature
# extractor produced for the same waveform.
class TestWhisperMel < Test::Unit::TestCase
  include TestHelper

  MODEL_DIR = File.expand_path('../data/whisper-tiny', __dir__)
  DUMP = File.expand_path('../data/whisper-tiny_encoder_state.safetensors', __dir__)
  FIXTURE = File.expand_path('../python/fixtures/whisper-tiny_greedy.json', __dir__)

  # The DFT is two fp32 matmuls over 400 terms where the reference runs
  # numpy's FFT in float64, and log10 magnifies whatever is left in the bins
  # that nearly cancel. The worst seen is 1.6e-03, all of it where the
  # reference is within 0.05 of its own floor; above 0.5 it is 1.5e-05.
  TOLERANCE = 2.5e-3
  LOUD_TOLERANCE = 5.0e-5

  def setup
    omit("#{DUMP} not found; run `rake prepare:whisper`") unless File.exist?(DUMP)

    @dump = NArrayLLM::Safetensors.new(DUMP)
    @mel = NArrayLLM::Whisper::Mel.load(File.join(MODEL_DIR, 'preprocessor_config.json'))
  end

  def teardown
    @dump&.close
  end

  def waveform
    @waveform ||= NArrayLLM::Ops.contiguous(@dump['waveform'], NArrayLLM::Whisper::Mel::PRECISION)
  end

  def test_the_mel_matches_the_feature_extractor
    want = @dump['input_features']
    got = @mel.call(waveform)
    assert_equal([@mel.mel_bins, @mel.frames], got.shape)
    worst = host((want - got).abs.max)
    assert_operator(worst, :<, TOLERANCE)
    notify(format('最大 max|d| = %.3e (閾値 %.1e)', worst, TOLERANCE))
  end

  # Where the signal has energy the two agree to 1.5e-05. The 1.6e-03 lives in
  # the bins that nearly cancel, which is what an fp32 DFT cannot hold and
  # log10 then magnifies. Repeating the pipeline in float64 brings the worst
  # case to 4.8e-05, which is how that was established rather than assumed.
  def test_the_error_is_in_the_quiet_bins
    want = @dump['input_features']
    difference = (want - @mel.call(waveform)).abs
    loud = XF.cast(want.ge(0.5))
    assert_operator(host((difference * loud).max), :<, LOUD_TOLERANCE)
    notify(format('参照が 0.5 以上の %d 個では max|d| = %.3e',
                  host(loud.sum).to_i, host((difference * loud).max)))
  end

  # The point of the front end is that the model takes what it produces.
  def test_the_tokens_survive_our_own_mel
    omit("#{FIXTURE} not found; run python/whisper_fixtures.py") unless File.exist?(FIXTURE)

    fixture = JSON.parse(File.read(FIXTURE))
    model = NArrayLLM::Whisper::Model.load(File.join(MODEL_DIR, 'model.safetensors'))
    tokens = model.generate(@mel.call(waveform), prompt: fixture['prompt'],
                                                 max_new_tokens: fixture['max_new_tokens'],
                                                 suppress: fixture['suppress_tokens'],
                                                 begin_suppress: fixture['begin_suppress_tokens'])
    assert_equal(fixture['tokens'], tokens)
  end

  # window_function(400, "hann") with periodic true, which is 0.5 - 0.5
  # cos(2 pi n / N) and not the symmetric one that divides by N - 1.
  def test_the_window_is_a_periodic_hann
    window = @mel.send(:instance_variable_get, :@window).reshape(400)
    assert_in_delta(0.0, host(window[0]), 1.0e-7, '先頭は 0')
    assert_in_delta(1.0, host(window[200]), 1.0e-7, '中央は 1')
    assert_in_delta(0.5, host(window[100]), 1.0e-6)
    # Symmetric would put the 1.0 between samples and never reach it exactly.
    assert_not_in_delta(0.0, host(window[399]), 1.0e-7, '末尾は 0 ではない')
  end

  # np.pad(..., mode="reflect") does not repeat the edge sample: the left side
  # runs x[pad], x[pad - 1], ... x[1].
  def test_the_padding_reflects_without_repeating_the_edge
    mel = NArrayLLM::Whisper::Mel.new(small_preprocessor)
    signal = XF[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    padded = host(mel.send(:reflect, signal)).to_a
    # np.pad(np.arange(8), 3, mode="reflect")
    assert_equal([3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 4.0], padded)
  end

  def test_a_short_waveform_is_padded_and_a_long_one_trimmed
    mel = NArrayLLM::Whisper::Mel.new(small_preprocessor)
    short = host(mel.send(:fit, XF[1.0, 2.0])).to_a
    assert_equal([1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], short)
    long = host(mel.send(:fit, XF.new(20).seq)).to_a
    assert_equal((0...8).map(&:to_f), long)
  end

  def test_a_filterbank_of_the_wrong_shape_is_rejected
    broken = small_preprocessor.merge('mel_filters' => [[1.0, 2.0]])
    assert_raise_kind_of(NArrayLLM::FormatError) { NArrayLLM::Whisper::Mel.new(broken) }
  end

  private

  # n_fft 6 gives 4 bins and a padding of 3, small enough to write out.
  def small_preprocessor
    { 'n_fft' => 6, 'hop_length' => 2, 'n_samples' => 8, 'nb_max_frames' => 4,
      'feature_size' => 2, 'mel_filters' => Array.new(2) { Array.new(4, 0.5) } }
  end
end
