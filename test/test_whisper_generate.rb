# frozen_string_literal: true

require_relative 'test_helper'

# Stage 2 acceptance tests (docs/plans/PLAN-whisper.md).
#
# The token sequence has to match transformers, but the waveform is synthetic
# and the model answers one id thirty-two times over, which on its own would
# pass for an implementation that was wrong in almost any way. The logits are
# therefore checked position by position against a teacher-forced reference
# run, and that is the check with teeth.
class TestWhisperGenerate < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/whisper-tiny/model.safetensors', __dir__)
  DUMP = File.expand_path('../data/whisper-tiny_encoder_state.safetensors', __dir__)
  FIXTURE = File.expand_path('../python/fixtures/whisper-tiny_greedy.json', __dir__)

  # Logits over 51865 ids with a range of 28.4, after four encoder layers, two
  # convolutions and four decoder layers. The worst seen is 7.4e-05 on Cumo.
  TOLERANCE = 2.0e-4

  def setup
    omit("#{MODEL} not found; run `rake prepare:whisper`") unless File.exist?(MODEL)
    omit("#{DUMP} not found; run `rake prepare:whisper`") unless File.exist?(DUMP)
    omit("#{FIXTURE} not found; run python/whisper_fixtures.py") unless File.exist?(FIXTURE)

    @dump = NArrayLLM::Safetensors.new(DUMP)
    @fixture = JSON.parse(File.read(FIXTURE))
  end

  def teardown
    @dump&.close
  end

  def model(spelling: :shift)
    @models ||= {}
    @models[spelling] ||= NArrayLLM::Whisper::Model.load(MODEL, spelling: spelling)
  end

  def mel
    @mel ||= NArrayLLM::Ops.contiguous(@dump['input_features'])
  end

  # Teacher forced over a fixed sequence, one reference logit row per
  # position. The encoder's own output is used rather than the reference's, so
  # this covers the whole model.
  def test_the_decoder_logits_match_at_every_position
    ids = @dump['decoder_input_ids'].to_a.map(&:to_i)
    want = @dump['decoder_logits']
    subject = model
    cache = subject.new_cache(subject.encode(mel))
    worst = 0.0
    ids.each_with_index do |id, i|
      got = subject.decode(id, cache: cache).reshape(subject.config.vocab_size)
      row = want[i, true]
      worst = [worst, host((row - got).abs.max)].max
      assert_equal(host(row.max_index).to_i, host(got.max_index).to_i, "位置 #{i} の argmax")
    end
    assert_operator(worst, :<, TOLERANCE)
    notify(format('%d 位置、最大 max|d| = %.3e (閾値 %.1e)', ids.size, worst, TOLERANCE))
  end

  data('shift', :shift)
  data('unfold', :unfold)
  def test_the_token_sequence_matches_transformers(spelling)
    tokens = model(spelling: spelling).generate(
      mel, prompt: @fixture['prompt'], max_new_tokens: @fixture['max_new_tokens'],
           suppress: @fixture['suppress_tokens'], begin_suppress: @fixture['begin_suppress_tokens']
    )
    assert_equal(@fixture['tokens'], tokens, spelling.to_s)
  end

  # The prompt is four forced ids, not one. Whisper decodes from
  # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>, and the
  # answer holds only what came after them.
  def test_the_prompt_is_consumed_and_not_returned
    prompt = @fixture['prompt']
    assert_equal(4, prompt.size)
    assert_equal(model.config.decoder_start_token_id, prompt.first)
    tokens = model.generate(mel, prompt: prompt, max_new_tokens: 3,
                            suppress: @fixture['suppress_tokens'],
                            begin_suppress: @fixture['begin_suppress_tokens'])
    assert_equal(3, tokens.size)
    assert_equal(@fixture['tokens'][0, 3], tokens)
  end

  # Suppression is not decoration: without it the first generated id differs,
  # which is what makes it part of the gate rather than a detail.
  def test_suppression_changes_what_comes_out
    prompt = @fixture['prompt']
    with = model.generate(mel, prompt: prompt, max_new_tokens: 1,
                          suppress: @fixture['suppress_tokens'],
                          begin_suppress: @fixture['begin_suppress_tokens'])
    without = model.generate(mel, prompt: prompt, max_new_tokens: 1, suppress: [], begin_suppress: [])
    assert_equal(@fixture['tokens'][0, 1], with)
    assert_not_equal(with, without, '抑制が効いていること')
    notify(format('抑制あり %s、抑制なし %s', with.inspect, without.inspect))
  end

  def test_an_out_of_range_token_is_rejected
    subject = model
    cache = subject.new_cache(subject.encode(mel))
    assert_raise_kind_of(NArrayLLM::Error) { subject.decode(-1, cache: cache) }
    assert_raise_kind_of(NArrayLLM::Error) { subject.decode(subject.config.vocab_size, cache: cache) }
  end
end
