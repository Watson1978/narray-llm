# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-whisper.md).
#
# The expected numbers come from openai/whisper-tiny's config.json and from
# transformers' modeling_whisper.py, read on 2026-09-20. The shapes the loader
# checks are computed from the config, so a file that disagrees with its own
# config is rejected.
class TestWhisperCheckpoint < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/whisper-tiny/model.safetensors', __dir__)

  NOMINAL = {
    d_model: 384, encoder_layers: 4, decoder_layers: 4,
    encoder_attention_heads: 6, decoder_attention_heads: 6,
    encoder_ffn_dim: 1536, decoder_ffn_dim: 1536, num_mel_bins: 80,
    max_source_positions: 1500, max_target_positions: 448, vocab_size: 51_865
  }.freeze

  TENSORS = 167
  PARAMETERS = 37_760_640

  def setup
    omit("#{MODEL} not found; run `rake prepare:whisper`") unless File.exist?(MODEL)

    @checkpoint = NArrayLLM::Whisper::Checkpoint.load(MODEL)
  end

  def teardown
    @checkpoint&.close
  end

  def test_config_matches_the_published_one
    config = @checkpoint.config
    NOMINAL.each { |field, want| assert_equal(want, config[field], field.to_s) }
    assert_equal('gelu', config.activation_function)
    assert_false(config.scale_embedding)
    assert_equal(50_258, config.decoder_start_token_id)
    assert_equal(50_257, config.eos_token_id)
    assert_equal(64, config.head_dim)
    assert_equal(3000, config.mel_frames, '2 つ目の畳み込みが stride 2')
  end

  def test_counts_agree_with_the_config
    assert_equal(TENSORS, @checkpoint.names.size)
    assert_equal(PARAMETERS, @checkpoint.num_parameters)
    assert_equal(TENSORS, @checkpoint.expected_shapes.size)
  end

  # Whisper leaves k_proj without a bias and gives q, v and out one. The
  # expected shapes say so, so a file that carried one would be rejected.
  def test_only_the_key_projection_has_no_bias
    %w[encoder decoder].each do |stack|
      kinds = stack == 'encoder' ? %w[self_attn] : %w[self_attn encoder_attn]
      kinds.each do |kind|
        at = "model.#{stack}.layers.0.#{kind}"
        assert_false(@checkpoint.include?("#{at}.k_proj.bias"), "#{at}.k_proj.bias")
        %w[q_proj v_proj out_proj].each do |part|
          assert_true(@checkpoint.include?("#{at}.#{part}.bias"), "#{at}.#{part}.bias")
        end
      end
    end
  end

  # The checkpoint has no proj_out: the classifier is the embedding read the
  # other way, the way Switch shares its own.
  def test_the_output_projection_is_the_embedding
    embedding = @checkpoint[NArrayLLM::Whisper::Checkpoint::EMBED]
    assert_equal([51_865, 384], embedding.shape)
    assert_true(@checkpoint.include?('proj_out.weight'))
    assert_equal(0.0, host((@checkpoint['proj_out.weight'] - embedding).abs.max))
  end

  # The sinusoidal positions the paper builds are stored in the file, so
  # nothing has to generate them. The decoder's are learned.
  def test_both_position_tables_are_stored
    config = @checkpoint.config
    assert_equal([config.max_source_positions, config.d_model],
                 @checkpoint['model.encoder.embed_positions.weight'].shape)
    assert_equal([config.max_target_positions, config.d_model],
                 @checkpoint['model.decoder.embed_positions.weight'].shape)
  end

  def test_the_convolutions_are_shaped_for_the_mel_input
    config = @checkpoint.config
    assert_equal([config.d_model, config.num_mel_bins, 3],
                 @checkpoint['model.encoder.conv1.weight'].shape)
    assert_equal([config.d_model, config.d_model, 3],
                 @checkpoint['model.encoder.conv2.weight'].shape)
  end

  def test_a_file_that_disagrees_with_its_config_is_rejected
    Dir.mktmpdir do |dir|
      config_path = File.join(dir, 'config.json')
      raw = JSON.parse(File.read(NArrayLLM::Whisper::Checkpoint.config_beside(MODEL)))
      File.write(config_path, JSON.dump(raw.merge('encoder_layers' => 2)))
      error = assert_raise_kind_of(NArrayLLM::FormatError) do
        NArrayLLM::Whisper::Checkpoint.load(MODEL, config_path: config_path)
      end
      assert_match(/unexpected tensors/, error.message)
    end
  end
end
