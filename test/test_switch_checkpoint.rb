# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-switch.md).
#
# The expected numbers come from google/switch-base-8's config.json and from
# transformers' modeling_switch_transformers.py, read on 2026-09-19. Nothing
# here is guessed: the shapes the loader checks are computed from the config,
# so a file that disagrees with its own config is rejected.
class TestSwitchCheckpoint < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/switch-base-8.safetensors', __dir__)

  # config.json of google/switch-base-8.
  NOMINAL = {
    num_layers: 12, num_decoder_layers: 12, d_model: 768, d_ff: 3072, d_kv: 64,
    num_heads: 12, num_experts: 8, expert_capacity: 64, vocab_size: 32_128
  }.freeze

  # 437 tensors: the shared embedding, two final norms, and the blocks. The
  # published pickle has 440 because it stores the embedding under four names
  # that share one storage; the export writes it once.
  TENSORS = 437
  PARAMETERS = 619_339_008

  def checkpoint
    omit("#{MODEL} not found; run `rake prepare:switch`") unless File.exist?(MODEL)

    @checkpoint ||= NArrayLLM::Switch::Checkpoint.load(MODEL)
  end

  def teardown
    @checkpoint&.close
    @checkpoint = nil
  end

  def test_config_matches_the_published_one
    config = checkpoint.config
    NOMINAL.each { |field, want| assert_equal(want, config[field], field.to_s) }
    assert_equal(1e-6, config.layer_norm_epsilon)
    assert_equal('relu', config.dense_act_fn)
    assert_false(config.is_gated_act, 'switch-base-8 は gated ではない')
    assert_false(config.router_bias)
    assert_equal(0, config.router_jitter_noise)
    assert_equal(0, config.decoder_start_token_id)
    assert_equal(1, config.eos_token_id)
  end

  # encoder_sparse_step and decoder_sparse_step are both 2, and the published
  # weights carry a router on blocks 1, 3, 5, 7, 9 and 11 of each stack.
  def test_sparse_layers_are_the_odd_blocks
    config = checkpoint.config
    sparse = (0...config.num_layers).select { |b| config.sparse_encoder_layer?(b) }
    assert_equal([1, 3, 5, 7, 9, 11], sparse)
    assert_equal(sparse, (0...config.num_decoder_layers).select { |b| config.sparse_decoder_layer?(b) })
    sparse.each do |block|
      assert_true(checkpoint.include?("encoder.block.#{block}.layer.1.mlp.router.classifier.weight"))
      assert_true(checkpoint.include?("decoder.block.#{block}.layer.2.mlp.router.classifier.weight"))
    end
    (0...config.num_layers).reject { |b| config.sparse_encoder_layer?(b) }.each do |block|
      assert_true(checkpoint.include?("encoder.block.#{block}.layer.1.mlp.wi.weight"))
      assert_false(checkpoint.include?("encoder.block.#{block}.layer.1.mlp.router.classifier.weight"))
    end
  end

  def test_every_expert_is_stored_separately
    config = checkpoint.config
    config.num_experts.times do |expert|
      wi = checkpoint["encoder.block.1.layer.1.mlp.experts.expert_#{expert}.wi.weight"]
      wo = checkpoint["encoder.block.1.layer.1.mlp.experts.expert_#{expert}.wo.weight"]
      assert_equal([config.d_ff, config.d_model], wi.shape)
      assert_equal([config.d_model, config.d_ff], wo.shape)
    end
    first = checkpoint['encoder.block.1.layer.1.mlp.experts.expert_0.wi.weight']
    second = checkpoint['encoder.block.1.layer.1.mlp.experts.expert_1.wi.weight']
    assert_operator(host((first - second).abs.max), :>, 0.0, '8 個が同じ重みではない')
  end

  def test_counts_agree_with_the_config
    assert_equal(TENSORS, checkpoint.names.size)
    assert_equal(PARAMETERS, checkpoint.num_parameters)
    assert_equal(TENSORS, checkpoint.expected_shapes.size)
  end

  # The pickle ties shared, both embed_tokens and lm_head to one storage. The
  # export writes only shared.weight, so asking for the other three has to
  # answer the same values rather than fail.
  def test_the_four_tied_names_answer_the_same_weights
    shared = checkpoint['shared.weight']
    NArrayLLM::Switch::Checkpoint::TIED.each do |name|
      assert_true(checkpoint.include?(name), name)
      assert_equal(0.0, host((checkpoint[name] - shared).abs.max), name)
    end
  end

  def test_the_relative_bias_lives_on_the_first_block_only
    config = checkpoint.config
    bias = checkpoint['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight']
    assert_equal([config.relative_attention_num_buckets, config.num_heads], bias.shape)
    assert_false(checkpoint.include?('encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight'))
  end

  def test_a_file_that_disagrees_with_its_config_is_rejected
    omit("#{MODEL} not found") unless File.exist?(MODEL)

    Dir.mktmpdir do |dir|
      config_path = File.join(dir, 'config.json')
      raw = JSON.parse(File.read(NArrayLLM::Switch::Checkpoint.config_beside(MODEL)))
      File.write(config_path, JSON.dump(raw.merge('num_experts' => 4)))
      error = assert_raise_kind_of(NArrayLLM::FormatError) do
        NArrayLLM::Switch::Checkpoint.load(MODEL, config_path: config_path)
      end
      assert_match(/unexpected tensors/, error.message)
    end
  end
end
