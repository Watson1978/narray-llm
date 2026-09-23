# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-mamba.md).
#
# The fixtures are transcribed from kroggen/mamba.c on its `learning` branch:
# the Config struct at mamba.c:19, memory_map_weights at mamba.c:152, and the
# header export.py writes. Nothing here is guessed.
class TestMambaCheckpoint < Test::Unit::TestCase
  include TestHelper

  # config.json of state-spaces/mamba-130m, plus the shapes export.py derives
  # from the weights themselves.
  NOMINAL_130M = {
    num_layers: 24, vocab_size: 50_277, dim: 768, d_inner: 1536,
    dt_rank: 48, d_state: 16, d_conv: 4
  }.freeze

  TINY = {
    num_layers: 2, vocab_size: 64, dim: 32, d_inner: 64,
    dt_rank: 2, d_state: 4, d_conv: 4
  }.freeze

  # memory_map_weights (mamba.c:152): the order the pointers walk and the
  # extent each one covers.
  SHAPES_TINY = [
    [:embedding,      [64, 32]],
    [:in_proj,        [2, 128, 32]],
    [:conv1d_weight,  [2, 64, 4]],
    [:conv1d_bias,    [2, 64]],
    [:x_proj,         [2, 10, 64]],
    [:dt_proj_weight, [2, 64, 2]],
    [:dt_proj_bias,   [2, 64]],
    [:a,              [2, 64, 4]],
    [:d,              [2, 64]],
    [:out_proj,       [2, 32, 64]],
    [:norm,           [2, 32]],
    [:final_norm,     [32]]
  ].freeze

  def tiny
    require_data('mamba_tiny.bin')
    NArrayLLM::Mamba::Checkpoint.load(data_path('mamba_tiny.bin'))
  end

  def test_tiny_header_matches_the_generator
    config = tiny.config
    TINY.each { |field, value| assert_equal(value, config[field], field.to_s) }
    assert_true(config.shared_classifier)
  end

  def test_tiny_shapes_follow_memory_map_weights
    checkpoint = tiny
    SHAPES_TINY.each do |name, shape|
      assert_equal(shape, checkpoint[name].shape, name.to_s)
    end
  end

  # The classifier is not stored when it is shared, so it has to be the very
  # same object as the embedding rather than a second copy.
  def test_tiny_shared_classifier_aliases_the_embedding
    checkpoint = tiny
    assert_same(checkpoint[:embedding], checkpoint[:lm_head])
    assert_equal(checkpoint.byte_offset(:embedding), checkpoint.byte_offset(:lm_head))
  end

  # If a tensor were the wrong size the walk would end somewhere other than the
  # end of the file, and the reader would either run short or leave bytes over.
  def test_tiny_file_size_accounts_for_every_byte
    checkpoint = tiny
    assert_equal(File.size(data_path('mamba_tiny.bin')), checkpoint.expected_file_size)
  end

  def test_tiny_offsets_are_contiguous_and_start_after_the_header
    checkpoint = tiny
    position = NArrayLLM::Mamba::Checkpoint::HEADER_BYTES
    SHAPES_TINY.each do |name, shape|
      assert_equal(position, checkpoint.byte_offset(name), "offset of #{name}")
      position += 4 * shape.inject(:*)
    end
    assert_equal(File.size(data_path('mamba_tiny.bin')), position)
  end

  def test_rounded_vocab_size_rounds_up_to_a_multiple_of_eight
    config = NArrayLLM::Mamba::Config.new(vocab_size: 50_277)
    assert_equal(50_280, config.rounded_vocab_size)
    assert_equal(64, NArrayLLM::Mamba::Config.new(vocab_size: 64).rounded_vocab_size)
  end

  def test_bad_magic_is_rejected
    path = File.join(Dir.tmpdir, "mamba_bad_magic_#{Process.pid}.bin")
    File.binwrite(path, "\0" * 256)
    assert_raise_kind_of(NArrayLLM::FormatError) { NArrayLLM::Mamba::Checkpoint.load(path) }
  ensure
    FileUtils.rm_f(path)
  end

  def test_130m_header_matches_the_published_config
    require_data('mamba-130m.bin')
    checkpoint = NArrayLLM::Mamba::Checkpoint.load(data_path('mamba-130m.bin'))
    NOMINAL_130M.each { |field, value| assert_equal(value, checkpoint.config[field], field.to_s) }
    assert_true(checkpoint.config.shared_classifier)
    # 50277 is not a multiple of 8, so the stored tables are wider than the
    # vocabulary (mamba.c:187).
    assert_equal(50_280, checkpoint.config.rounded_vocab_size)
    assert_equal([50_280, 768], checkpoint[:embedding].shape)
    assert_equal(File.size(data_path('mamba-130m.bin')), checkpoint.expected_file_size)
  end
end
