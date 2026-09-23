# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-llama2.md).
#
# Every fixture below is transcribed from docs/checkpoint-format-llama2.md,
# which cites llama2.c by function and line. Nothing here is guessed.
class TestLlama2Checkpoint < Test::Unit::TestCase
  include TestHelper

  # The tinyllamas table in the llama2.c README. stories260K is multiquery
  # (n_kv_heads 4 < n_heads 8); the larger models are not.
  NOMINAL = {
    'stories260K.bin' => {
      dim: 64, hidden_dim: 172, num_layers: 5, num_heads: 8, num_kv_heads: 4,
      vocab_size: 512, max_seq_len: 512
    },
    'stories110M.bin' => {
      dim: 768, hidden_dim: 2048, num_layers: 12, num_heads: 12, num_kv_heads: 12,
      vocab_size: 32_000, max_seq_len: 1024
    }
  }.freeze

  # memory_map_weights (run.c:110-140): the order the pointers walk, and the
  # extent each one covers.
  SHAPES_260K = [
    [:token_embedding_table, [512, 64]],
    [:rms_att_weight,        [5, 64]],
    [:wq,                    [5, 64, 64]],
    [:wk,                    [5, 32, 64]],
    [:wv,                    [5, 32, 64]],
    [:wo,                    [5, 64, 64]],
    [:rms_ffn_weight,        [5, 64]],
    [:w1,                    [5, 172, 64]],
    [:w2,                    [5, 64, 172]],
    [:w3,                    [5, 172, 64]],
    [:rms_final_weight,      [64]]
  ].freeze

  # The same pointer arithmetic carried out by hand, in bytes from the start of
  # the file. This is what run.c would read; the loader must land on the same
  # places. wcls is not listed because stories260K shares it (see below).
  OFFSETS_260K = {
    token_embedding_table: 28,
    rms_att_weight: 131_100,
    wq: 132_380,
    wk: 214_300,
    wv: 255_260,
    wo: 296_220,
    rms_ffn_weight: 378_140,
    w1: 379_420,
    w2: 599_580,
    w3: 819_740,
    rms_final_weight: 1_039_900
  }.freeze

  STORIES260K_BYTES = 1_056_540

  def load_checkpoint(name)
    NArrayLLM::Llama2::Checkpoint.load(require_data(name))
  end

  data(NOMINAL.keys.to_h { |n| [n, n] })
  def test_the_header_matches_the_published_config(name)
    config = load_checkpoint(name).config
    NOMINAL.fetch(name).each do |field, expected|
      assert_equal(expected, config[field], "#{name}: #{field}")
    end
  end

  def test_the_derived_head_geometry_follows_run_c(_data = nil)
    config = load_checkpoint('stories260K.bin').config
    # head_size = dim / n_heads and kv_dim = dim * n_kv_heads / n_heads (run.c:111, 236).
    assert_equal(8, config.head_size)
    assert_equal(32, config.kv_dim)
    assert_equal(2, config.kv_mul)
    assert_true(config.grouped_query?, 'stories260K is multiquery, not MHA')
  end

  def test_stories110m_is_plain_multi_head
    config = load_checkpoint('stories110M.bin').config
    assert_equal(64, config.head_size)
    assert_equal(768, config.kv_dim)
    assert_equal(1, config.kv_mul)
    assert_false(config.grouped_query?)
  end

  def test_the_tensor_shapes_match_the_pointer_walk
    checkpoint = load_checkpoint('stories260K.bin')
    SHAPES_260K.each do |name, shape|
      assert_equal(shape, checkpoint[name].shape, "shape of #{name}")
    end
  end

  data(NOMINAL.keys.to_h { |n| [n, n] })
  def test_the_parameter_count_accounts_for_the_whole_file(name)
    checkpoint = load_checkpoint(name)
    assert_equal(File.size(data_path(name)), checkpoint.expected_file_size,
                 "#{name}: header + parameters + skipped freq_cis must be the file")
  end

  # Reading the file a second way: unpack('e*') straight out of the byte
  # offsets above, which is a different code path from the loader's from_binary.
  def test_each_tensor_starts_where_run_c_would_look
    path = require_data('stories260K.bin')
    checkpoint = load_checkpoint('stories260K.bin')
    OFFSETS_260K.each do |name, offset|
      assert_equal(offset, checkpoint.byte_offset(name), "byte offset of #{name}")
      assert_bit_identical(raw_fp32(path, offset, 5), checkpoint[name].flatten[0...5],
                           "first five floats of #{name}")
    end
  end

  # run.c:139: wcls points at the embedding when the flag says shared, and the
  # file does not carry a second copy.
  def test_a_shared_classifier_aliases_the_embedding
    checkpoint = load_checkpoint('stories260K.bin')
    assert_true(checkpoint.config.shared_classifier)
    assert_equal(checkpoint.byte_offset(:token_embedding_table),
                 checkpoint.byte_offset(:wcls))
    assert_bit_identical(checkpoint[:token_embedding_table].flatten[0...5].to_a,
                         checkpoint[:wcls].flatten[0...5], 'wcls aliases the embedding')
  end

  # Neither published checkpoint stores an unshared classifier, so the negative
  # vocab_size path needs a file built from the spec to be exercised at all.
  def test_a_negative_vocab_size_means_a_separate_classifier
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'unshared.bin')
      config = { dim: 4, hidden_dim: 8, num_layers: 1, num_heads: 2, num_kv_heads: 1,
                 vocab_size: 3, max_seq_len: 6 }
      write_synthetic(path, config, shared: false)

      checkpoint = NArrayLLM::Llama2::Checkpoint.load(path)
      assert_false(checkpoint.config.shared_classifier)
      assert_equal(3, checkpoint.config.vocab_size, 'vocab_size is stored negated, not negative')
      assert_not_equal(checkpoint.byte_offset(:token_embedding_table),
                       checkpoint.byte_offset(:wcls))
      # The synthetic classifier is filled with a value the embedding never takes.
      assert_bit_identical([-1.0] * 5, checkpoint[:wcls].flatten[0...5], 'wcls is its own tensor')
      assert_equal(File.size(path), checkpoint.expected_file_size)
    end
  end

  def test_a_truncated_file_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'short.bin')
      config = { dim: 4, hidden_dim: 8, num_layers: 1, num_heads: 2, num_kv_heads: 1,
                 vocab_size: 3, max_seq_len: 6 }
      write_synthetic(path, config, shared: true)
      File.truncate(path, File.size(path) - 4)

      assert_raise(NArrayLLM::FormatError) { NArrayLLM::Llama2::Checkpoint.load(path) }
    end
  end

  def test_trailing_bytes_are_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'long.bin')
      config = { dim: 4, hidden_dim: 8, num_layers: 1, num_heads: 2, num_kv_heads: 1,
                 vocab_size: 3, max_seq_len: 6 }
      write_synthetic(path, config, shared: true)
      File.open(path, 'ab') { |io| io.write([0.0].pack('e')) }

      assert_raise(NArrayLLM::FormatError) { NArrayLLM::Llama2::Checkpoint.load(path) }
    end
  end

  def test_a_head_count_that_does_not_divide_dim_is_rejected
    Dir.mktmpdir do |dir|
      path = File.join(dir, 'bad.bin')
      File.binwrite(path, [5, 8, 1, 2, 1, 3, 6].pack('l<7'))
      assert_raise(NArrayLLM::FormatError) { NArrayLLM::Llama2::Checkpoint.load(path) }
    end
  end

  private

  # A file laid out exactly as memory_map_weights walks it, so the loader can be
  # tested on shapes the published checkpoints do not cover.
  def write_synthetic(path, config, shared:)
    dim = config[:dim]
    layers = config[:num_layers]
    hidden = config[:hidden_dim]
    vocab = config[:vocab_size]
    head_size = dim / config[:num_heads]
    kv_dim = dim * config[:num_kv_heads] / config[:num_heads]

    counts = [vocab * dim, layers * dim, layers * dim * dim, layers * kv_dim * dim,
              layers * kv_dim * dim, layers * dim * dim, layers * dim,
              layers * hidden * dim, layers * dim * hidden, layers * hidden * dim, dim]
    body = counts.sum
    freq_cis = 2 * (config[:max_seq_len] * head_size / 2)

    File.open(path, 'wb') do |io|
      io.write([dim, hidden, layers, config[:num_heads], config[:num_kv_heads],
                shared ? vocab : -vocab, config[:max_seq_len]].pack('l<7'))
      io.write(Array.new(body) { |i| i * 0.5 }.pack('e*'))
      io.write(Array.new(freq_cis, 0.25).pack('e*'))
      io.write(Array.new(vocab * dim, -1.0).pack('e*')) unless shared
    end
  end
end
