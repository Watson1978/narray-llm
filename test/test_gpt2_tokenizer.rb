# frozen_string_literal: true

require_relative 'test_helper'

class TestTokenizer < Test::Unit::TestCase
  include TestHelper

  # tokenizer.h:56-58 and :65, transcribed in docs/tokenizer-format-gpt2.md.
  MAGIC = 20_240_328
  VERSION = 2
  VOCAB_SIZE = 50_257
  EOT_TOKEN = 50_256
  FILE_BYTES = 372_108

  # "あ" is 3 UTF-8 bytes and GPT-2's byte-level BPE splits it into three
  # single-byte tokens, so any prefix of them is malformed UTF-8.
  HIRAGANA_A_IDS = [159, 223, 224].freeze
  HIRAGANA_A_BYTES = [227, 129, 130].freeze
  REPLACEMENT = "�"

  def tokenizer
    @tokenizer ||= NArrayLLM::GPT2::Tokenizer.load(require_data('gpt2_tokenizer.bin'))
  end

  def test_header_matches_tokenizer_h
    assert_equal(MAGIC, tokenizer.header[0], 'magic')
    assert_equal(VERSION, tokenizer.version, 'version')
    assert_equal(VOCAB_SIZE, tokenizer.vocab_size, 'vocab_size')
    assert_equal(VOCAB_SIZE, tokenizer.size)
  end

  # The id has to come out of header[3], not out of a constant in our code.
  def test_eot_token_is_read_from_the_header
    assert_equal(EOT_TOKEN, tokenizer.eot_token)
    assert_equal(tokenizer.header[3], tokenizer.eot_token)
    assert_true(tokenizer.eot?(EOT_TOKEN))
    assert_false(tokenizer.eot?(0))
  end

  def test_unused_header_slots_are_zero
    assert_equal([0], tokenizer.header[4..255].uniq)
  end

  def test_token_table_covers_the_whole_file
    total = (0...VOCAB_SIZE).sum { |id| 1 + tokenizer[id].bytesize }
    assert_equal(FILE_BYTES, 1024 + total)
    assert_equal(FILE_BYTES, File.size(data_path('gpt2_tokenizer.bin')))
  end

  def test_every_token_is_between_one_and_255_bytes
    lengths = (0...VOCAB_SIZE).map { |id| tokenizer[id].bytesize }
    assert_operator(lengths.min, :>=, 1, 'tokenizer.h:75 asserts length > 0')
    assert_operator(lengths.max, :<=, 255, 'train_gpt2.py:521 asserts length < 256')
  end

  def test_token_bytes_are_binary
    assert_equal(Encoding::ASCII_8BIT, tokenizer[0].encoding)
    assert_equal('!', tokenizer[0], 'GPT-2 token 0 is the byte 0x21')
    assert_equal('<|endoftext|>', tokenizer[EOT_TOKEN])
  end

  def test_decodes_a_known_token_sequence
    assert_equal('Hello, world!', tokenizer.decode([15_496, 11, 995, 0]))
    assert_equal(Encoding::UTF_8, tokenizer.decode([15_496]).encoding)
  end

  # --- byte-level BPE / UTF-8 ---

  def test_multibyte_character_split_across_tokens_reassembles
    assert_equal(HIRAGANA_A_BYTES, tokenizer.decode_bytes(HIRAGANA_A_IDS).bytes)
    assert_equal('あ', tokenizer.decode(HIRAGANA_A_IDS))
  end

  def test_a_partial_multibyte_character_is_scrubbed
    partial = tokenizer.decode(HIRAGANA_A_IDS[0, 2])
    assert_true(partial.valid_encoding?, 'decode must always return valid UTF-8')
    assert_equal(REPLACEMENT, partial)
    # The bytes themselves are kept intact; only the reinterpretation replaces.
    assert_equal(HIRAGANA_A_BYTES[0, 2], tokenizer.decode_bytes(HIRAGANA_A_IDS[0, 2]).bytes)
  end

  def test_decode_bytes_stays_binary_and_never_scrubs
    bytes = tokenizer.decode_bytes(HIRAGANA_A_IDS[0, 1])
    assert_equal(Encoding::ASCII_8BIT, bytes.encoding)
    assert_equal([227], bytes.bytes)
  end

  def test_accepts_an_integer_an_array_or_an_narray
    assert_equal('Hello', tokenizer.decode(15_496))
    assert_equal('Hello', tokenizer.decode([15_496]))
    assert_equal('Hello', tokenizer.decode(HM::Int32.cast([[15_496]])))
  end

  # --- errors ---

  def test_rejects_an_out_of_range_token_id
    assert_raise(NArrayLLM::Error) { tokenizer[VOCAB_SIZE] }
    assert_raise(NArrayLLM::Error) { tokenizer[-1] }
    assert_raise(NArrayLLM::Error) { tokenizer.decode([VOCAB_SIZE]) }
  end

  def test_rejects_a_bad_magic
    with_tokenizer_file(header: [1234, 2, 2, 1], tokens: %w[a b]) do |path|
      assert_raise(NArrayLLM::FormatError) { NArrayLLM::GPT2::Tokenizer.load(path) }
    end
  end

  def test_rejects_an_unsupported_version
    with_tokenizer_file(header: [MAGIC, 3, 2, 1], tokens: %w[a b]) do |path|
      assert_raise(NArrayLLM::FormatError) { NArrayLLM::GPT2::Tokenizer.load(path) }
    end
  end

  def test_rejects_trailing_bytes
    with_tokenizer_file(header: [MAGIC, 2, 1, 0], tokens: %w[a b]) do |path|
      assert_raise(NArrayLLM::FormatError) { NArrayLLM::GPT2::Tokenizer.load(path) }
    end
  end

  # tokenizer.h:59-63 -- version 1 has no EOT field, so llm.c uses 50256 and
  # asserts the vocabulary is the standard one.
  def test_version_one_falls_back_to_the_gpt2_eot_token
    with_tokenizer_file(header: [MAGIC, 1, VOCAB_SIZE, 0],
                        tokens: Array.new(VOCAB_SIZE) { 'x' }) do |path|
      loaded = NArrayLLM::GPT2::Tokenizer.load(path)
      assert_equal(1, loaded.version)
      assert_equal(EOT_TOKEN, loaded.eot_token)
    end
  end

  def test_version_one_with_a_non_standard_vocabulary_is_rejected
    with_tokenizer_file(header: [MAGIC, 1, 2, 0], tokens: %w[a b]) do |path|
      assert_raise(NArrayLLM::FormatError) { NArrayLLM::GPT2::Tokenizer.load(path) }
    end
  end

  private

  def with_tokenizer_file(header:, tokens:)
    path = File.join(Dir.tmpdir, "narray_llm_tokenizer_#{Process.pid}.bin")
    padded = header + [0] * (256 - header.size)
    body = tokens.map { |t| [t.bytesize].pack('C') + t }.join
    File.binwrite(path, padded.pack('L<*') + body)
    yield path
  ensure
    File.unlink(path) if path && File.exist?(path)
  end
end
