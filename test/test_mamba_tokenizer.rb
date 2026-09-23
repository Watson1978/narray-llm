# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-mamba.md).
#
# The table is GPT-NeoX, written by mamba.c's tokenizer.py. The pieces below
# were read back from EleutherAI/gpt-neox-20b through transformers, so they
# check the file against its upstream rather than against itself.
class TestMambaTokenizer < Test::Unit::TestCase
  include TestHelper

  PIECES = {
    0 => '<|endoftext|>',
    1 => '<|padding|>',
    187 => "\n",
    209 => ' ',
    15_496 => ' counting',
    50_000 => ' Playing',
    50_275 => '   ',
    50_276 => '  '
  }.freeze

  def tokenizer
    require_data('mamba_tokenizer.bin')
    NArrayLLM::Mamba::Tokenizer.load(data_path('mamba_tokenizer.bin'))
  end

  # tokenizer.py writes 50277 entries, which is what the model's vocab_size is
  # before mamba.c rounds it up for the stored tables.
  def test_table_size_matches_the_model_vocabulary
    assert_equal(50_277, tokenizer.vocab_size)
  end

  def test_pieces_match_the_upstream_tokenizer
    table = tokenizer
    PIECES.each do |id, text|
      assert_equal(text, table[id].dup.force_encoding(Encoding::UTF_8), "token #{id}")
    end
  end

  def test_file_size_accounts_for_every_byte
    table = tokenizer
    assert_equal(File.size(data_path('mamba_tokenizer.bin')), table.expected_file_size)
  end

  def test_out_of_range_ids_are_rejected
    table = tokenizer
    assert_raise_kind_of(NArrayLLM::Error) { table[-1] }
    assert_raise_kind_of(NArrayLLM::Error) { table[50_277] }
  end

  # mamba.c:573 drops the space a piece carries when it follows the delimiter,
  # and mamba.c makes BOS and EOS the same token (mamba.c:506).
  def test_leading_space_is_dropped_after_the_delimiter
    table = tokenizer
    assert_equal(0, table.eot_token)
    assert_equal('Playing', table.decode_piece(0, 50_000))
    assert_equal(' Playing', table.decode_piece(187, 50_000))
  end

  def test_render_joins_every_token_after_the_first
    table = tokenizer
    assert_equal("\n counting", table.render([0, 187, 15_496]))
  end
end
