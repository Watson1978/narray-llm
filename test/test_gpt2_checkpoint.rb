# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-gpt2.md).
#
# Every fixture below is transcribed from docs/checkpoint-format-gpt2.md, which in
# turn cites llm.c by function and line. Nothing here is guessed.
class TestCheckpoint < Test::Unit::TestCase
  include TestHelper

  # train_gpt2.c:722-727. GPT-2 124M nominal config.
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 50_257
  PADDED_VOCAB_SIZE = 50_304
  NUM_LAYERS = 12
  NUM_HEADS = 12
  CHANNELS = 768

  # fill_in_parameter_sizes (train_gpt2.c:556-577) and the pointer order in
  # malloc_and_point_parameters (train_gpt2.c:588-592).
  TENSOR_SHAPES = [
    [:wte,      [50_304, 768]],
    [:wpe,      [1024, 768]],
    [:ln1w,     [12, 768]],
    [:ln1b,     [12, 768]],
    [:qkvw,     [12, 2304, 768]],
    [:qkvb,     [12, 2304]],
    [:attprojw, [12, 768, 768]],
    [:attprojb, [12, 768]],
    [:ln2w,     [12, 768]],
    [:ln2b,     [12, 768]],
    [:fcw,      [12, 3072, 768]],
    [:fcb,      [12, 3072]],
    [:fcprojw,  [12, 768, 3072]],
    [:fcprojb,  [12, 768]],
    [:lnfw,     [768]],
    [:lnfb,     [768]]
  ].freeze

  NUM_PARAMETERS = 124_475_904
  CHECKPOINT_BYTES = 1024 + 4 * NUM_PARAMETERS

  # Reference values read out of HuggingFace openai-community/gpt2
  # model.safetensors (dtype F32) at the offsets its own header declares. That
  # is the same tensor data train_gpt2.py:216 exports from, so fp32 equality is
  # exact, not approximate. See the last section of docs/checkpoint-format-gpt2.md.
  HF_REFERENCE = {
    # [tensor, index expression, expected first 5 floats]
    wte_row0: [-0.110103011, -0.0392667241, 0.033107508, 0.13382645, -0.0484756939],
    wte_row50256: [0.0513520129, -0.0276890472, 0.0499369018, -0.0422121696, -0.0616769791],
    wpe_row0: [-0.0188207198, -0.1974186, 0.00402672496, 0.0113468589, 0.0638241172],
    ln1w_layer0: [0.223220333, 0.18195866, 0.153432459, 0.191682562, 0.203618452],
    ln1b_layer0: [-0.00367732509, 0.027196737, -0.0640409067, -0.00496289041, -0.0156569183],
    qkvb_layer0: [0.48033914, -0.525432587, -0.429264545, -0.205952495, -0.127733797],
    fcb_layer0: [0.0396194793, -0.0881253183, -0.140249088, -0.0331509039, -0.0160597693],
    lnfw: [1.39708042, 1.37495291, 1.88695681, 1.16883683, 1.27238488],
    lnfb: [0.00108716474, 0.0365293846, -0.0672961622, 0.000164160519, -0.067443952],
    # HF stores Conv1D weights as [in, out]; train_gpt2.py:223-232 transposes
    # them on export. So llm.c qkvw[0][i][j] == HF h.0.attn.c_attn.weight[j][i],
    # and these are HF column 0, i.e. our row 0.
    qkvw_layer0_row0: [-0.473848403, 0.087422058, 0.00388936442, 0.221499607, -0.094702445],
    fcprojw_layer0_row0: [-0.106606409, 0.0363972858, -0.0766659304, 0.0923936069, -0.037520349]
  }.freeze

  def checkpoint
    @checkpoint ||= NArrayLLM::GPT2::Checkpoint.load(require_data('gpt2_124M.bin'))
  end

  # --- 受け入れ条件 1: ヘッダから読んだ設定が公称値と一致する ---

  def test_header_config_matches_gpt2_124m
    config = checkpoint.config
    assert_equal(NArrayLLM::GPT2::Checkpoint::MAGIC, checkpoint.header[0], 'magic')
    assert_equal(NArrayLLM::GPT2::Checkpoint::VERSION, checkpoint.header[1], 'version')
    assert_equal(MAX_SEQ_LEN, config.max_seq_len, 'maxT')
    assert_equal(VOCAB_SIZE, config.vocab_size, 'V')
    assert_equal(NUM_LAYERS, config.num_layers, 'L')
    assert_equal(NUM_HEADS, config.num_heads, 'NH')
    assert_equal(CHANNELS, config.channels, 'C')
    assert_equal(PADDED_VOCAB_SIZE, config.padded_vocab_size, 'Vp')
  end

  def test_unused_header_slots_are_zero
    path = require_data('gpt2_124M.bin')
    tail = raw_int32(path, 8 * 4, 248)
    assert_equal([0], tail.uniq, 'header[8..255] must be zero padding')
  end

  def test_rejects_wrong_magic
    path = File.join(Dir.tmpdir, 'narray_llm_bad_magic.bin')
    File.binwrite(path, [1234, 3].pack('l<2') + ("\0" * (1024 - 8)))
    assert_raise(NArrayLLM::FormatError) { NArrayLLM::GPT2::Checkpoint.load(path) }
  ensure
    File.unlink(path) if path && File.exist?(path)
  end

  # --- 受け入れ条件 2: 全パラメータの要素数の合計が実データサイズと一致する ---

  def test_total_parameter_count_matches_file_size
    path = require_data('gpt2_124M.bin')
    assert_equal(NUM_PARAMETERS, checkpoint.num_parameters, 'num_parameters')
    assert_equal(NUM_PARAMETERS, TENSOR_SHAPES.sum { |_, shape| shape.inject(:*) },
                 'fixture shapes must sum to num_parameters')
    assert_equal(CHECKPOINT_BYTES, File.size(path), 'file size')
    assert_equal(File.size(path), checkpoint.expected_file_size, 'header + params covers the whole file')
  end

  def test_loaded_element_counts_sum_to_num_parameters
    total = NArrayLLM::GPT2::Checkpoint::TENSOR_NAMES.sum { |name| checkpoint[name].size }
    assert_equal(NUM_PARAMETERS, total)
  end

  # --- 受け入れ条件 3: 各テンソルの shape 一覧が llm.c と一致する ---

  def test_tensor_names_and_order_match_llm_c
    assert_equal(TENSOR_SHAPES.map(&:first), NArrayLLM::GPT2::Checkpoint::TENSOR_NAMES)
  end

  def test_tensor_shapes_match_llm_c
    TENSOR_SHAPES.each do |name, shape|
      assert_equal(shape, checkpoint[name].shape, "shape of #{name}")
      assert_equal(XM::SFloat, checkpoint[name].class, "dtype of #{name}")
    end
  end

  def test_tensor_byte_offsets_are_contiguous_from_the_header
    offset = 1024
    TENSOR_SHAPES.each do |name, shape|
      assert_equal(offset, checkpoint.byte_offset(name), "byte offset of #{name}")
      offset += 4 * shape.inject(:*)
    end
    assert_equal(CHECKPOINT_BYTES, offset)
  end

  # --- 受け入れ条件 4: 数本のテンソルの先頭数要素が参照と一致する ---

  def test_head_values_match_huggingface_reference
    assert_bit_identical(HF_REFERENCE[:wte_row0], checkpoint[:wte][0, 0...5], 'wte[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:wte_row50256], checkpoint[:wte][50_256, 0...5], 'wte[50256, 0...5]')
    assert_bit_identical(HF_REFERENCE[:wpe_row0], checkpoint[:wpe][0, 0...5], 'wpe[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:ln1w_layer0], checkpoint[:ln1w][0, 0...5], 'ln1w[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:ln1b_layer0], checkpoint[:ln1b][0, 0...5], 'ln1b[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:qkvb_layer0], checkpoint[:qkvb][0, 0...5], 'qkvb[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:fcb_layer0], checkpoint[:fcb][0, 0...5], 'fcb[0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:lnfw], checkpoint[:lnfw][0...5], 'lnfw[0...5]')
    assert_bit_identical(HF_REFERENCE[:lnfb], checkpoint[:lnfb][0...5], 'lnfb[0...5]')
  end

  # The subtlest fact in the format: weight matrices are stored [out, in], not
  # HuggingFace's Conv1D [in, out]. If this passes, the transpose claim holds.
  def test_weight_matrices_are_stored_out_by_in
    assert_bit_identical(HF_REFERENCE[:qkvw_layer0_row0], checkpoint[:qkvw][0, 0, 0...5],
                         'qkvw[0, 0, 0...5]')
    assert_bit_identical(HF_REFERENCE[:fcprojw_layer0_row0], checkpoint[:fcprojw][0, 0, 0...5],
                         'fcprojw[0, 0, 0...5]')
  end

  # pad_vocab(..., value=0) at train_gpt2.py:429. Any offset error anywhere in
  # wte would break this.
  def test_wte_padding_rows_are_exactly_zero
    padding = checkpoint[:wte][VOCAB_SIZE...PADDED_VOCAB_SIZE, true]
    assert_equal([PADDED_VOCAB_SIZE - VOCAB_SIZE, CHANNELS], padding.shape)
    assert_equal(0.0, NArrayLLM.scalar(padding.abs.max))
    refute_equal(0.0, NArrayLLM.scalar(checkpoint[:wte][VOCAB_SIZE - 1, true].abs.max),
                 'the last real row must not be zero')
  end

  def test_loader_agrees_with_an_independent_raw_read
    path = require_data('gpt2_124M.bin')
    offset = 1024
    TENSOR_SHAPES.each do |name, shape|
      count = shape.inject(:*)
      tensor = checkpoint[name].flatten
      assert_bit_identical(raw_fp32(path, offset, 5), tensor[0...5], "head of #{name}")
      assert_bit_identical(raw_fp32(path, offset + 4 * (count - 5), 5), tensor[(count - 5)...count],
                           "tail of #{name}")
      offset += 4 * count
    end
  end
end
