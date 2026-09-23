# frozen_string_literal: true

$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))

require 'tmpdir'
require 'test/unit'
require 'narray_llm'

module TestHelper
  DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }

  def data_path(name)
    File.join(DATA_DIR, name)
  end

  # Missing .bin files are an omission, not a failure: they are several hundred
  # MB and a fresh clone has not run `rake prepare` yet.
  def require_data(name)
    path = data_path(name)
    omit("#{name} not found under #{DATA_DIR}; run `rake prepare`") unless File.exist?(path)
    path
  end

  # Little-endian fp32, read straight out of the file. Deliberately a different
  # code path from the loader's from_binary, so the two can be cross-checked.
  def raw_fp32(path, byte_offset, count)
    File.binread(path, count * 4, byte_offset).unpack('e*')
  end

  def raw_int32(path, byte_offset, count)
    File.binread(path, count * 4, byte_offset).unpack('l<*')
  end

  # Cumo element access returns a 0-d NArray where Numo returns a Ruby Float,
  # and `0d_narray > x` yields a Bit array, which is truthy even when the
  # comparison is false. Without this, assert_operator passes vacuously on GPU.
  def host(value)
    NArrayLLM.scalar(value)
  end

  def assert_bit_identical(expected_floats, actual_tensor, message = nil)
    expected = XM::SFloat[*expected_floats]
    actual = actual_tensor.flatten
    assert_equal(expected.size, actual.size, "#{message}: length")
    diff = NArrayLLM.scalar((actual - expected).abs.max)
    assert_equal(0.0, diff, "#{message}: expected #{expected.to_a.inspect}, got #{actual.to_a.inspect}")
  end
end
