# frozen_string_literal: true

# Numo is always loaded: even with the GPU backend, token ids and indices stay
# on the host (an NArray index on the device forces a sync).
require 'numo/narray'
HM = Numo

# Numo's dot only reaches BLAS when Numo::Linalg is defined; without it the
# fallback is a single-threaded broadcast + mulsum, about 27x slower here.
# Optional so that the tests still run where the gem is not installed.
begin
  require 'numo/linalg'
rescue LoadError
  nil
end

if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  require 'cumo/narray'
  XM = Cumo
else
  XM = Numo
end

# The dtype the forward pass holds every tensor in. fp32 is the default and the
# acceptance tests depend on it: the token sequence has to match llm.c exactly
# and fp16 parts from it at the third token. DTYPE picks another one for
# measurement. Weights stay fp32 on disk whatever it says; Ops.contiguous
# casts them on load.
XF =
  case (dtype = ENV['DTYPE'].to_s.downcase)
  when '', 'fp32', 'sfloat'
    XM::SFloat
  when 'fp16', 'hfloat', 'bf16', 'bfloat'
    name = %w[fp16 hfloat].include?(dtype) ? :HFloat : :BFloat
    unless XM.const_defined?(name)
      raise NArrayLLM::Error,
            "DTYPE=#{dtype} needs a backend with #{name}; #{XM.name} has none (GPU=1 selects Cumo)"
    end
    XM.const_get(name)
  else
    raise NArrayLLM::Error, "unknown DTYPE #{ENV['DTYPE'].inspect}; use fp32, fp16 or bf16"
  end

module NArrayLLM
  module_function

  # fp16 is the only one of the three whose exponent range is narrower than
  # fp32's, so it is the only one that needs the clipped mask and exp limit.
  def fp16?
    XM.const_defined?(:HFloat) && XF == XM::HFloat
  end

  def bf16?
    XM.const_defined?(:BFloat) && XF == XM::BFloat
  end

  # Both 16-bit types carry too few mantissa bits to hold a token id: fp16
  # loses them past 2048, bf16 past 256.
  def reduced_precision?
    XF != XM::SFloat
  end

  def dtype_name
    return 'fp16' if fp16?
    return 'bf16' if bf16?

    'fp32'
  end

  # Cumo reductions return a 0-d NArray where Numo returns a Ruby Numeric.
  # Reading it back blocks until the queue drains, so only call this when
  # printing or deciding something -- never inside a hot loop.
  def scalar(value)
    return value if value.is_a?(Numeric)

    value.respond_to?(:extract_cpu) ? value.extract_cpu : value.extract
  end

  def gpu?
    XM.name == 'Cumo'
  end

  def blas?
    defined?(Numo::Linalg) ? true : false
  end
end
