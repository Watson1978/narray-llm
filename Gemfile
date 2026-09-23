source 'https://rubygems.org'

# numo-narray-alt is the maintained fork; it installs the same `Numo` namespace
# and `require "numo/narray"` works unchanged. Plain numo-narray also works.
gem 'numo-narray-alt', '~> 0.11'

# Numo の dot が BLAS を通るのはこれが require されているときだけ。無くても
# 動くが、GEMM が単一スレッドの broadcast + mulsum に落ちて 27 倍遅くなる。
gem 'numo-linalg-alt', '~> 0.10'

gem 'rake'
gem 'test-unit', '~> 3.6'

# GPU backend. Needed only with GPU=1, and it wants a CUDA toolkit to build,
# so the group is optional: a plain `bundle install` skips it and the whole
# suite still passes on Numo. Install it with
#
#   bundle config set --local with gpu
#   CUMO_NVCC_GENERATE_CODE=arch=compute_120,code=sm_120 bundle install
#
# The nvcc flag is what keeps the first launch from paying a JIT.
group :gpu, optional: true do
  gem 'cumo'
end
