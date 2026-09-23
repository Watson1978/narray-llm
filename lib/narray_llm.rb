# frozen_string_literal: true

require 'json'

module NArrayLLM
  class Error < StandardError; end
  class FormatError < Error; end
end

# 実装に依らず共有するもの。
require_relative 'narray_llm/backend'
require_relative 'narray_llm/binary_io'
require_relative 'narray_llm/safetensors'
require_relative 'narray_llm/compare'
require_relative 'narray_llm/profiler'
require_relative 'narray_llm/ops'
require_relative 'narray_llm/backward'
require_relative 'narray_llm/adam_w'
require_relative 'narray_llm/kv_cache'
require_relative 'narray_llm/sampler'
require_relative 'narray_llm/generator'

# モデルごとの実装。
require_relative 'narray_llm/models/gpt2'
require_relative 'narray_llm/models/llama2'
require_relative 'narray_llm/models/mamba'
require_relative 'narray_llm/models/switch'
require_relative 'narray_llm/models/whisper'
require_relative 'narray_llm/models/resnet'
