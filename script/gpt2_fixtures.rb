#!/usr/bin/env ruby
# frozen_string_literal: true

# Writes the greedy token sequences other implementations are checked against.
#
#   ruby script/gpt2_fixtures.rb python/fixtures/gpt2_124M_greedy.json
#
# Greedy decoding is deterministic, so these are an exact expectation, not a
# tolerance. The KV cache and the recomputing path agree bit for bit here, so
# either can produce them.

require 'json'
require_relative '../lib/narray_llm'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
LENGTHS = [64, 256].freeze

out_path = ARGV[0] || File.expand_path('../python/fixtures/gpt2_124M_greedy.json', __dir__)

model = NArrayLLM::GPT2::Model.load(File.join(DATA_DIR, 'gpt2_124M.bin'))
tokenizer = NArrayLLM::GPT2::Tokenizer.load(File.join(DATA_DIR, 'gpt2_tokenizer.bin'))
generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)

fixture = {
  'source' => 'narray-llm (Ruby)',
  'backend' => NArrayLLM.gpu? ? 'Cumo' : 'Numo',
  'prompt' => [tokenizer.eot_token],
  'eot_token' => tokenizer.eot_token,
  'sequences' => {}
}

LENGTHS.each do |length|
  tokens = generator.generate([tokenizer.eot_token], max_new_tokens: length, cache: true)
  generated = tokens[1..]
  fixture['sequences'][length.to_s] = generated
  warn "length=#{length}: #{tokenizer.decode(generated)[0, 60].inspect}..."
end

File.write(out_path, "#{JSON.pretty_generate(fixture)}\n")
warn "wrote #{out_path}"
