#!/usr/bin/env ruby
# frozen_string_literal: true

# Writes the greedy token sequences the Python implementations are checked
# against.
#
#   ruby script/llama2_fixtures.rb
#   ruby script/llama2_fixtures.rb --length 256 stories110M
#
# The whole sequence is stored, BOS included, so a shorter run is a prefix of
# it and one fixture covers every length. Greedy decoding is deterministic, so
# this is an exact expectation, not a tolerance. The Ruby output is itself held
# to run.c byte for byte (test/test_llama2_generate.rb).

require 'json'
require_relative '../lib/narray_llm'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
FIXTURE_DIR = File.expand_path('../python/fixtures', __dir__)
TOKENIZERS = { 'stories260K' => 'tok512.bin' }.freeze
DEFAULT_TOKENIZER = 'tokenizer.bin'

length = 256
models = []
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/llama2_fixtures.rb [options] [models...]'
  o.on('--length N', Integer, '生成するトークン数') { |v| length = v }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end
models.concat(argv.map { |m| m.sub(/\.bin\z/, '') })
models = %w[stories260K stories110M] if models.empty?

models.each do |name|
  path = File.join(DATA_DIR, "#{name}.bin")
  quantized = File.binread(path, 4).unpack1('L<') ==
              NArrayLLM::Llama2::QuantizedCheckpoint::MAGIC
  model = (quantized ? NArrayLLM::Llama2::QuantizedModel : NArrayLLM::Llama2::Model).load(path)
  tokenizer = NArrayLLM::Llama2::Tokenizer.load(
    File.join(DATA_DIR, TOKENIZERS.fetch(name, DEFAULT_TOKENIZER)),
    vocab_size: model.config.vocab_size
  )
  generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
  bos = NArrayLLM::Llama2::Tokenizer::BOS_TOKEN
  tokens = generator.generate([bos], max_new_tokens: length, cache: true)

  fixture = {
    'source' => 'narray-llm (Ruby)',
    'backend' => NArrayLLM.gpu? ? 'Cumo' : 'Numo',
    'model' => name,
    'prompt' => [bos],
    'stop_token' => tokenizer.eot_token,
    'tokens' => tokens
  }
  out_path = File.join(FIXTURE_DIR, "#{name}_greedy.json")
  File.write(out_path, "#{JSON.pretty_generate(fixture)}\n")
  warn "#{name}: #{tokens.size} tokens -> #{out_path}"
  warn "  #{tokenizer.render(tokens)[0, 60].inspect}..."
end
