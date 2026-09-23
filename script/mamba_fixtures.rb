#!/usr/bin/env ruby
# frozen_string_literal: true

# Writes the greedy token sequence the acceptance tests are checked against.
#
#   ruby script/mamba_fixtures.rb
#   ruby script/mamba_fixtures.rb --length 512 mamba-130m
#
# Unlike the Llama 2 fixtures, the source here is the C reference rather than
# this repository: script/mamba_dump.c drives mamba.c's own forward() and feeds
# its own argmax back in. So the expectation is mamba.c's, not ours.
#
# mamba.c's CLI cannot produce this. It insists on a prompt and encodes it with
# the 50277 entry tokenizer, so there is no way to ask it to start from
# <|endoftext|> alone or to point it at a small checkpoint.

require 'json'
require 'fileutils'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
FIXTURE_DIR = File.expand_path('../python/fixtures', __dir__)
VENDOR_DIR = File.expand_path('../vendor/mamba.c', __dir__)
BINARY = File.join(VENDOR_DIR, 'mamba_dump')
BOS = 0

length = 256
models = []
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/mamba_fixtures.rb [options] [models...]'
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
models = %w[mamba-130m] if models.empty?

unless File.exist?(BINARY)
  abort "#{BINARY} not found; run `ruby script/mamba_dump.rb` first"
end

models.each do |name|
  checkpoint = File.join(DATA_DIR, "#{name}.bin")
  abort "#{checkpoint} not found; run `ruby script/download_mamba.rb` first" unless File.exist?(checkpoint)

  # "-" asks for the token sequence without the activations, which at this
  # length would be gigabytes.
  output = IO.popen([BINARY, checkpoint, '-', length.to_s, BOS.to_s], &:read)
  abort "reference failed for #{name}" unless $?.success?
  abort 'forward_dump no longer matches mamba.c forward' unless output.include?('(exact)')

  line = output[/^tokens:(.*)$/, 1] or abort "no token line for #{name}"
  tokens = line.split.map(&:to_i)
  # The binary prints the token fed at each position, so the last one it chose
  # is on the "next" line.
  tokens << Integer(output[/^next: (\d+)$/, 1])

  fixture = {
    'source' => 'kroggen/mamba.c via script/mamba_dump.c',
    'model' => name,
    'prompt' => [BOS],
    'stop_token' => BOS,
    'tokens' => tokens
  }
  FileUtils.mkdir_p(FIXTURE_DIR)
  out_path = File.join(FIXTURE_DIR, "#{name}_greedy.json")
  File.write(out_path, "#{JSON.pretty_generate(fixture)}\n")
  warn "#{name}: #{tokens.size} tokens -> #{out_path}"
end
