#!/usr/bin/env ruby
# frozen_string_literal: true

# Builds and runs the reference activation generator (script/mamba_dump.c).
#
#   ruby script/mamba_dump.rb                     # both models
#   ruby script/mamba_dump.rb --steps 16
#   ruby script/mamba_dump.rb mamba-130m.bin
#
# Writes data/<model>_debug_state.bin, which lib/narray_llm/models/mamba/
# debug_state.rb reads. Needs vendor/mamba.c/mamba.c, which
# script/download_mamba.rb fetches.
#
# mamba_tiny is fed a fixed sequence rather than decoded greedily: its argmax
# is a fixed point at whatever token went in, so greedy decoding would never
# vary the input and the embedding lookup would go unchecked.

require 'fileutils'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
VENDOR_DIR = File.expand_path('../vendor/mamba.c', __dir__)
SOURCE = File.expand_path('mamba_dump.c', __dir__)
BINARY = File.join(VENDOR_DIR, 'mamba_dump')

# model => the 4th argument: a token list, or the token to start greedy from.
DEFAULT_MODELS = {
  'mamba_tiny.bin' => '3,17,42,8,0,61',
  'mamba-130m.bin' => '9038'
}.freeze

def build
  source = File.join(VENDOR_DIR, 'mamba.c')
  unless File.exist?(source)
    abort "#{source} not found; run `ruby script/download_mamba.rb --sources` first"
  end
  return if File.exist?(BINARY) && File.mtime(BINARY) > [File.mtime(SOURCE), File.mtime(source)].max

  cc = ENV.fetch('CC', 'cc')
  warn "build #{BINARY}"
  ok = system(cc, '-O2', '-o', BINARY, SOURCE, '-lm', "-I#{VENDOR_DIR}")
  abort 'build failed' unless ok
end

def dump(model, steps, tokens)
  checkpoint = File.join(DATA_DIR, model)
  unless File.exist?(checkpoint)
    abort "#{checkpoint} not found; run `ruby script/download_mamba.rb` first"
  end

  out = File.join(DATA_DIR, "#{File.basename(model, '.bin')}_debug_state.bin")
  warn "dump  #{model} -> #{File.basename(out)}"
  args = [BINARY, checkpoint, out, steps.to_s]
  args << tokens if tokens
  output = IO.popen(args, &:read)
  abort "dump failed for #{model}" unless $?.success?

  puts output
  # forward_dump is a copy of mamba.c's forward; the binary compares the two
  # and says so. Anything but an exact match means the copy has drifted.
  abort 'forward_dump no longer matches mamba.c forward' unless output.include?('(exact)')
end

steps = 16
models = []
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/mamba_dump.rb [options] [models...]'
  o.on('--steps N', Integer, 'デコードするステップ数') { |v| steps = v }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end
models.concat(argv)

FileUtils.mkdir_p(VENDOR_DIR)
build
if models.empty?
  DEFAULT_MODELS.each { |model, tokens| dump(model, steps, tokens) }
else
  models.each { |m| dump(m, steps, DEFAULT_MODELS[m]) }
end
