#!/usr/bin/env ruby
# frozen_string_literal: true

# Builds and runs the reference activation generator (script/llama2_dump.c).
#
#   ruby script/llama2_dump.rb                    # both models, 8 steps
#   ruby script/llama2_dump.rb --steps 16
#   ruby script/llama2_dump.rb stories260K.bin
#
# Writes data/<model>_debug_state.bin, which lib/narray_llm/models/llama2/
# debug_state.rb reads. Needs vendor/llama2.c/run.c, which
# script/download_llama2.rb fetches.

require 'fileutils'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
VENDOR_DIR = File.expand_path('../vendor/llama2.c', __dir__)
SOURCE = File.expand_path('llama2_dump.c', __dir__)
BINARY = File.join(VENDOR_DIR, 'llama2_dump')
DEFAULT_MODELS = %w[stories260K.bin stories110M.bin].freeze

def build
  run_c = File.join(VENDOR_DIR, 'run.c')
  unless File.exist?(run_c)
    abort "#{run_c} not found; run `ruby script/download_llama2.rb run.c` first"
  end
  return if File.exist?(BINARY) && File.mtime(BINARY) > [File.mtime(SOURCE), File.mtime(run_c)].max

  cc = ENV.fetch('CC', 'cc')
  warn "build #{BINARY}"
  ok = system(cc, '-O2', '-o', BINARY, SOURCE, '-lm', "-I#{VENDOR_DIR}")
  abort 'build failed' unless ok
end

def dump(model, steps)
  checkpoint = File.join(DATA_DIR, model)
  abort "#{checkpoint} not found; run `ruby script/download_llama2.rb` first" unless File.exist?(checkpoint)

  out = File.join(DATA_DIR, "#{File.basename(model, '.bin')}_debug_state.bin")
  warn "dump  #{model} -> #{File.basename(out)} (#{steps} steps)"
  output = IO.popen([BINARY, checkpoint, out, steps.to_s], &:read)
  abort "dump failed for #{model}" unless $?.success?

  puts output
  # forward_dump is a copy of run.c's forward; the binary compares the two and
  # says so. Anything but an exact match means the copy has drifted.
  abort 'forward_dump no longer matches run.c forward' unless output.include?('(exact)')
end

steps = 8
models = []
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/llama2_dump.rb [options] [models...]'
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
models = DEFAULT_MODELS if models.empty?

FileUtils.mkdir_p(VENDOR_DIR)
build
models.each { |m| dump(m, steps) }
