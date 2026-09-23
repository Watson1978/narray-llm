#!/usr/bin/env ruby
# frozen_string_literal: true

# Downloads what the Mamba implementation needs.
#
# The reference is kroggen/mamba.c on its default `learning` branch, which is
# the Mamba 1 version. There is no distributed .bin: the weights come from
# HuggingFace as PyTorch and mamba.c's own export.py converts them.
#
#   ruby script/download_mamba.rb              # sources + mamba-130m weights
#   ruby script/download_mamba.rb --sources    # just the C and Python sources
#   ruby script/download_mamba.rb config.json
#   FORCE=1 ruby script/download_mamba.rb      # re-download even if present

require 'open3'
require 'fileutils'
require 'optparse'

MAMBA_C = 'https://raw.githubusercontent.com/kroggen/mamba.c/learning/'
HF = 'https://huggingface.co/state-spaces/mamba-130m/resolve/main/'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
MODEL_DIR = File.join(DATA_DIR, 'mamba-130m')
VENDOR_DIR = File.expand_path('../vendor/mamba.c', __dir__)

SOURCES = {
  'mamba.c' => ["#{MAMBA_C}mamba.c", VENDOR_DIR],
  'export.py' => ["#{MAMBA_C}export.py", VENDOR_DIR],
  'tokenizer.py' => ["#{MAMBA_C}tokenizer.py", VENDOR_DIR],
  'makefile' => ["#{MAMBA_C}makefile", VENDOR_DIR],
  'config.json' => ["#{HF}config.json", MODEL_DIR],
  'pytorch_model.bin' => ["#{HF}pytorch_model.bin", MODEL_DIR]
}.freeze

SOURCE_FILES = %w[mamba.c export.py tokenizer.py makefile].freeze
WEIGHT_FILES = %w[config.json pytorch_model.bin].freeze

def remote_size(url)
  out, _err, status = Open3.capture3('curl', '-sSL', '-I', url)
  return nil unless status.success?

  out.scan(/^content-length:\s*(\d+)/i).flatten.last&.to_i
end

def download(name)
  url, dir = SOURCES.fetch(name) { raise ArgumentError, "unknown file: #{name}" }
  FileUtils.mkdir_p(dir)
  path = File.join(dir, name)
  expected = remote_size(url)

  if File.exist?(path) && ENV['FORCE'].to_s.empty?
    have = File.size(path)
    if expected.nil?
      warn "skip  #{name} (#{have} bytes, could not verify remote size)"
      return
    elsif have == expected
      warn "skip  #{name} (#{have} bytes, already complete)"
      return
    else
      warn "redo  #{name} (#{have} bytes, expected #{expected})"
    end
  end

  warn "get   #{name}#{expected ? " (#{expected} bytes)" : ''}"
  part = "#{path}.part"
  ok = system('curl', '--fail', '--location', '--progress-bar', '-C', '-',
              '-o', part, url)
  raise "download failed: #{name}" unless ok

  if expected && File.size(part) != expected
    raise "size mismatch for #{name}: got #{File.size(part)}, expected #{expected}"
  end

  File.rename(part, path)
  warn "ok    #{name}"
end

argv = ARGV.dup
everything = false
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/download_mamba.rb [options] [files...]'
  o.on('--sources', '参照の C だけ取る (重みは取らない)') { everything = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

# A file named on the command line wins over both defaults.
files =
  if !argv.empty?
    argv
  elsif everything
    SOURCE_FILES
  else
    SOURCE_FILES + WEIGHT_FILES
  end

warn "data dir:   #{DATA_DIR}"
warn "vendor dir: #{VENDOR_DIR}"
files.each { |f| download(f) }

exit if ARGV.include?('--sources')

warn ''
warn 'next, convert the weights and export the tokenizer:'
warn "  cd #{VENDOR_DIR} && ../../python/.venv/bin/python export.py \\"
warn "    #{MODEL_DIR} #{File.join(DATA_DIR, 'mamba-130m.bin')}"
warn "  cd #{VENDOR_DIR} && ../../python/.venv/bin/python tokenizer.py"
warn "  mv #{VENDOR_DIR}/tokenizer.bin #{File.join(DATA_DIR, 'mamba_tokenizer.bin')}"
