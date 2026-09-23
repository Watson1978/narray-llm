#!/usr/bin/env ruby
# frozen_string_literal: true

# Downloads the llama2.c checkpoints and tokenizers this project needs.
#
# Sources: the tinyllamas model table in the llama2.c README for the
# checkpoints, and llama2.c's own repository for the 32000-entry tokenizer.
# test_all.py is what names the four stories260K files.
#
#   ruby script/download_llama2.rb                 # stage 0-2 needs these
#   ruby script/download_llama2.rb --all           # plus the MHA models
#   ruby script/download_llama2.rb stories110M.bin
#   FORCE=1 ruby script/download_llama2.rb         # re-download even if present

require 'open3'
require 'fileutils'
require 'optparse'

TINYLLAMAS = 'https://huggingface.co/karpathy/tinyllamas/resolve/main/'
LLAMA2C = 'https://raw.githubusercontent.com/karpathy/llama2.c/master/'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
# run.c is a source file, not data: script/llama2_dump.c includes it to build
# the reference generator. Kept out of data/ so NARRAY_LLM_DATA can point
# anywhere without moving it.
VENDOR_DIR = File.expand_path('../vendor/llama2.c', __dir__)

# name => [URL, destination]. stories260K lives in a subdirectory with its own
# vocab-512 tokenizer; the larger models share llama2.c's 32000-entry one.
SOURCES = {
  'stories260K.bin' => ["#{TINYLLAMAS}stories260K/stories260K.bin", DATA_DIR],
  'tok512.bin' => ["#{TINYLLAMAS}stories260K/tok512.bin", DATA_DIR],
  'stories110M.bin' => ["#{TINYLLAMAS}stories110M.bin", DATA_DIR],
  'tokenizer.bin' => ["#{LLAMA2C}tokenizer.bin", DATA_DIR],
  'run.c' => ["#{LLAMA2C}run.c", VENDOR_DIR],
  'runq.c' => ["#{LLAMA2C}runq.c", VENDOR_DIR],
  'export.py' => ["#{LLAMA2C}export.py", VENDOR_DIR],
  'model.py' => ["#{LLAMA2C}model.py", VENDOR_DIR],
  'stories15M.pt' => ["#{TINYLLAMAS}stories15M.pt", DATA_DIR],
  'stories110M.pt' => ["#{TINYLLAMAS}stories110M.pt", DATA_DIR],
  'stories15M.bin' => ["#{TINYLLAMAS}stories15M.bin", DATA_DIR],
  'stories42M.bin' => ["#{TINYLLAMAS}stories42M.bin", DATA_DIR]
}.freeze

REQUIRED = %w[stories260K.bin tok512.bin stories110M.bin tokenizer.bin run.c].freeze
OPTIONAL = %w[stories15M.bin stories42M.bin runq.c export.py model.py stories15M.pt stories110M.pt].freeze

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
  o.banner = 'Usage: ruby script/download_llama2.rb [options] [files...]'
  o.on('--all', '任意のファイルも取る') { everything = true }
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
    REQUIRED + OPTIONAL
  else
    REQUIRED
  end

warn "data dir:   #{DATA_DIR}"
warn "vendor dir: #{VENDOR_DIR}"
files.each { |f| download(f) }
