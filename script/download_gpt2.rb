#!/usr/bin/env ruby
# frozen_string_literal: true

# Downloads the llm.c starter pack files this project needs.
#
# Source of the URL and file list: llm.c dev/download_starter_pack.sh, which the
# llm.c README points at ("./dev/download_starter_pack.sh").
#
#   ruby script/download_gpt2.rb                 # only what stage 0-2 needs
#   ruby script/download_gpt2.rb --all           # every starter pack file
#   ruby script/download_gpt2.rb gpt2_tokenizer.bin
#   FORCE=1 ruby script/download_gpt2.rb         # re-download even if present

require 'open3'
require 'optparse'

BASE_URL = 'https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/'

REQUIRED = %w[
  gpt2_124M.bin
  gpt2_124M_debug_state.bin
  gpt2_tokenizer.bin
].freeze

OPTIONAL = %w[
  gpt2_124M_bf16.bin
  tiny_shakespeare_train.bin
  tiny_shakespeare_val.bin
  hellaswag_val.bin
].freeze

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }

def remote_size(url)
  out, _err, status = Open3.capture3('curl', '-sSL', '-I', url)
  return nil unless status.success?

  # -L prints one header block per hop; HF redirects to a CDN, so take the last.
  out.scan(/^content-length:\s*(\d+)/i).flatten.last&.to_i
end

def download(name)
  path = File.join(DATA_DIR, name)
  url = "#{BASE_URL}#{name}?download=true"
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
  # -C - resumes a partial file, so an interrupted multi-hundred-MB download
  # does not start over.
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
  o.banner = 'Usage: ruby script/download_gpt2.rb [options] [files...]'
  o.on('--all', 'starter pack のファイルを全部取る') { everything = true }
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

Dir.mkdir(DATA_DIR) unless Dir.exist?(DATA_DIR)
warn "data dir: #{DATA_DIR}"
files.each { |f| download(f) }
