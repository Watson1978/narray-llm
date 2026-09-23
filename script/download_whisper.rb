#!/usr/bin/env ruby
# frozen_string_literal: true

# Downloads what the Whisper implementation needs.
#
# There is no C reference for this one either, so nothing is vendored: the
# reference is transformers. Unlike switch-base-8 this model publishes a
# safetensors, so there is no conversion step and lib/narray_llm/safetensors.rb
# reads it directly.
#
#   ruby script/download_whisper.rb            # config, tokenizer and weights
#   ruby script/download_whisper.rb config.json
#   FORCE=1 ruby script/download_whisper.rb    # re-download even if present

require 'open3'
require 'fileutils'
require 'optparse'

HF = 'https://huggingface.co/openai/whisper-tiny/resolve/main/'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
MODEL_DIR = File.join(DATA_DIR, 'whisper-tiny')

FILES = %w[config.json generation_config.json preprocessor_config.json
           tokenizer.json tokenizer_config.json special_tokens_map.json
           added_tokens.json vocab.json merges.txt normalizer.json
           model.safetensors].freeze

def remote_size(url)
  out, _err, status = Open3.capture3('curl', '-sSL', '-I', url)
  return nil unless status.success?

  out.scan(/^content-length:\s*(\d+)/i).flatten.last&.to_i
end

def download(name)
  FileUtils.mkdir_p(MODEL_DIR)
  path = File.join(MODEL_DIR, name)
  url = "#{HF}#{name}"
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

  raise "size mismatch for #{name}: got #{File.size(part)}, expected #{expected}" if expected && File.size(part) != expected

  File.rename(part, path)
  warn "ok    #{name}"
end

argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/download_whisper.rb [files...]'
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

files = argv.empty? ? FILES : argv
warn "data dir: #{MODEL_DIR}"
files.each { |f| download(f) }
