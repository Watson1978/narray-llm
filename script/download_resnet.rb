#!/usr/bin/env ruby
# frozen_string_literal: true

# Downloads what the ResNet-18 implementation needs.
#
# There is no C reference for this one, so nothing is vendored: the reference
# is transformers. The model publishes a safetensors, which
# lib/narray_llm/safetensors.rb reads directly. The sample images and the
# reference values come from python/resnet_dump.py, not from here.
#
#   ruby script/download_resnet.rb            # config, preprocessing and weights
#   ruby script/download_resnet.rb config.json
#   FORCE=1 ruby script/download_resnet.rb    # re-download even if present

require 'open3'
require 'fileutils'
require 'optparse'

HF = 'https://huggingface.co/microsoft/resnet-18/resolve/main/'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
MODEL_DIR = File.join(DATA_DIR, 'resnet-18')

FILES = %w[config.json preprocessor_config.json model.safetensors].freeze

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
  o.banner = 'Usage: ruby script/download_resnet.rb [files...]'
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
