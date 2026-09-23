#!/usr/bin/env ruby
# frozen_string_literal: true

# Writes a small Mamba checkpoint in mamba.c's version 1 format.
#
# There is no published Mamba smaller than 130m, so there is nothing playing
# the part stories260K plays for Llama 2. This makes one, so the acceptance
# tests can run mamba.c as the reference without loading 500 MB.
#
#   ruby script/mamba_tiny.rb                       # data/mamba_tiny.bin
#   ruby script/mamba_tiny.rb --layers 4 out.bin
#
# The weights come from a fixed seed, so the file is reproducible: the same
# command gives the same bytes, and mamba.c's output for it is a stable
# reference. A is written negative, as -exp(A_log) always is.

require 'fileutils'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }

MAGIC = 0x4d616d62
VERSION = 1
HEADER_BYTES = 256
SEED = 20_260_919

options = { layers: 2, vocab: 64, dim: 32, d_inner: 64, dt_rank: 2, d_state: 4, d_conv: 4 }
out_path = nil
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/mamba_tiny.rb [options] [out]'
  o.on('--layers N', Integer, '層の数') { |v| options[:layers] = v }
  o.on('--vocab N', Integer, '語彙の大きさ') { |v| options[:vocab] = v }
  o.on('--dim N', Integer, 'd_model') { |v| options[:dim] = v }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end
out_path = argv.first unless argv.empty?
out_path ||= File.join(DATA_DIR, 'mamba_tiny.bin')

rng = Random.new(SEED)
# Small enough that 24 sequential state updates stay in range, and the logits
# spread enough for greedy decoding to pick different tokens.
sample = ->(n, scale) { Array.new(n) { (rng.rand - 0.5) * 2.0 * scale } }

l = options[:layers]
d = options[:dim]
di = options[:d_inner]
rounded_vocab = options[:vocab] + (options[:vocab] % 8).then { |r| r.zero? ? 0 : 8 - r }
x_proj_out = options[:dt_rank] + 2 * options[:d_state]

FileUtils.mkdir_p(File.dirname(out_path))
File.open(out_path, 'wb') do |io|
  io.write([MAGIC, VERSION].pack('L<l<'))
  io.write([l, options[:vocab], d, di, options[:dt_rank],
            options[:d_state], options[:d_conv], 1].pack('l<8'))
  io.write("\0" * (HEADER_BYTES - io.pos))

  write = ->(values) { io.write(values.pack('e*')) }

  write.call(sample.call(rounded_vocab * d, 0.5))          # embedding
  write.call(sample.call(l * 2 * di * d, 0.2))             # in_proj
  write.call(sample.call(l * di * options[:d_conv], 0.5))  # conv1d_weight
  write.call(sample.call(l * di, 0.1))                     # conv1d_bias
  write.call(sample.call(l * x_proj_out * di, 0.2))        # x_proj
  write.call(sample.call(l * di * options[:dt_rank], 0.2)) # dt_proj_weight
  write.call(sample.call(l * di, 0.1))                     # dt_proj_bias
  # A is -exp(A_log), so every entry is negative and the state decays.
  write.call(Array.new(l * di * options[:d_state]) { -Math.exp(rng.rand * 1.5) })
  write.call(sample.call(l * di, 0.5))                     # D
  write.call(sample.call(l * d * di, 0.2))                 # out_proj
  write.call(Array.new(l * d) { 1.0 + (rng.rand - 0.5) * 0.2 })  # norm
  write.call(Array.new(d) { 1.0 + (rng.rand - 0.5) * 0.2 })      # final_norm
end

warn "#{out_path}: #{File.size(out_path)} bytes"
warn "  layers=#{l} vocab=#{options[:vocab]} (rounded #{rounded_vocab}) dim=#{d} " \
     "d_inner=#{di} dt_rank=#{options[:dt_rank]} d_state=#{options[:d_state]} d_conv=#{options[:d_conv]}"
