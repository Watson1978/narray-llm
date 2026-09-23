#!/usr/bin/env ruby
# frozen_string_literal: true

# Runs the Llama 2 forward pass against the reference activations and reports,
# stage by stage, where the two first disagree.
#
#   ruby script/llama2_forward.rb                     # stories260K
#   GPU=1 ruby script/llama2_forward.rb
#   ruby script/llama2_forward.rb stories110M
#   ruby script/llama2_forward.rb --layers            # every layer, not a summary
#
# The reference comes from script/llama2_dump.rb.

require_relative '../lib/narray_llm'
require 'optparse'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
PER_LAYER = %w[rms_att q_pre_rope k_pre_rope v q k attn_out attproj res_att
               rms_ffn w1h w3h swiglu ffn_out res_ffn].freeze

model_name = 'stories260K'
show_layers = false
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/llama2_forward.rb [options] [model]'
  o.on('--layers', '層ごとの中間活性も出す') { show_layers = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end
model_name = argv.first.sub(/\.bin\z/, '') unless argv.empty?

checkpoint = File.join(DATA_DIR, "#{model_name}.bin")
reference = File.join(DATA_DIR, "#{model_name}_debug_state.bin")
[checkpoint, reference].each do |path|
  next if File.exist?(path)

  warn "#{path} not found. Run:"
  warn '  ruby script/download_llama2.rb'
  warn '  ruby script/llama2_dump.rb'
  exit 1
end

model = NArrayLLM::Llama2::Model.load(checkpoint)
state = NArrayLLM::Llama2::DebugState.load(reference)
config = model.config

puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}"
puts "model:   #{model_name} dim=#{config.dim} hidden=#{config.hidden_dim} " \
     "L=#{config.num_layers} NH=#{config.num_heads} NKV=#{config.num_kv_heads} " \
     "V=#{config.vocab_size} maxT=#{config.max_seq_len}"
puts "         head_size=#{config.head_size} kv_mul=#{config.kv_mul} " \
     "#{config.grouped_query? ? 'GQA' : 'MHA'}"
puts "tokens:  #{state.tokens.inspect}"
puts

trace = {}
logits = model.forward(state.tokens, trace: trace)

# Worst over every position, so one bad position cannot hide behind the rest.
worst = Hash.new { |h, k| h[k] = 0.0 }
state.steps.times do |pos|
  worst['embed'] = [worst['embed'],
                    NArrayLLM::Compare.diff(state['embed', pos], trace['embed'][pos, true]).max_abs].max
  config.num_layers.times do |layer|
    PER_LAYER.each do |name|
      d = NArrayLLM::Compare.diff(state[name, pos, layer], trace["L#{layer}/#{name}"][pos, true])
      worst[name] = [worst[name], d.max_abs].max
      worst["L#{layer}/#{name}"] = [worst["L#{layer}/#{name}"], d.max_abs].max
    end
    expected = state['att', pos, layer].reshape(config.num_heads, pos + 1)
    d = NArrayLLM::Compare.diff(expected, trace["L#{layer}/att"][true, pos, 0..pos])
    worst['att'] = [worst['att'], d.max_abs].max
    worst["L#{layer}/att"] = [worst["L#{layer}/att"], d.max_abs].max
  end
  worst['rms_final'] = [worst['rms_final'],
                        NArrayLLM::Compare.diff(state['rms_final', pos],
                                                trace['rms_final'][pos, true]).max_abs].max
  worst['logits'] = [worst['logits'],
                     NArrayLLM::Compare.diff(state.logits(pos), logits[pos, true]).max_abs].max
end

order = ['embed'] + PER_LAYER + %w[att rms_final logits]
puts '== 段ごとの最大 max|d| (全位置・全層) =='
order.each { |name| puts format('  %-12s %.3e', name, worst[name]) }

if show_layers
  puts
  puts '== 層ごと =='
  config.num_layers.times do |layer|
    row = (PER_LAYER + ['att']).map { |n| format('%s=%.1e', n, worst["L#{layer}/#{n}"]) }
    puts "  L#{layer}: #{row.join(' ')}"
  end
end

puts
greedy = (0...(state.steps - 1)).map { |pos| NArrayLLM.scalar(logits[pos, true].max_index) }
expected = state.tokens[1..]
puts '== 貪欲法のトークン列 =='
puts "  参照: #{expected.inspect}"
puts "  実装: #{greedy.inspect}"
puts(greedy == expected ? '  完全一致' : '  不一致')
exit(greedy == expected ? 0 : 1)
