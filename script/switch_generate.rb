#!/usr/bin/env ruby
# frozen_string_literal: true

# Greedy generation from google/switch-base-8.
#
#   ruby script/switch_generate.rb --tokens 8774,15,3,9,794,1
#   GPU=1 ruby script/switch_generate.rb --router dense
#
# The prompt is token ids, not text: there is no SentencePiece reader here yet
# (docs/plans/PLAN-switch.md). These checkpoints were trained on masked span filling, so
# a prompt usually carries sentinel ids such as 32099.

require 'optparse'

require_relative '../lib/narray_llm'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
MODEL = File.join(DATA_DIR, 'switch-base-8.safetensors')

options = { tokens: [8774, 15, 3, 9, 794, 1], length: 24, router: :dispatch, capacity: nil,
            repeat: Integer(ENV.fetch('REPEAT', 3)) }
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/switch_generate.rb [options]'
  o.on('--tokens a,b,c', Array, 'プロンプトのトークン id') do |v|
    options[:tokens] = v.map { |id| Integer(id.strip) }
  end
  o.on('--prompt-length N', Integer, 'N 個のプロンプトを組み立てる') do |v|
    options[:tokens] = (100...(100 + v - 1)).to_a + [1]
  end
  o.on('--length N', Integer, '生成するトークン数') { |v| options[:length] = v }
  o.on('--router NAME', NArrayLLM::Switch::Stack::ROUTERS.map(&:to_s),
       "expert の選び方 (#{NArrayLLM::Switch::Stack::ROUTERS.join(' / ')})") do |v|
    options[:router] = v.to_sym
  end
  o.on('--capacity N', Integer, 'expert 1 つが受ける上限') { |v| options[:capacity] = v }
  o.on('--encode-only', 'encoder までで止める') { options[:encode_only] = true }
  o.on('--rounds N', Integer, 'best-of-N の N (既定 3、環境変数 REPEAT でも指定できる)') { |v| options[:repeat] = v }
  o.on('--no-bench', '速度を測らない') { options[:no_bench] = true }
  o.on('--no-eos', 'EOS が出ても止めない') { options[:no_eos] = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(ARGV.dup)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

unless File.exist?(MODEL)
  warn "#{MODEL} not found. Run `ruby script/download_switch.rb` and " \
       '`python/.venv/bin/python python/export_switch.py data/switch-base-8 ' \
       "#{MODEL}` first."
  exit 1
end

model = NArrayLLM::Switch::Model.load(MODEL, router: options[:router],
                                             expert_capacity: options[:capacity])
config = model.config
puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:   switch-base-8 d_model=#{config.d_model} experts=#{config.num_experts} " \
     "layers=#{config.num_layers}+#{config.num_decoder_layers} V=#{config.vocab_size}"
puts "router:  #{options[:router]}#{options[:capacity] ? " (capacity #{options[:capacity]})" : ' (capacity なし、参照と同じ)'}"
puts "prompt:  #{options[:tokens].inspect}"
puts

run = lambda do
  if options[:encode_only]
    model.encode(options[:tokens])
  else
    model.generate(options[:tokens], max_new_tokens: options[:length], stop_at_eos: !options[:no_eos])
  end
end

result = run.call
if options[:encode_only]
  puts "encoder 出力: #{result.shape.inspect}"
  produced = options[:tokens].size
else
  puts "ids: #{result.inspect}"
  produced = result.size - 1
  puts "生成トークン数: #{produced}#{result.last == config.eos_token_id ? ' (eos で停止)' : ''}"
end
exit 0 if options[:no_bench]

run.call # warm up
times = Array.new(options[:repeat]) do
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  run.call
  NArrayLLM::Profiler::NULL.synchronize
  Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
end
best = times.min
puts format('%.3f s / %d %s = %.2f %s/sec  (best of %d)',
            best, produced, options[:encode_only] ? 'tokens' : 'tokens',
            produced / best, options[:encode_only] ? 'tokens' : 'tokens',
            options[:repeat])
