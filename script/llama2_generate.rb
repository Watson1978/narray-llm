#!/usr/bin/env ruby
# frozen_string_literal: true

# Greedy (argmax) generation from a llama2.c checkpoint.
#
#   ruby script/llama2_generate.rb                      # stories260K, 200 tokens
#   GPU=1 ruby script/llama2_generate.rb
#   ruby script/llama2_generate.rb stories110M --length 64
#   ruby script/llama2_generate.rb --tokens 1,403       # token ids, not text
#   CACHE=0 ruby script/llama2_generate.rb              # recompute every step
#   ruby script/llama2_generate.rb --no-bench
#
# There is no BPE encoder (docs/plans/PLAN-llama2.md), so the prompt is token ids. The
# default is BOS alone, which is how run.c starts with an empty prompt.

require 'optparse'

require_relative '../lib/narray_llm'
require_relative 'clock_window'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
# tok512.bin is the vocab-512 tokenizer trained with stories260K; everything
# else uses llama2.c's 32000-entry one.
TOKENIZERS = { 'stories260K' => 'tok512.bin' }.freeze
DEFAULT_TOKENIZER = 'tokenizer.bin'

options = { model: 'stories260K', length: 200, tokens: nil,
            repeat: Integer(ENV.fetch('REPEAT', 3)),
            cache: ENV.fetch('CACHE', '1') !~ /\A(0|off|false)\z/i }
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = "Usage: ruby script/llama2_generate.rb [options] [model]"
  o.on('--length N', Integer, '生成するトークン数') { |v| options[:length] = v }
  o.on('--tokens a,b,c', Array, 'プロンプトのトークン id') do |v|
    options[:tokens] = v.map { |id| Integer(id.strip) }
  end
  o.on('--rounds N', Integer, 'best-of-N の N (既定 3、環境変数 REPEAT でも指定できる)') { |v| options[:repeat] = v }
  o.on('--no-bench', '速度を測らない') { options[:no_bench] = true }
  o.on('--no-eot', 'EOT が出ても止めない') { options[:no_eot] = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end
# What parse! leaves behind is the model, given with or without the suffix.
options[:model] = argv.first.sub(/\.bin\z/, '') unless argv.empty?

checkpoint = File.join(DATA_DIR, "#{options[:model]}.bin")
tokenizer_path = File.join(DATA_DIR, TOKENIZERS.fetch(options[:model], DEFAULT_TOKENIZER))
[checkpoint, tokenizer_path].each do |path|
  next if File.exist?(path)

  warn "#{path} not found. Run `ruby script/download_llama2.rb` first."
  exit 1
end

# runq.c's int8 checkpoints carry their own magic, so the file says which
# model to build rather than a flag having to agree with it.
quantized = File.binread(checkpoint, 4).unpack1('L<') ==
            NArrayLLM::Llama2::QuantizedCheckpoint::MAGIC
model = (quantized ? NArrayLLM::Llama2::QuantizedModel : NArrayLLM::Llama2::Model).load(checkpoint)
tokenizer = NArrayLLM::Llama2::Tokenizer.load(tokenizer_path, vocab_size: model.config.vocab_size)
generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
config = model.config
prompt = options[:tokens] || [NArrayLLM::Llama2::Tokenizer::BOS_TOKEN]

puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:   #{options[:model]} dim=#{config.dim} hidden=#{config.hidden_dim} " \
     "L=#{config.num_layers} NH=#{config.num_heads} NKV=#{config.num_kv_heads} " \
     "V=#{config.vocab_size} maxT=#{config.max_seq_len} " \
     "#{config.grouped_query? ? "GQA(kv_mul=#{config.kv_mul})" : 'MHA'}"
puts "prompt:  #{prompt.inspect}"
puts "生成長:  #{options[:length]} (貪欲法、KV キャッシュ #{options[:cache] ? '有り' : '無し'})"
cache_bytes = options[:cache] ? NArrayLLM::KVCache.bytes_for(
  num_layers: config.num_layers, max_seq_len: config.max_seq_len, channels: config.kv_dim
) : 0
mib = ->(bytes) { bytes / 1024.0 / 1024.0 }
puts format('%s 見積もり: 重み %.1f MiB + KV キャッシュ %.1f MiB = %.1f MiB (別途、一時配列)',
            NArrayLLM.gpu? ? 'VRAM' : 'メモリ', mib.call(model.parameter_bytes),
            mib.call(cache_bytes), mib.call(model.parameter_bytes + cache_bytes))
puts

if prompt.size + options[:length] > config.max_seq_len
  warn "エラー: プロンプト #{prompt.size} + 生成 #{options[:length]} が maxT #{config.max_seq_len} を超えています"
  exit 1
end

puts '== 生成結果 =='
tokens = generator.generate(prompt, max_new_tokens: options[:length], cache: options[:cache], stop_at_eot: !options[:no_eot])
print tokenizer.render(tokens)
puts
puts

generated = tokens.size - prompt.size
puts "生成トークン数: #{generated}#{generated < options[:length] ? ' (BOS で停止)' : ''}"
puts "ids: #{tokens[prompt.size..].inspect}"
puts
exit 0 if options[:no_bench]

generator.generate(prompt, max_new_tokens: 2, cache: options[:cache], stop_at_eot: !options[:no_eot]) # warm up
windows = []
times = Array.new(options[:repeat]) do
  from = ClockWindow.stamp
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  generator.generate(prompt, max_new_tokens: options[:length], cache: options[:cache], stop_at_eot: !options[:no_eot])
  NArrayLLM::Profiler::NULL.synchronize
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  windows << [from, ClockWindow.stamp]
  elapsed
end
best = times.min
puts "== 速度 (best-of-#{options[:repeat]}, 区間同期なし) =="
puts format('%.3f s / %d tokens = %.2f tokens/sec  (cache %s)',
            best, generated, generated / best, options[:cache] ? 'ON' : 'OFF')
puts format('各回: %s', times.map { |t| format('%.3f', t) }.join(', '))
puts ClockWindow.report(windows[times.index(best)]) if ClockWindow.enabled?
