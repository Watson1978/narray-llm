#!/usr/bin/env ruby
# frozen_string_literal: true

# Greedy (argmax) generation from a mamba.c checkpoint.
#
#   ruby script/mamba_generate.rb                       # mamba-130m, 200 tokens
#   GPU=1 ruby script/mamba_generate.rb
#   ruby script/mamba_generate.rb mamba_tiny --length 8
#   ruby script/mamba_generate.rb --tokens 9038,13      # token ids, not text
#   ruby script/mamba_generate.rb --no-bench
#
# There is no BPE encoder (docs/plans/PLAN-mamba.md), so the prompt is token ids. The
# default is <|endoftext|> alone, which is what mamba.c prepends.
#
# There is no cache switch: Mamba has no recompute path to compare against.
# Its two recurrences are the only way it advances.

require 'optparse'

require_relative '../lib/narray_llm'
require_relative 'clock_window'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
TOKENIZER = 'mamba_tokenizer.bin'

options = { model: 'mamba-130m', length: 200, tokens: nil,
            repeat: Integer(ENV.fetch('REPEAT', 3)) }
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = "Usage: ruby script/mamba_generate.rb [options] [model]"
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
tokenizer_path = File.join(DATA_DIR, TOKENIZER)
[checkpoint, tokenizer_path].each do |path|
  next if File.exist?(path)

  warn "#{path} not found. Run `ruby script/download_mamba.rb` first."
  exit 1
end

model = NArrayLLM::Mamba::Model.load(checkpoint)
config = model.config
# mamba_tiny's vocabulary is 64 entries of random weights, so the real table
# would index past it. Its output is not text either way.
tokenizer = NArrayLLM::Mamba::Tokenizer.load(tokenizer_path) if config.vocab_size > 1024
generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)
prompt = options[:tokens] || [NArrayLLM::Mamba::Tokenizer::BOS_TOKEN]

puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:   #{options[:model]} dim=#{config.dim} d_inner=#{config.d_inner} " \
     "L=#{config.num_layers} dt_rank=#{config.dt_rank} d_state=#{config.d_state} " \
     "d_conv=#{config.d_conv} V=#{config.vocab_size} (#{config.rounded_vocab_size})"
puts "prompt:  #{prompt.inspect}"
puts "生成長:  #{options[:length]} (貪欲法)"
state_bytes = NArrayLLM::Mamba::State.bytes_for(
  num_layers: config.num_layers, d_inner: config.d_inner,
  d_conv: config.d_conv, d_state: config.d_state
)
mib = ->(bytes) { bytes / 1024.0 / 1024.0 }
puts format('%s 見積もり: 重み %.1f MiB + 状態 %.1f MiB = %.1f MiB (別途、一時配列)',
            NArrayLLM.gpu? ? 'VRAM' : 'メモリ', mib.call(model.parameter_bytes),
            mib.call(state_bytes), mib.call(model.parameter_bytes + state_bytes))
puts

puts '== 生成結果 =='
tokens = generator.generate(prompt, max_new_tokens: options[:length],
                            stop_at_eot: !options[:no_eot])
print tokenizer.render(tokens) if tokenizer
puts
puts

generated = tokens.size - prompt.size
puts "生成トークン数: #{generated}#{generated < options[:length] ? ' (EOT で停止)' : ''}"
puts "ids: #{tokens[prompt.size..].inspect}"
puts
exit 0 if options[:no_bench]

generator.generate(prompt, max_new_tokens: 2, stop_at_eot: !options[:no_eot]) # warm up
windows = []
times = Array.new(options[:repeat]) do
  from = ClockWindow.stamp
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  generator.generate(prompt, max_new_tokens: options[:length], stop_at_eot: !options[:no_eot])
  NArrayLLM::Profiler::NULL.synchronize
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  windows << [from, ClockWindow.stamp]
  elapsed
end

best = times.min
puts format('%.3f s / %d tokens = %.2f tokens/sec  (best of %d)',
            best, generated, generated / best, options[:repeat])
puts ClockWindow.report(windows[times.index(best)]) if ClockWindow.enabled?
