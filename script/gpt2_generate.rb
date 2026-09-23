#!/usr/bin/env ruby
# frozen_string_literal: true

# Greedy (argmax) generation from the GPT-2 124M weights.
#
#   ruby script/gpt2_generate.rb                          # 64 tokens from a single EOT
#   GPU=1 ruby script/gpt2_generate.rb
#   ruby script/gpt2_generate.rb --length 128
#   ruby script/gpt2_generate.rb --tokens 15496,11,995    # token ids, not text
#   ruby script/gpt2_generate.rb --batch 8                # 8 sequences from one cache
#   ruby script/gpt2_generate.rb --top-k 50 --top-p 0.9 --seed 42   # sample instead of argmax
#   CACHE=0 ruby script/gpt2_generate.rb                  # recompute every step (stage 2 path)
#   DETAIL=1 ruby script/gpt2_generate.rb                 # per-token breakdown, one section per block
#   DETAIL=op ruby script/gpt2_generate.rb                # per-operation breakdown (see the caveat below)
#
# The default prompt is one EOT token, which is how llm.c kicks off
# unconditional generation (train_gpt2.c:1127-1129) and needs no BPE encoder.

require 'optparse'

require_relative '../lib/narray_llm'
require_relative 'clock_window'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
DEFAULT_LENGTH = 64

def parse_options(argv)
  options = { length: DEFAULT_LENGTH, tokens: nil, batch: nil,
              temperature: nil, top_k: nil, top_p: nil, seed: nil,
              repeat: Integer(ENV.fetch('REPEAT', 3)),
              cache: ENV.fetch('CACHE', '1') !~ /\A(0|off|false)\z/i,
              detail: ENV['DETAIL'].to_s.empty? ? nil : ENV['DETAIL'] }
  parser = OptionParser.new do |parser|
    parser.banner = 'Usage: ruby script/gpt2_generate.rb [options]'
    parser.on('--length N', Integer, "生成するトークン数 (既定 #{DEFAULT_LENGTH})") { |v| options[:length] = v }
    parser.on('--batch N', Integer, '1 つのキャッシュから走らせる系列の数') { |v| options[:batch] = v }
    parser.on('--temperature F', Float, 'logits を割る温度') { |v| options[:temperature] = v }
    parser.on('--top-k N', Integer, '上位 k 個だけ残す') { |v| options[:top_k] = v }
    parser.on('--top-p F', Float, '累積確率 p までを残す') { |v| options[:top_p] = v }
    parser.on('--seed N', Integer, 'サンプリングの乱数の種') { |v| options[:seed] = v }
    parser.on('--tokens a,b,c', Array, 'プロンプトのトークン id (文字列ではない)') do |v|
      options[:tokens] = v.map { |id| Integer(id.strip) }
    end
    parser.on('--rounds N', Integer, 'best-of-N の N (既定 3、環境変数 REPEAT でも指定できる)') { |v| options[:repeat] = v }
    parser.on('--no-bench', '速度を測らない') { options[:no_bench] = true }
    parser.on('--no-eot', 'EOT が出ても止めない') { options[:no_eot] = true }
    parser.on('-h', '--help', 'この説明を出す') do
      puts parser
      exit 0
    end
  end
  parser.parse!(argv)
  options
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

# Tokens are byte-level BPE pieces, so a character can straddle two of them.
# Print only the bytes that already form complete UTF-8 and carry the rest over.
def split_complete_utf8(buffer)
  0.upto(3) do |drop|
    size = buffer.bytesize - drop
    break if size.negative?

    head = buffer.byteslice(0, size).dup.force_encoding(Encoding::UTF_8)
    return [head, buffer.byteslice(size, drop)] if head.valid_encoding?
  end
  [buffer.dup.force_encoding(Encoding::UTF_8).scrub, +'']
end

def print_timing(profiler, tokens)
  puts format('%-12s %10s %10s %10s %9s %8s',
              'section', 'seconds', 'ms/token', 'calls', 'calls/tok', 'share')
  puts '-' * 66
  total_calls = 0
  profiler.rows.each do |name, seconds, calls, share|
    total_calls += calls
    puts format('%-12s %10.4f %10.3f %10d %9.1f %7.1f%%', name, seconds, seconds / tokens * 1000,
                calls, calls.to_f / tokens, share * 100)
  end
  puts '-' * 66
  puts format('%-12s %10.4f %10.3f %10d %9.1f', 'total', profiler.total,
              profiler.total / tokens * 1000, total_calls, total_calls.to_f / tokens)
  puts '区間呼び出し数はカーネル起動数の下限の目安 (1 区間に 1 個以上のカーネルが入る)'
end

options = parse_options(ARGV.dup)
backend = "#{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"

model = NArrayLLM::GPT2::Model.load(File.join(DATA_DIR, 'gpt2_124M.bin'))
tokenizer = NArrayLLM::GPT2::Tokenizer.load(File.join(DATA_DIR, 'gpt2_tokenizer.bin'))
generator = NArrayLLM::Generator.new(model, tokenizer: tokenizer)

prompt = options[:tokens] || [tokenizer.eot_token]
config = model.config

puts "backend: #{backend}"
puts "model:   L=#{config.num_layers} NH=#{config.num_heads} C=#{config.channels} " \
     "V=#{config.vocab_size} maxT=#{config.max_seq_len}"
puts "prompt:  #{prompt.inspect} #{tokenizer.decode(prompt).inspect}"
puts "生成長:  #{options[:length]} (貪欲法、KV キャッシュ #{options[:cache] ? '有り' : '無し'})"
puts "バッチ:  #{options[:batch]}" if options[:batch]

sampler =
  if options[:temperature] || options[:top_k] || options[:top_p]
    NArrayLLM::Sampler.new(
      temperature: options[:temperature] || NArrayLLM::Sampler::DEFAULT_TEMPERATURE,
      top_k: options[:top_k], top_p: options[:top_p], seed: options[:seed]
    )
  end
if sampler
  puts format('抽出:    温度 %.3g, top_k %s, top_p %s, seed %s',
              sampler.temperature, sampler.top_k || '-', sampler.top_p || '-',
              sampler.seed || 'なし')
end

weight_bytes = model.parameter_bytes
cache_bytes = options[:cache] ? NArrayLLM::KVCache.bytes_for(
  num_layers: config.num_layers, max_seq_len: config.max_seq_len, channels: config.channels,
  batch_size: options[:batch] || 1
) : 0
mib = ->(bytes) { bytes / 1024.0 / 1024.0 }
puts format('%s 見積もり: 重み %.1f MiB + KV キャッシュ %.1f MiB = %.1f MiB (別途、一時配列)',
            NArrayLLM.gpu? ? 'VRAM' : 'メモリ',
            mib.call(weight_bytes), mib.call(cache_bytes), mib.call(weight_bytes + cache_bytes))
puts

$stdout.flush # so the message below is not reordered ahead of the header
if prompt.size + options[:length] > config.max_seq_len
  warn "エラー: プロンプト #{prompt.size} + 生成 #{options[:length]} が maxT #{config.max_seq_len} を超えています"
  exit 1
end

if options[:batch]
  if sampler
    warn 'エラー: バッチ生成とサンプリングはまだ併用できません'
    exit 1
  end
  unless options[:cache]
    warn 'エラー: バッチ生成は KV キャッシュ前提です (CACHE=0 とは併用できません)'
    exit 1
  end

  prompts = Array.new(options[:batch]) { prompt.dup }
  run = lambda do
    generator.generate_batch(prompts.map(&:dup), max_new_tokens: options[:length],
                             stop_at_eot: !options[:no_eot])
  end
  sequences = run.call
  generated = sequences.first.size - prompt.size
  puts '== 生成結果 (先頭の 1 本) =='
  puts tokenizer.decode(sequences.first)
  puts
  puts "生成トークン数: #{generated} x #{options[:batch]} 本"
  exit 0 if options[:no_bench]

  generator.generate_batch(prompts.map(&:dup), max_new_tokens: 2, stop_at_eot: !options[:no_eot])
  times = Array.new(options[:repeat]) do
    started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    run.call
    NArrayLLM::Profiler::NULL.synchronize
    Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  end
  best = times.min
  puts "== 速度 (best-of-#{options[:repeat]}, 区間同期なし) =="
  puts format('%.3f s / %d tokens x %d = %.2f tokens/sec 合計  (%.2f / 本, %s)',
              best, generated, options[:batch], generated * options[:batch] / best,
              generated / best, backend)
  puts format('各回: %s', times.map { |t| format('%.3f', t) }.join(', '))
  puts format('1 ステップあたり %.2f ms', best / generated * 1000)
  exit 0
end

puts '== 生成結果 =='
print tokenizer.decode(prompt)
$stdout.flush
pending = +''
tokens = generator.generate(prompt, max_new_tokens: options[:length], cache: options[:cache], stop_at_eot: !options[:no_eot], sampler: sampler) do |token|
  pending << tokenizer[token]
  printable, pending = split_complete_utf8(pending)
  print printable
  $stdout.flush
end
print pending.force_encoding(Encoding::UTF_8).scrub unless pending.empty?
puts
puts

generated = tokens.size - prompt.size
puts "生成トークン数: #{generated}#{generated < options[:length] ? ' (EOT で停止)' : ''}"
puts "ids: #{tokens[prompt.size..].inspect}"
puts
exit 0 if options[:no_bench]

# tokens/sec: best-of-N with no per-section synchronization (AGENTS.md).
#
# The timed runs never stop at EOT. A sampler draws a different sequence each
# time, so one run can end early and take best-of-N with it: the rate would be
# the requested length over the time of the shortest run.
generator.generate(prompt, max_new_tokens: 2, cache: options[:cache], stop_at_eot: false, sampler: sampler) # warm up cuBLAS and the allocator
windows = []
times = Array.new(options[:repeat]) do
  from = ClockWindow.stamp
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  run = generator.generate(prompt, max_new_tokens: options[:length], cache: options[:cache], stop_at_eot: false, sampler: sampler)
  NArrayLLM::Profiler::NULL.synchronize
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
  windows << [from, ClockWindow.stamp]
  raise "timed run made #{run.size - prompt.size} of #{options[:length]} tokens" unless
    run.size - prompt.size == options[:length]

  elapsed
end
generated = options[:length]
best = times.min
puts "== 速度 (best-of-#{options[:repeat]}, 区間同期なし) =="
puts format('%.3f s / %d tokens = %.2f tokens/sec  (%s, cache %s)',
            best, generated, generated / best, backend, options[:cache] ? 'ON' : 'OFF')
# 全試行を出す。他の負荷が乗った回は遅くなるだけなので、ばらつきを見れば
# 最小値が汚れていないかを後から判断できる。
puts format('各回: %s', times.map { |t| format('%.3f', t) }.join(', '))
puts format('1 トークンあたり %.1f ms', best / generated * 1000)
puts ClockWindow.report(windows[times.index(best)]) if ClockWindow.enabled?
puts

if options[:detail]
  coarse = options[:detail] != 'op'
  profiler = coarse ? NArrayLLM::Profiler.coarse : NArrayLLM::Profiler.new(enabled: true)
  generator.generate(prompt, max_new_tokens: options[:length], prof: profiler, cache: options[:cache], stop_at_eot: !options[:no_eot])
  puts format('== 1 トークンあたりの時間内訳 (%s) ==', coarse ? 'DETAIL=1, ブロック単位' : 'DETAIL=op, 演算単位')
  puts '区間の境界ごとに同期するので合計は上の総時間より大きい。同期の費用は区間の中身に'
  puts '依らないため、呼び出し回数の多い区間ほど余計に課金される。演算単位の表を区間どうしの'
  puts '順位として読まないこと。'
  print_timing(profiler, generated)
end
