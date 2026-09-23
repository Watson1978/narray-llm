#!/usr/bin/env ruby
# frozen_string_literal: true

# Runs one forward pass over the llm.c debug state and compares it with the
# reference logits and loss.
#
#   ruby script/gpt2_forward.rb                       # Numo
#   GPU=1 ruby script/gpt2_forward.rb                 # Cumo
#   DETAIL=1 ruby script/gpt2_forward.rb              # breakdown, one section per block
#   DETAIL=op ruby script/gpt2_forward.rb             # per-operation breakdown (see the caveat below)
#   ruby script/gpt2_forward.rb --dump numo.acts      # save every intermediate
#   GPU=1 ruby script/gpt2_forward.rb --compare numo.acts
#
# The debug state carries no reference intermediates (only x, y, logits, loss and
# grads), so the layer-by-layer table compares the two backends against each
# other: dump from one, compare from the other.

require 'optparse'

require_relative '../lib/narray_llm'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }

# test_gpt2.c:127 fails the logits check at diff >= 1e-2f.
LOGITS_TOLERANCE = 1e-2
# test_gpt2.c:141 fails the loss check at >= 1e-2.
LOSS_TOLERANCE = 1e-2
# Two backends running the same fp32 arithmetic should agree an order of
# magnitude more tightly than either agrees with PyTorch.
BACKEND_TOLERANCE = 1e-3
# Intermediates have no reference to check against and their magnitudes span
# 1 to ~3000, so the layer table flags on relative error instead. fp32 rounding
# lands around 1e-6 relative; anything at 1e-4 is a bug, not arithmetic.
ACTIVATION_REL_TOLERANCE = 1e-4

def parse_options(argv)
  options = { repeat: Integer(ENV.fetch('REPEAT', 3)),
              detail: ENV['DETAIL'].to_s.empty? ? nil : ENV['DETAIL'] }
  parser = OptionParser.new do |o|
    o.banner = 'Usage: ruby script/gpt2_forward.rb [options]'
    o.on('--dump PATH', '中間活性を書き出す先') { |v| options[:dump] = v }
    o.on('--compare PATH', '書き出したものと突き合わせる') { |v| options[:compare] = v }
    o.on('--full', '全位置の logits を出す') { options[:full] = true }
    o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
  end
  parser.parse!(argv)
  options
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

def dump_activations(path, activations)
  payload = activations.transform_values { |t| [t.shape, t.to_binary] }
  File.binwrite(path, Marshal.dump(payload))
end

def load_activations(path)
  Marshal.load(File.binread(path)).transform_values do |(shape, binary)|
    XM::SFloat.from_binary(binary, shape)
  end
end

def summary_keys(activations, config)
  keys = ['encoded']
  config.num_layers.times { |l| keys << "L#{l}/residual3" }
  keys + %w[lnf logits]
end

def print_stats_table(activations, keys)
  puts format('%-16s %-16s %12s %12s %12s %8s', 'activation', 'shape', 'mean', 'min', 'max', 'finite')
  puts '-' * 82
  keys.each do |key|
    s = NArrayLLM::Compare.stats(activations.fetch(key), label: key)
    puts format('%-16s %-16s %12.5f %12.5f %12.5f %8s',
                key, s[:shape].inspect, s[:mean], s[:min], s[:max], s[:finite] ? 'yes' : 'NO')
  end
end

def print_diff_table(reference, activations, keys, rel_tolerance)
  puts format('%-16s %12s %12s %11s %11s  %s',
              'activation', 'max|ref|', 'max|d|', 'mean|d|', 'rel', 'first position over rel tol')
  puts '-' * 92
  first_divergence = nil
  keys.each do |key|
    unless reference.key?(key)
      puts format('%-16s %12s', key, '(absent)')
      next
    end
    expected = reference.fetch(key)
    scale = NArrayLLM.scalar(expected.abs.max)
    # Turn the relative budget into the absolute threshold for this tensor, so
    # the reported position is the first element that actually exceeds it.
    d = NArrayLLM::Compare.diff(expected, activations.fetch(key),
                                 tolerance: rel_tolerance * [scale, 1.0].max, label: key)
    first_divergence ||= key unless d.ok?
    puts format('%-16s %12.4e %12.4e %11.4e %11.4e  %s',
                key, d.ref_max_abs, d.max_abs, d.mean_abs, d.rel_max,
                d.ok? ? '-' : d.first_bad_index.inspect)
  end
  first_divergence
end

def print_timing(profiler)
  puts format('%-12s %10s %8s %8s', 'section', 'seconds', 'calls', 'share')
  puts '-' * 42
  profiler.rows.each do |name, seconds, calls, share|
    puts format('%-12s %10.4f %8d %7.1f%%', name, seconds, calls, share * 100)
  end
  puts '-' * 42
  puts format('%-12s %10.4f', 'total', profiler.total)
end

options = parse_options(ARGV.dup)
backend = NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'

checkpoint = NArrayLLM::GPT2::Checkpoint.load(File.join(DATA_DIR, 'gpt2_124M.bin'))
config = checkpoint.config
state = NArrayLLM::GPT2::DebugState.load(File.join(DATA_DIR, 'gpt2_124M_debug_state.bin'), config: config)
model = NArrayLLM::GPT2::Model.new(checkpoint)
checkpoint = nil # the model keeps its own prepared copies; let this one go

puts "backend: #{backend}"
puts "model:   L=#{config.num_layers} NH=#{config.num_heads} C=#{config.channels} " \
     "V=#{config.vocab_size} maxT=#{config.max_seq_len}"
puts "input:   B=#{state.batch_size} T=#{state.seq_len} (gpt2_124M_debug_state.bin)"
puts

trace = {}
logits = model.forward(state.inputs, trace: trace)
loss = model.loss(logits, state.targets)

puts '== 参照値との比較 =='
logits_diff = NArrayLLM::Compare.diff(state.logits, logits,
                                       tolerance: LOGITS_TOLERANCE, label: 'logits')
puts logits_diff
loss_diff = (state.loss - loss).abs
puts format('%-22s ref=%.9f got=%.9f |d|=%.3e tol=%.1e %s',
            'loss', state.loss, loss, loss_diff, LOSS_TOLERANCE,
            loss_diff <= LOSS_TOLERANCE ? 'OK' : 'NG')
puts

keys = options[:full] || options[:compare] ? trace.keys : summary_keys(trace, config)

if options[:compare]
  reference = load_activations(options[:compare])
  puts "== 層ごとの中間活性: #{options[:compare]} との比較 (相対 tol=#{ACTIVATION_REL_TOLERANCE}) =="
  first = print_diff_table(reference, trace, keys, ACTIVATION_REL_TOLERANCE)
  puts
  if first
    puts "最初に乖離した箇所: #{first}"
  else
    puts '全ての中間活性が相対閾値内 (fp32 の丸めのみ)。'
  end
  final = NArrayLLM::Compare.diff(reference.fetch('logits'), trace.fetch('logits'),
                                   tolerance: BACKEND_TOLERANCE, label: 'logits (backend)')
  puts final
else
  puts '== 層ごとの中間活性 =='
  print_stats_table(trace, keys)
end
puts

if options[:dump]
  dump_activations(options[:dump], trace)
  puts "dumped #{trace.size} activations to #{options[:dump]} (#{File.size(options[:dump])} bytes)"
  puts
end

# Total time: no per-section synchronization, best-of-N (AGENTS.md).
model.forward(state.inputs) # warm up caches / cuBLAS handles
times = Array.new(options[:repeat]) do
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  model.forward(state.inputs)
  NArrayLLM::Profiler::NULL.synchronize
  Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
end
puts "== 総時間 (best-of-#{options[:repeat]}, 区間同期なし) =="
puts format('%.4f s  (%s, B=%d T=%d)', times.min, backend, state.batch_size, state.seq_len)
# 全試行を出す。他の負荷が乗った回は遅くなるだけなので、ばらつきを見れば
# 最小値が汚れていないかを後から判断できる。
puts format('各回: %s', times.map { |t| format('%.4f', t) }.join(', '))
puts

if options[:detail]
  coarse = options[:detail] != 'op'
  profiler = coarse ? NArrayLLM::Profiler.coarse : NArrayLLM::Profiler.new(enabled: true)
  model.forward(state.inputs, prof: profiler)
  puts format('== 時間内訳 (%s) ==', coarse ? 'DETAIL=1, ブロック単位' : 'DETAIL=op, 演算単位')
  puts '区間の境界ごとに同期するので合計は上の総時間より大きい。同期の費用は区間の中身に'
  puts '依らないため、呼び出し回数の多い区間ほど余計に課金される。演算単位の表を区間どうしの'
  puts '順位として読まないこと。'
  print_timing(profiler)
end
