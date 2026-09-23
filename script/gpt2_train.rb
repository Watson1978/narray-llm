#!/usr/bin/env ruby
# frozen_string_literal: true

# Ten AdamW steps against llm.c's debug state, which is what test_gpt2.c does.
#
#   ruby script/gpt2_train.rb                  # 10 steps, checks the loss sequence
#   GPU=1 ruby script/gpt2_train.rb --steps 4
#   GPU=1 ruby script/gpt2_train.rb --no-check # skip the sequence check
#
# The batch is whatever the debug state holds (B=4, T=64). Steps are not
# repeatable -- each one moves the weights -- so the time per step is the total
# divided by the count, with the first step dropped: it pays for cuBLAS
# handles, the allocator and the optimiser's first allocation.

require 'optparse'

require_relative '../lib/narray_llm'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }

# test_gpt2.c:89-99, at the settings test_gpt2.c:172 passes.
EXPECTED_LOSSES = [
  5.270007133483887, 4.059706687927246, 3.3751230239868164, 2.8007826805114746,
  2.315382242202759, 1.8490285873413086, 1.3946564197540283, 0.9991465210914612,
  0.6240804195404053, 0.37651097774505615
].freeze
LOSS_TOLERANCE = 1e-2

options = { steps: 10, stop_after: :update }
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/gpt2_train.rb [options]'
  o.on('--steps N', Integer, 'AdamW のステップ数') { |v| options[:steps] = v }
  o.on('--no-check', 'llm.c の損失列と照合しない') { options[:no_check] = true }
  # Counting the three phases apart: the difference between these is the split.
  o.on('--stop-after PHASE', %w[forward backward update],
       'どこで止めるか (forward / backward / update)') { |v| options[:stop_after] = v.to_sym }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

checkpoint = NArrayLLM::GPT2::Checkpoint.load(File.join(DATA_DIR, 'gpt2_124M.bin'))
state = NArrayLLM::GPT2::DebugState.load(File.join(DATA_DIR, 'gpt2_124M_debug_state.bin'),
                                         config: checkpoint.config)
model = NArrayLLM::GPT2::Model.new(checkpoint)
optimiser = NArrayLLM::AdamW.new(model)
config = checkpoint.config

puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:   L=#{config.num_layers} NH=#{config.num_heads} C=#{config.channels} " \
     "V=#{config.vocab_size}"
puts "batch:   B=#{state.batch_size} T=#{state.seq_len}"
puts "steps:   #{options[:steps]} (AdamW lr #{NArrayLLM::AdamW::DEFAULTS[:learning_rate]})"
puts "止める:  #{options[:stop_after]}" unless options[:stop_after] == :update
puts

times = []
options[:steps].times do |step|
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  loss, acts = model.forward_train(state.inputs, state.targets)
  unless options[:stop_after] == :forward
    grads = model.backward(acts)
    optimiser.step(grads) unless options[:stop_after] == :backward
  end
  NArrayLLM::Profiler::NULL.synchronize
  times << Process.clock_gettime(Process::CLOCK_MONOTONIC) - started

  # Without the update the weights never move, so only the first loss is the
  # sequence's.
  want = EXPECTED_LOSSES[options[:stop_after] == :update ? step : 0]
  ok = want.nil? || (loss - want).abs < LOSS_TOLERANCE
  unless options[:no_check] || ok
    warn "step #{step}: loss #{loss} but llm.c has #{want}"
    exit 1
  end
  puts format('step %2d  loss %.6f%s  %.3f s', step, loss,
              want ? format('  (参照 %.6f, 差 %.1e)', want, (loss - want).abs) : '', times.last)
end

puts
timed = times.drop(1)
if timed.empty?
  puts '1 ステップだけなので時間は出さない (最初の 1 本は確保と初期化を払う)'
else
  total = timed.sum
  puts format('%d ステップ (先頭を除く) %.3f s = %.5f s/step, %.2f steps/sec',
              timed.size, total, total / timed.size, timed.size / total)
  puts format('各回: %s', timed.map { |t| format('%.3f', t) }.join(', '))
end
puts format('AdamW の状態: %.1f MiB', optimiser.bytes / 1048576.0)
