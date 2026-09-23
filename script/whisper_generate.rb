#!/usr/bin/env ruby
# frozen_string_literal: true

# Greedy generation from openai/whisper-tiny.
#
#   ruby script/whisper_generate.rb                    # dump のメルを使う
#   GPU=1 ruby script/whisper_generate.rb --spelling unfold
#
# There is no audio front end here yet (docs/plans/PLAN-whisper.md): the mel comes from
# the reference dump python/whisper_dump.py writes. The prompt is the forced
# one Whisper builds for English transcription with no timestamps.

require 'optparse'

require_relative '../lib/narray_llm'
require 'json'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
MODEL = File.join(DATA_DIR, 'whisper-tiny', 'model.safetensors')
DUMP = File.join(DATA_DIR, 'whisper-tiny_encoder_state.safetensors')
FIXTURE = File.expand_path('../python/fixtures/whisper-tiny_greedy.json', __dir__)

# inner repeats the work inside one timed interval. A single encode is 6 ms,
# which is far under the second that AGENTS.md's rule 7 says a condition needs
# before its spread settles; looping it is what makes the interval long enough.
options = { length: 32, spelling: :shift, inner: 1,
            repeat: Integer(ENV.fetch('REPEAT', 3)) }
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/whisper_generate.rb [options]'
  o.on('--length N', Integer, '生成するトークン数') { |v| options[:length] = v }
  o.on('--spelling NAME', NArrayLLM::Whisper::Conv1d::SPELLINGS.map(&:to_s),
       "畳み込みの書き方 (#{NArrayLLM::Whisper::Conv1d::SPELLINGS.join(' / ')})") do |v|
    options[:spelling] = v.to_sym
  end
  o.on('--encode-only', 'encoder までで止める') { options[:encode_only] = true }
  o.on('--mel-only', 'メルまでで止める') { options[:mel_only] = true }
  o.on('--inner N', Integer, '計測の区間の中で繰り返す回数') { |v| options[:inner] = v }
  o.on('--rounds N', Integer, 'best-of-N の N (既定 3、環境変数 REPEAT でも指定できる)') { |v| options[:repeat] = v }
  o.on('--no-bench', '速度を測らない') { options[:no_bench] = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(ARGV.dup)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

[MODEL, DUMP, FIXTURE].each do |path|
  next if File.exist?(path)

  warn "#{path} not found. Run `ruby script/download_whisper.rb`, then " \
       'python/whisper_dump.py and python/whisper_fixtures.py.'
  exit 1
end

fixture = JSON.parse(File.read(FIXTURE))
dump = NArrayLLM::Safetensors.new(DUMP)
mel = NArrayLLM::Ops.contiguous(dump['input_features'])
dump.close

model = NArrayLLM::Whisper::Model.load(MODEL, spelling: options[:spelling])
config = model.config
puts "backend: #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:   whisper-tiny d_model=#{config.d_model} layers=#{config.encoder_layers}+" \
     "#{config.decoder_layers} mel=#{config.num_mel_bins}x#{config.mel_frames} V=#{config.vocab_size}"
puts "conv:    #{options[:spelling]}"
puts "prompt:  #{fixture['prompt'].inspect}"
puts

front = NArrayLLM::Whisper::Mel.load(File.join(DATA_DIR, 'whisper-tiny', 'preprocessor_config.json'))
waveform = nil
if options[:mel_only]
  d = NArrayLLM::Safetensors.new(DUMP)
  waveform = NArrayLLM::Ops.contiguous(d['waveform'], NArrayLLM::Whisper::Mel::PRECISION)
  d.close
end

run = lambda do
  if options[:mel_only]
    front.call(waveform)
  elsif options[:encode_only]
    model.encode(mel)
  else
    model.generate(mel, prompt: fixture['prompt'], max_new_tokens: options[:length],
                        suppress: fixture['suppress_tokens'],
                        begin_suppress: fixture['begin_suppress_tokens'])
  end
end

result = run.call
if options[:mel_only]
  puts "メル: #{result.shape.inspect}"
  produced = front.frames
elsif options[:encode_only]
  puts "encoder 出力: #{result.shape.inspect}"
  produced = config.max_source_positions
else
  puts "ids: #{result.inspect}"
  produced = result.size
end
exit 0 if options[:no_bench]

run.call # warm up
times = Array.new(options[:repeat]) do
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  options[:inner].times { run.call }
  NArrayLLM::Profiler::NULL.synchronize
  Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
end
best = times.min / options[:inner]
unit = if options[:mel_only] then 'frames'
       elsif options[:encode_only] then 'positions'
       else 'tokens'
       end
puts format('%.3f s / %d %s = %.2f %s/sec  (best of %d)', best, produced, unit,
            produced / best, unit, options[:repeat])
