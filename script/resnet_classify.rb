#!/usr/bin/env ruby
# frozen_string_literal: true

# Classifies the images python/resnet_dump.py preprocessed, with ResNet-18.
#
#   ruby script/resnet_classify.rb                      # 16 images, checks the fixture
#   GPU=1 ruby script/resnet_classify.rb
#   GPU=1 ruby script/resnet_classify.rb --spelling shift
#   GPU=1 ruby script/resnet_classify.rb --fold         # batch norm folded into the convolutions
#   GPU=1 ruby script/resnet_classify.rb --inner 50     # time it, 50 passes inside the region
#   ruby script/resnet_classify.rb --no-check           # skip the class number check
#
# The gate is the class number, not the true label: the reference is
# transformers and an implementation that reproduces its mistakes is what this
# is asking for (docs/plans/PLAN-conv2d.md).

require 'json'
require 'optparse'

require_relative '../lib/narray_llm'

DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.expand_path('../data', __dir__) }
FIXTURE = File.expand_path('../python/fixtures/resnet-18_classes.json', __dir__)

options = { spelling: :unfold, fold: false, repeat: Integer(ENV.fetch('REPEAT', 3)),
            batch: nil, passes: 1 }
argv = ARGV.dup
parser = OptionParser.new do |o|
  o.banner = 'Usage: ruby script/resnet_classify.rb [options]'
  o.on('--spelling NAME', NArrayLLM::ResNet::Conv2d::SPELLINGS.map(&:to_s),
       "畳み込みの書き方 (#{NArrayLLM::ResNet::Conv2d::SPELLINGS.join(' / ')})") do |v|
    options[:spelling] = v.to_sym
  end
  # Everything unfold except one part, which is how the layer where the two
  # spellings change sign gets found (docs/plans/PLAN-conv2d.md).
  o.on('--shift-at PART', 'stem か 0..3。そこだけ shift、残りは unfold') do |v|
    options[:spelling] = { default: :unfold, (v == 'stem' ? :stem : Integer(v)) => :shift }
  end
  o.on('--fold', 'BatchNorm を畳み込みに畳む') { options[:fold] = true }
  o.on('--batch N', Integer, '先頭 N 枚だけ使う') { |v| options[:batch] = v }
  # One pass is 16 images and a few milliseconds, which is far under the
  # second the measurement rules ask for, so the region repeats (AGENTS.md).
  o.on('--inner N', Integer, '計測の区間の中で通す回数') { |v| options[:passes] = v }
  o.on('--repeat N', Integer, 'inner の別名 (以前の綴り)') { |v| options[:passes] = v }
  o.on('--rounds N', Integer, 'best-of-N の N (既定 3、環境変数 REPEAT でも指定できる)') do |v|
    options[:repeat] = v
  end
  o.on('--no-check', 'クラス番号を参照と照合しない') { options[:no_check] = true }
  o.on('--no-bench', '速度を測らない') { options[:no_bench] = true }
  o.on('-h', '--help', 'この説明を出す') { puts o; exit 0 }
end
begin
  parser.parse!(argv)
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 1
end

state = NArrayLLM::Safetensors.new(File.join(DATA_DIR, 'resnet-18_state.safetensors'))
pixels = state['pixel_values']
pixels = NArrayLLM::Ops.contiguous(pixels[0...options[:batch], true, true, true]) if options[:batch]
checkpoint = NArrayLLM::ResNet::Checkpoint.load(File.join(DATA_DIR, 'resnet-18/model.safetensors'))
model = NArrayLLM::ResNet::Model.new(checkpoint, spelling: options[:spelling], fold: options[:fold])

puts "backend:  #{NArrayLLM.gpu? ? 'Cumo (GPU)' : 'Numo (CPU)'}, #{NArrayLLM.dtype_name}"
puts "model:    ResNet-18, #{checkpoint.num_parameters} parameters"
puts "spelling: #{options[:spelling].inspect}#{options[:fold] ? ', batch norm folded' : ''}"
puts "images:   #{pixels.shape[0]}"
puts

classes = model.classify(pixels)
unless options[:no_check]
  want = JSON.parse(File.read(FIXTURE))['images']
  labels = want.map { |row| row['label'] }
  wrong = classes.each_with_index.reject { |id, i| want[i] && want[i]['class'] == id }
  classes.each_with_index do |id, i|
    puts format('  %-28s -> %4d  %s', want[i] ? want[i]['image'] : '?', id, labels[i])
  end
  unless wrong.empty?
    warn "class numbers differ from the reference at #{wrong.map(&:last).inspect}"
    exit 1
  end
  puts
  puts "#{classes.size} 枚すべてが参照と一致"
end
exit 0 if options[:no_bench]

# Twice: cuDNN searches for an algorithm the first time it sees a shape, and
# the cache is keyed by the workspace limit, so an arm that changes it pays
# the search again. One pass covers all twenty shapes; the second is
# insurance that the search is not still in the first timed round.
2.times { model.forward(pixels) }
NArrayLLM::Profiler::NULL.synchronize
times = Array.new(options[:repeat]) do
  started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  options[:passes].times { model.forward(pixels) }
  NArrayLLM::Profiler::NULL.synchronize
  Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
end

best = times.min
images = pixels.shape[0] * options[:passes]
puts
puts "== 速度 (best-of-#{options[:repeat]}, #{options[:passes]} 回通す) =="
puts format('%.3f s / %d 枚 = %.1f 枚/sec', best, images, images / best)
puts format('各回: %s', times.map { |t| format('%.3f', t) }.join(', '))
