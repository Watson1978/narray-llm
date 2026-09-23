require 'rake/testtask'

ROOT = __dir__
DATA_DIR = ENV.fetch('NARRAY_LLM_DATA') { File.join(ROOT, 'data') }
PYTHON = ENV.fetch('PYTHON') { File.join(ROOT, 'python/.venv/bin/python') }

Rake::TestTask.new(:test) do |t|
  t.libs << 'lib' << 'test'
  t.test_files = FileList['test/test_*.rb']
  t.warning = false
end

def data(name) = File.join(DATA_DIR, name)

LLAMA2_FILES = %w[stories260K.bin tok512.bin stories110M.bin tokenizer.bin run.c
                  runq.c export.py model.py stories15M.bin stories15M.pt stories110M.pt].freeze
SWITCH_128_TOKENS = [*100..226, 1].join(',')

namespace :download do
  desc 'GPT-2: the llm.c starter pack (weights, reference state, tokenizer)'
  task(:gpt2) { ruby 'script/download_gpt2.rb' }

  desc 'Llama 2: llama2.c checkpoints and tokenizers, and the sources int8 is exported with'
  task(:llama2) { ruby 'script/download_llama2.rb', *LLAMA2_FILES }

  desc 'Mamba: mamba.c sources and the mamba-130m weights'
  task(:mamba) { ruby 'script/download_mamba.rb' }

  desc 'Switch: the switch-base-8 config, tokenizer and pickled weights'
  task(:switch) { ruby 'script/download_switch.rb' }

  desc 'Whisper: the whisper-tiny config, tokenizer and weights'
  task(:whisper) { ruby 'script/download_whisper.rb' }

  desc 'ResNet-18: the config, preprocessing and weights'
  task(:resnet) { ruby 'script/download_resnet.rb' }
end

desc 'Download what every model needs (Ruby and curl only)'
task download: %w[gpt2 llama2 mamba switch whisper resnet].map { |m| "download:#{m}" }

file(data('stories260K_debug_state.bin')) { ruby 'script/llama2_dump.rb', 'stories260K.bin' }
file(data('stories110M_debug_state.bin')) { ruby 'script/llama2_dump.rb', 'stories110M.bin' }
%w[stories15M stories110M].each do |model|
  file(data("#{model}_q80.bin")) do
    Dir.chdir(File.join(ROOT, 'vendor/llama2.c')) do
      sh PYTHON, 'export.py', data("#{model}_q80.bin"), '--version', '2', '--checkpoint', data("#{model}.pt")
    end
  end
end

file(data('mamba-130m.bin')) do
  Dir.chdir(File.join(ROOT, 'vendor/mamba.c')) { sh PYTHON, 'export.py', data('mamba-130m'), data('mamba-130m.bin') }
end
file(data('mamba_tokenizer.bin')) do
  Dir.chdir(File.join(ROOT, 'vendor/mamba.c')) { sh PYTHON, 'tokenizer.py' }
  mv File.join(ROOT, 'vendor/mamba.c/tokenizer.bin'), data('mamba_tokenizer.bin')
end
file(data('mamba_tiny.bin')) { ruby 'script/mamba_tiny.rb' }
file(data('mamba_tiny_debug_state.bin')) { ruby 'script/mamba_dump.rb', 'mamba_tiny.bin' }
file(data('mamba-130m_debug_state.bin')) { ruby 'script/mamba_dump.rb', 'mamba-130m.bin' }

file(data('switch-base-8.safetensors')) do
  sh PYTHON, 'python/export_switch.py', data('switch-base-8'), data('switch-base-8.safetensors')
end
file(data('switch-base-8_encoder_state.safetensors')) do
  sh PYTHON, 'python/switch_dump.py', data('switch-base-8'), data('switch-base-8_encoder_state.safetensors')
end
file(data('switch-base-8_encoder_state_128.safetensors')) do
  sh PYTHON, 'python/switch_dump.py', data('switch-base-8'), data('switch-base-8_encoder_state_128.safetensors'),
     '--tokens', SWITCH_128_TOKENS
end

file(data('whisper-tiny_encoder_state.safetensors')) do
  sh PYTHON, 'python/whisper_dump.py', data('whisper-tiny'), data('whisper-tiny_encoder_state.safetensors')
end

file(data('resnet-18_state.safetensors')) do
  sh PYTHON, 'python/resnet_dump.py', data('resnet-18'), data('resnet-18_state.safetensors')
end

PREPARED = {
  llama2: %w[stories260K_debug_state.bin stories110M_debug_state.bin stories15M_q80.bin stories110M_q80.bin],
  mamba: %w[mamba-130m.bin mamba_tokenizer.bin mamba_tiny.bin mamba_tiny_debug_state.bin mamba-130m_debug_state.bin],
  switch: %w[switch-base-8.safetensors switch-base-8_encoder_state.safetensors
             switch-base-8_encoder_state_128.safetensors],
  whisper: %w[whisper-tiny_encoder_state.safetensors],
  resnet: %w[resnet-18_state.safetensors]
}.freeze

namespace :prepare do
  desc 'GPT-2: nothing beyond the download, which already carries the reference state'
  task gpt2: 'download:gpt2'

  PREPARED.each do |model, files|
    desc "#{model}: download, then build #{files.join(', ')} where missing"
    task model => ["download:#{model}", *files.map { |f| data(f) }]
  end
end

desc 'Everything the tests read: downloads, converted weights and reference values ' \
     '(needs python/.venv and a C compiler)'
task prepare: %w[gpt2 llama2 mamba switch whisper resnet].map { |m| "prepare:#{m}" }

task default: :test
