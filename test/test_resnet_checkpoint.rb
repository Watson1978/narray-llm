# frozen_string_literal: true

require_relative 'test_helper'

# Stage 0 acceptance tests (docs/plans/PLAN-conv2d.md).
#
# The expected numbers come from microsoft/resnet-18's config.json and from
# the model as transformers 5.16.1 builds it, read on 2026-09-21. The shapes
# the loader checks are computed from the config, so a file that disagrees
# with its own config is rejected.
class TestResNetCheckpoint < Test::Unit::TestCase
  include TestHelper

  MODEL = File.expand_path('../data/resnet-18/model.safetensors', __dir__)
  STATE = File.expand_path('../data/resnet-18_state.safetensors', __dir__)
  FIXTURE = File.expand_path('../python/fixtures/resnet-18_classes.json', __dir__)

  NOMINAL = {
    embedding_size: 64, hidden_sizes: [64, 128, 256, 512], depths: [2, 2, 2, 2],
    layer_type: 'basic', hidden_act: 'relu', downsample_in_first_stage: false,
    num_labels: 1000
  }.freeze

  TENSORS = 122
  # What torch reports for the same model. The running statistics are buffers
  # and are not counted, which is the whole reason this number can disagree.
  PARAMETERS = 11_689_512
  CONVOLUTIONS = 20
  IMAGES = 16

  def setup
    omit("#{MODEL} not found") unless File.exist?(MODEL)

    @checkpoint = NArrayLLM::ResNet::Checkpoint.load(MODEL)
  end

  def teardown
    @checkpoint&.close
  end

  def test_config_matches_the_published_one
    config = @checkpoint.config
    NOMINAL.each { |field, want| assert_equal(want, config[field], field.to_s) }
    assert_equal(4, config.stages)
  end

  def test_the_file_holds_what_the_config_says
    assert_equal(TENSORS, @checkpoint.names.size)
    assert_equal(PARAMETERS, @checkpoint.num_parameters)
  end

  # 16 of 3x3, 3 of 1x1 and the 7x7 stem, which is what the model has.
  def test_every_convolution_is_accounted_for
    convolutions = @checkpoint.convolutions
    assert_equal(CONVOLUTIONS, convolutions.size)
    kernels = convolutions.map { |_, weight, _, _| weight[2..] }
    assert_equal(16, kernels.count([3, 3]))
    assert_equal(3, kernels.count([1, 1]))
    assert_equal(1, kernels.count([7, 7]))

    _, stem, stride, padding = convolutions.first
    assert_equal([64, 3, 7, 7], stem)
    assert_equal(2, stride)
    assert_equal(3, padding)
  end

  # A stage that changes width or stride needs the 1x1 shortcut, and the first
  # one changes neither.
  def test_only_the_later_stages_have_a_shortcut
    config = @checkpoint.config
    assert_false(config.shortcut?(0))
    (1..3).each { |stage| assert_true(config.shortcut?(stage), "stage #{stage}") }
    assert_equal([1, 2, 2, 2], (0..3).map { |stage| config.stride_for(stage) })
  end

  def test_the_weights_read_back_at_the_shape_the_config_gives
    @checkpoint.convolutions.each do |prefix, want, _stride, _padding|
      weight = @checkpoint["#{prefix}.convolution.weight"]
      assert_equal(want, weight.shape, prefix)
      assert_equal([want[0]], @checkpoint["#{prefix}.normalization.weight"].shape, prefix)
    end
  end

  def test_the_reference_activations_are_readable
    omit("#{STATE} not found; run `rake prepare:resnet`") unless File.exist?(STATE)

    NArrayLLM::Safetensors.open(STATE) do |store|
      assert_equal([IMAGES, 3, 224, 224], store.shape('pixel_values'))
      assert_equal([IMAGES, 64, 112, 112], store.shape('conv1'))
      assert_equal([IMAGES, 64, 56, 56], store.shape('embedder'))
      assert_equal([IMAGES, 512, 7, 7], store.shape('stage.3'))
      assert_equal([IMAGES, 1000], store.shape('logits'))
    end
  end

  # The gate is agreement with transformers on the class number, so the
  # fixture has to carry one per image and they have to be the argmax of the
  # logits dumped beside them.
  def test_the_class_fixture_agrees_with_the_dumped_logits
    omit("#{FIXTURE} not found; run python/resnet_fixtures.py") unless File.exist?(FIXTURE)
    omit("#{STATE} not found; run `rake prepare:resnet`") unless File.exist?(STATE)

    want = JSON.parse(File.read(FIXTURE))['images']
    assert_equal(IMAGES, want.size)

    NArrayLLM::Safetensors.open(STATE) do |store|
      logits = store['logits']
      want.each_with_index do |row, i|
        flat = logits[i, true]
        chosen = NArrayLLM.scalar(flat.max_index).to_i
        assert_equal(row['class'], chosen, row['image'])
      end
    end
  end
end
