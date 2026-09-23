# frozen_string_literal: true

module NArrayLLM
  module ResNet
    # ResNetForImageClassification. Takes pixels the way torch stores them,
    # [N, 3, H, W], and turns them once into the [N, H, W, C] the convolutions
    # want. The answer is [N, num_labels].
    class Model
      DEFAULT_SPELLING = :unfold

      attr_reader :config, :spelling, :folded

      def self.load(path, **options)
        new(Checkpoint.load(path), **options)
      end

      # fold: multiply each batch norm into the convolution before it, which
      # is exact arithmetic but not the same rounding. The class numbers do
      # not move; the logits move by 7e-06 (docs/plans/PLAN-conv2d.md).
      #
      # spelling takes one name for the whole model, or a Hash keyed by :stem
      # and the stage number with a :default. Which spelling wins is not the
      # same at every layer, since the window matrix spans 256x in rows
      # (docs/plans/PLAN-conv2d.md), and a Hash is how that gets measured and used.
      def initialize(checkpoint, spelling: DEFAULT_SPELLING, fold: false)
        @config = checkpoint.config
        @spelling = spelling
        @folded = fold
        @layout = Model.layout_of(spelling, @config.stages)
        @embedder = Embedder.new(checkpoint, spelling: Model.spelling_for(spelling, :stem),
                                             fold: fold)
        @stages = Array.new(@config.stages) do |stage|
          Stage.new(checkpoint, stage, spelling: Model.spelling_for(spelling, stage), fold: fold)
        end
        @classifier_t = Ops.contiguous(checkpoint["#{Checkpoint::CLASSIFIER}.weight"].transpose)
        @classifier_b = checkpoint["#{Checkpoint::CLASSIFIER}.bias"]
      end

      def self.spelling_for(spelling, part)
        return spelling unless spelling.is_a?(Hash)

        spelling.fetch(part) { spelling.fetch(:default) }
      end

      # Every part has to agree on the layout: the activations flow from one
      # to the next and nothing transposes between them.
      def self.layout_of(spelling, stages)
        parts = [:stem, *0...stages].map { |part| spelling_for(spelling, part) }
        layouts = parts.map { |name| Conv2d.layout_for(name) }.uniq
        raise Error, "spellings disagree on the layout: #{parts.inspect}" unless layouts.size == 1

        layouts.first
      end

      # pixels: [N, 3, H, W]. Answers [N, num_labels].
      def forward(pixels, prof: Profiler::NULL)
        # cuDNN takes the pixels the way torch stores them, so that arm pays
        # no transpose at all. The matrix products want the channels last.
        x = pixels
        x = prof.section(:layout) { Ops.contiguous(pixels.transpose(0, 2, 3, 1)) } if @layout == :nhwc
        x = @embedder.call(x, prof: prof)
        x = @stages.reduce(x) { |acc, stage| stage.call(acc, prof: prof) }
        pooled = prof.section(:pool) { Ops.global_average_pool(x, layout: @layout) }
        prof.section(:classifier) { Ops.linear(pooled, @classifier_t, @classifier_b) }
      end

      # The class numbers, as Ruby Integers. max_index answers a flattened
      # running number (AGENTS.md), so the row offsets come off first.
      def classify(pixels, prof: Profiler::NULL)
        logits = forward(pixels, prof: prof)
        rows, labels = logits.shape
        flat = logits.max_index(axis: 1)
        (flat - (flat.class.new(rows).seq * labels)).to_a
      end
    end
  end
end
