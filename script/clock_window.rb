# frozen_string_literal: true

# Records the wall-clock window of the iteration a benchmark reports, so that a
# separately running `nvidia-smi ... --format=csv` log can be joined against it
# afterwards.
#
# An external sampler on its own cannot tell "the stage never rose" from "the
# sampler never looked" (AGENTS.md の計測の作法 15), and the outlier this
# repository keeps hitting is a whole run that stayed on a lower memory stage.
# Sampling from inside the timed process was tried first and rejected: driving
# nvidia-smi from within cost 2.7% and widened the spread from 5.9% to 9.1%
# (docs/results/llama2-110m.md). Printing two timestamps costs nothing, and the
# stage still comes from samples taken while the work was running.
#
# The format matches nvidia-smi's own `timestamp` field, so the join is a
# string comparison.
module ClockWindow
  FORMAT = '%Y/%m/%d %H:%M:%S.%L'

  module_function

  def enabled?
    ENV['CLOCKS'].to_s =~ /\A(1|on|true)\z/i
  end

  def stamp
    Time.now.strftime(FORMAT)
  end

  def report(window)
    return nil unless enabled? && window

    format('window: %s -> %s', *window)
  end
end
