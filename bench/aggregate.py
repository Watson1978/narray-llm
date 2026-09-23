import collections
import os
import statistics
import sys

conditions, out = sys.argv[1], sys.argv[2]
names = [line.split("\t", 1)[0] for line in open(conditions)
         if line.strip() and not line.startswith("#")]

print("| 条件 | Cumo | CuPy | PyTorch | Cumo / CuPy | Cumo / PyTorch | 対照 / Cumo |")
print("|---|---|---|---|---|---|---|")
for name in names:
    rows = collections.defaultdict(dict)
    for line in open(os.path.join(out, f"{name}.tsv")):
        round_, label, value = line.split("\t")
        if int(round_) >= 1:
            rows[int(round_)][label] = float(value)
    rounds = [r for r in sorted(rows) if len(rows[r]) == 4]
    median = {k: statistics.median(rows[r][k] for r in rounds)
              for k in ("cumo", "cupy", "torch")}

    def ratio(numerator, denominator):
        values = [rows[r][numerator] / rows[r][denominator] for r in rounds]
        text = f"{statistics.median(values):.3f} {sum(v > 1 for v in values)}/{len(values)}"
        if min(values) < 1 < max(values):
            text += " またぐ"
        return text

    print(f"| {name} | {median['cumo']:.1f} | {median['cupy']:.1f} | {median['torch']:.1f} | "
          f"{ratio('cumo', 'cupy')} | {ratio('cumo', 'torch')} | {ratio('control', 'cumo')} |")
