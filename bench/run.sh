#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."

conditions=${1:?conditions file}
out=${2:?output directory}
rounds=${ROUNDS:-10}
export PYTHON=${PYTHON:-python/.venv/bin/python}
labels=(cumo cupy torch control)
mkdir -p "$out"

extract() {
  local label=$1 name=$2
  if [[ $label == cumo || $label == control ]]; then
    sed -nE 's#.*= ([0-9.]+) (tokens|positions|枚)/sec.*#\1#p' | head -1
  elif [[ $name == resnet_* ]]; then
    awk -F'\t' 'NF>3 {v=$NF} END {print v}'
  else
    awk -F'\t' 'NF>3 {v=$(NF-2)} END {print v}'
  fi
}

stop() {
  echo "STOPPED: $1 ($name, ${labels[$k]})"
  echo "  ${commands[$k]}"
  tail -5 "$out/last_stderr" | sed 's/^/  /'
  exit 1
}

while IFS=$'\t' read -r name ruby cupy torch; do
  [[ -z $name || $name == \#* ]] && continue
  commands=("$ruby" "$cupy" "$torch" "$ruby")
  file="$out/$name.tsv"
  : > "$file"
  for ((round = 0; round <= rounds; round++)); do
    for i in 0 1 2 3; do
      k=$(( (round + i) % 4 ))
      raw=$(timeout 1800 bash -c "${commands[$k]}" 2>"$out/last_stderr") || stop "exit $?"
      value=$(printf '%s\n' "$raw" | extract "${labels[$k]}" "$name")
      [[ $value =~ ^[0-9]+(\.[0-9]+)?$ ]] || stop "no number in the output"
      printf '%s\t%s\t%s\n' "$round" "${labels[$k]}" "$value" >> "$file"
    done
  done
  echo "done $name"
done < "$conditions"
