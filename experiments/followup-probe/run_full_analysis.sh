#!/usr/bin/env bash
# Run the full Exp 2 analysis pipeline, end to end. Assumes that
# `run_followup.py` and `classify.py` have already produced
# results/{followup,classified}_<benchmark>_<model>.jsonl for each
# (benchmark, model) cell.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SUMMARIES=(
  "$ROOT/../2026-05-01-asymmetry-baseline-regression/data/trolley_summary.csv"
  "$ROOT/../2026-05-01-asymmetry-baseline-regression/data/bbq_summary.csv"
)

# Classify any followup_*.jsonl that doesn't have a corresponding classified_*.
for fp in "$ROOT"/results/followup_*.jsonl; do
  base=$(basename "$fp" .jsonl)
  out="$ROOT/results/classified_${base#followup_}.jsonl"
  if [[ ! -f "$out" ]]; then
    echo "Classifying $base ..."
    uv run python "$ROOT/classify.py" \
      --input "$fp" \
      --output "$out" \
      --judge-model gpt-4o-mini \
      --concurrency 50
  else
    echo "Already classified: $(basename "$out")"
  fi
done

# Aggregate analysis across all classified outputs.
uv run python "$ROOT/analyze.py" \
  --classified "$ROOT"/results/classified_*.jsonl \
  --summary-csvs "${SUMMARIES[@]}" \
  --output-dir "$ROOT/analysis_out"
