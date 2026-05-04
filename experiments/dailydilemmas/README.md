# DailyDilemmas Experiment

## Reasoning Trace Analysis

### Fig.9.dd (Daily Dilemmas — primary rationales)

You need three kinds of paths (all arbitrary on disk; point the CLI flags at them):

1. **Batch results (`--results-root`)** — A directory whose layout matches the Daily Dilemmas batch output: a `baseline/` folder and one subdirectory per nudge type (e.g. `survey`, `emotional`), each containing `results.json` with per-dilemma responses and reasoning.
2. **Party metadata (`--party-csv`)** — A CSV such as `action_to_party_to_value.csv` that maps dilemmas/actions to party rows. Required only when you use party-based filters (`smaller` / `party_unequal`); for `--baseline-filter all` and `--nudged-filter all` it is still read for counts but you must supply a valid path.
3. **Outputs** — Paths for edge JSON, rationale JSON, and PDFs (any folder you choose).

Example layout (names are illustrative): suppose you keep everything for Daily Dilemmas under `dd_dataset/`:

```text
dd_dataset/
  results/<model>/          # --results-root: baseline/ + one folder per nudge, each with results.json
  action_to_party_to_value.csv
  fig9/                       # optional: edges, rationales, PDFs for this figure
```

The commands below use `analysis/results_global/...`, `analysis/dd_dataset/...`, and `analysis/dd_fig9/...` as a concrete instance of that pattern. Large or machine-specific trees (`analysis/dd_dataset/`, `analysis/daily_dilemmas/`, `analysis/dd_fig9/`, etc.) are gitignored in this repo; copy or regenerate data locally.

- Build edges (include all non-null reasoning traces; relax party-based filters):
  -`uv run python experiments/dailydilemmas/daily_dilemmas_results_to_edges.py \
  --results-root analysis/results_global/deepseek-v3-2-reasoning \
  --party-csv analysis/dd_dataset/action_to_party_to_value.csv \
  --output-baseline analysis/dd_fig9/edges_baseline_smaller.json \
  --output-nudged analysis/dd_fig9/edges_nudged_smaller.json \
  --baseline-filter all --nudged-filter all
  `

- Rationale detection (baseline):
  -`uv run python experiments/dailydilemmas/rationale_detection.py \
  --input analysis/dd_fig9/edges_baseline_smaller.json \
  --output analysis/dd_fig9/baseline_smaller_rationales.json \
  --condition baseline \
  --max-samples 200 --max-total-samples 1000 --seed 42
  `

- Rationale detection (nudged; stratified by `nudge_type`, total capped):
  -`uv run python experiments/dailydilemmas/rationale_detection.py \
  --input analysis/dd_fig9/edges_nudged_smaller.json \
  --output analysis/dd_fig9/nudged_smaller_rationales.json \
  --condition nudged \
  --max-samples 200 --max-total-samples 1000 --seed 42
  `

- Generate plot (primary rationale):
  - `uv run python -m choices.analysis.reasoning_traces.plot_rationales \
  --source analysis/dd_fig9/baseline_smaller_rationales.json,baseline,"Baseline" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,survey,"Survey" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,few_shot_value,"Few-shot" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,user_preference,"User Preference" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,role_play,"Role-play" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,weak_evidence,"Weak Evidence" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,emotional,"Emotional" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,virtue,"Virtue" \
  --metric primary --threshold 5.0 --show-pct --pdf --no-title \
  --output analysis/dd_fig9/fig9_primary_rationales_daily_dilemmas_deepseek.pdf
  `

### Fig.19.dd (Daily Dilemmas — mentioned rationales)

Uses the same rationale JSON files as **Fig.9.dd**; only the plot metric changes.

  - `uv run python -m choices.analysis.reasoning_traces.plot_rationales \
  --source analysis/dd_fig9/baseline_smaller_rationales.json,baseline,"Baseline" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,survey,"Survey" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,few_shot_value,"Few-shot" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,user_preference,"User Preference" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,role_play,"Role-play" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,weak_evidence,"Weak Evidence" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,emotional,"Emotional" \
  --source analysis/dd_fig9/nudged_smaller_rationales.json,virtue,"Virtue" \
  --metric mentioned --threshold 5.0 --show-pct --pdf --no-title \
  --output analysis/dd_fig9/fig19_mentioned_rationales_daily_dilemmas_deepseek.pdf
  `
