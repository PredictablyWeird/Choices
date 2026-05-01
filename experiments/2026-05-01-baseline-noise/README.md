# Experiment 3: baseline-to-baseline noise estimate

Calibrates the C3 claim ("a single sentence shifts choice rates by 15–18pp")
against the natural baseline-to-baseline drift from running the same prompt
twice. Without a noise floor, reviewers can ask "is that bigger than re-running
baseline?" and the paper has no answer for trolley/BBQ. (DailyDilemmas already
has this measurement.)

## Method

For every `(benchmark × model × factor)` cell:
1. Pick one existing baseline graph (one of the `*_base/` dirs in the original
   results — they share prompt and seed across nudge_types).
2. Re-issue the same prompts at temperature 1.0 with a fresh RNG draw, k=8
   trials per (edge × direction), saving the new responses to a parallel
   `simple_<factor>/<model>/base/<timestamp>_base/preference_graph_*.json`.
3. Compute:
   - `f_0(B)` shift in pp (the direct C3 calibrant)
   - per-edge modal-choice flip rate

## How to run

```bash
# Replicate one (benchmark, model) at a time
uv run python experiments/2026-05-01-baseline-noise/run_replicate.py \
    --benchmark trolley \
    --model deepseek-v3-2-non-reasoning \
    --concurrency 30

# (or BBQ; see --help for full args)

# Aggregate analysis
uv run python experiments/2026-05-01-baseline-noise/analyze.py \
    --benchmark trolley \
    --replicate-dir experiments/2026-05-01-baseline-noise/results_trolley_baseline_replicate \
    --original-dirs ~/code/values/moral-steerability-paper/google_drive/results_clean_arxiv \
                    ~/code/values/moral-steerability-paper/google_drive/results_extra_arxiv

uv run python experiments/2026-05-01-baseline-noise/analyze.py \
    --benchmark bbq \
    --replicate-dir experiments/2026-05-01-baseline-noise/results_bbq_baseline_replicate \
    --original-dirs ~/code/values/moral-steerability-paper/google_drive/results_bbq_v2
```

## Outputs

- `results_<benchmark>_baseline_replicate/simple_<factor>/<model>/base/...` —
  fresh preference_graph_*.json per cell.
- `analysis_<benchmark>/per_condition.csv` — one row per (model, factor) with
  drift and flip-rate stats.
- `analysis_<benchmark>/headline.json` — per-benchmark and per-model means.
- `analysis_<benchmark>/abs_drift_hist.png` — distribution of |drift| in pp.

## Where the result goes

- **Main text §4.2 C3 (line ~262)**: add a "Baseline noise" column to the
  headline table populated with the per-benchmark baseline-to-baseline shift
  in pp; update the C3 prose to reference the ratio:
  > "An order of magnitude larger than the baseline-to-baseline noise floor
  > (X–Ypp), making this not a re-run artifact but a real property of the
  > prompt manipulation."
- **Methods (§3.5)**: one-sentence note on the replication procedure (same
  prompts, fresh RNG draw, temperature 1.0, k=8 per direction).
- **Appendix**: per-condition replication table (per_condition.csv).
