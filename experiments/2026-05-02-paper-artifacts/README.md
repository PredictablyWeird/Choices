# Paper artifacts — single-source-of-truth pipeline

Produces every numeric claim cited by `main_phil.tex` (the NeurIPS 2026
draft in `moral-steerability-paper/neurips_2026_rewrite/`) as one JSON,
plus first-pass figures.

## Why

`claim_verification.md` (in the latex repo) caught three internal
number drifts in the previous draft:

- DailyDilemmas backfire was 5.9% in the headline figure but 0.2% in
  the appendix; the data supports 0.2%.
- Trolley backfire was 14.3% but the data supports 17.7%.
- DailyDilemmas headline shift was 9pp but the data supports 12.3pp
  (or 15.4pp restricted to one-sentence user-message cues).

A separate adversarial review additionally argued the paper's
significance machinery was too thin for the design — IID Bernoulli
z-tests across 1,498 conditions, no FDR, no cluster bootstrap.

The fix is a single place where every paper-citable number is
computed once, against a frozen data snapshot, with the cluster
bootstrap and BH-FDR robustness checks built in. That is this
directory.

## What's here

```
produce_paper_artifacts.py    master script — run this
extract_pvalues.py            standalone p-value extraction
                              (preference_graph -> Wald test)
robustness.py                 standalone robustness pass
                              (cluster bootstrap + FDR + equivalence
                              margins)
canonical_numbers.py          earlier exploratory script, kept for
                              reference
PAPER_NUMBERS.md              human-readable diff vs the previous
                              paper headlines

data/                         outputs
  paper_numbers.json            <- the JSON the paper rewrite reads
  pvalues_by_condition.csv      <- exact Wald p-values per condition

figures/                      first-pass figures
  cross_benchmark.png           <- 3-panel headline figure
```

## How to run

```bash
cd Choices
uv run --with pandas --with numpy --with scipy --with statsmodels \
       --with matplotlib python \
       experiments/2026-05-02-paper-artifacts/produce_paper_artifacts.py
```

Re-extraction of p-values from preference_graph JSONs is slow on a
cold cache (~1–2 min); the script caches the output to
`data/pvalues_by_condition.csv` and re-uses it on subsequent runs.

## Inputs

The pipeline reads from three places, in order of precedence:

1. **`Choices/experiments/2026-05-01-asymmetry-baseline-regression/data/`**
   — per-benchmark summary CSVs produced by `choices.analysis.create_summary`.
   These are the May-1 frozen snapshot used by the regression experiment;
   we treat them as the canonical per-condition source.
2. **`moral-steerability-paper/google_drive/results_*`** — raw
   preference_graph JSONs, used to recompute Wald p-values from raw
   counts (the summary CSVs only carry boolean sig flags).
3. **`Choices/experiments/2026-05-01-{baseline-noise,followup-probe}/`**
   — pre-computed headline JSONs from sibling experiments; pulled in
   without recomputation.

## Paper-side scope

- Models: 9 configs (DeepSeek/Grok/GPT-5.2/Llama/Qwen, with reasoning
  on/off where applicable). gpt-4o-mini and llama-4-maverick exist in
  the source CSVs but are filtered out — the paper only cites these 9.
- Trolley factors: 5 (age_group, gender, handedness, nationality, wealth).
  diet and tech_view exist in the source CSVs but are filtered out.
- DailyDilemmas: excludes `dishonesty` (saturated baseline; paper
  excludes this in App. E).

## What this doesn't yet do

- Phrasing-robustness table (`tab:phrasing-young`) and realistic-scenario
  table (`tab:realistic-scenarios`) — raw data not in `google_drive/`.
  Tagged as `_status: deferred` in the JSON.
- Surface-form / negation / nonsensical tables — `choices/analysis/surface_form/`
  has the code, but it isn't wired in here yet.
- Most figures beyond the cross-benchmark headline. The
  `choices/analysis/plots/` library produces the per-model violins,
  steerability plots, etc.; first pass focuses on the load-bearing
  numbers.
