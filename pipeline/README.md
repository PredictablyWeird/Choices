# NeurIPS 2026 paper artifacts — single-source-of-truth pipeline

Produces every numeric claim cited by the **NeurIPS 2026 submission**
*Direction-Flipped Influence Audits Reveal Hidden Structure in LLM
Moral-Choice Benchmarks* (`main.tex` in
`moral-steerability-paper/neurips_2026/`) as one JSON, plus
first-pass figures.

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
  paper_numbers.json            <- the JSON the paper reads
  paper_numbers.tex             <- \input-able LaTeX with macros &
                                   appendix-table row bodies
  pvalues_by_condition.csv      <- exact Wald p-values per condition

figures/                      first-pass figures
  cross_benchmark.png           <- 3-panel headline figure
```

### Using `paper_numbers.tex` in the LaTeX source

The pipeline emits `pipeline/data/paper_numbers.tex` alongside the
JSON. It defines `\newcommand`s for inline-cited numbers (headline
shifts, asymmetry rates, backfire rates, regression coefficients,
follow-up-probe rate, baseline-noise floors) plus row-body macros for
the major appendix tables (F.4–F.7, J, K, D).

To use, copy `paper_numbers.tex` into the LaTeX source tree (e.g.
the Overleaf project) and `\input{paper_numbers}` once in the
preamble. Then:

- Inline: `... shifts of \HeadlineShiftTriagePP\,pp on triage ...`
  produces `... shifts of 15.0 pp on triage ...`.
- Tables: keep the existing `\begin{table}` / `\caption` /
  `\begin{tabular}` scaffolding and replace the hand-typed body with
  one of the `\PaperTab*Rows` macros, e.g.

  ```latex
  \begin{tabular}{lrrrrrrr}
    \toprule
    Nudge & n & |Effect| & |Steer| & |Asym| & |N-Asym| & Sig & BF \\
    \midrule
    \PaperTabFNudgeEffectsTrolleyRows
    \bottomrule
  \end{tabular}
  ```

The full list of generated commands and table macros is in the
header comment of `paper_numbers.tex` itself.

## How to run

The script declares its dependencies inline (PEP 723), so just:

```bash
cd Choices
uv run pipeline/produce_paper_artifacts.py
```

(or `./pipeline/produce_paper_artifacts.py`
since the script is executable and has a `uv run --script` shebang.)

Re-extraction of p-values from preference_graph JSONs is slow on a
cold cache (~1–2 min); the script caches the output to
`data/pvalues_by_condition.csv` and re-uses it on subsequent runs.

## Inputs

The pipeline reads from three places, in order of precedence:

1. **`Choices/experiments/asymmetry-regression/data/`**
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

## Reasoning-trace coverage

The paper's §6 (reasoning traces) is currently trolley-only, but the
underlying Gemini-Flash classifications and edge-level traces for BBQ
and DailyDilemmas already exist in `google_drive/`. The pipeline now
surfaces them under `section_5_reasoning_traces` in the JSON:

- per-slice (`baseline` / `nudged` for BBQ; `baseline_smaller` /
  `nudged_smaller` for DailyDilemmas) primary-rationale counts +
  percentages, mentioned-rate, and acted-on-rate across all 23
  rationale codes;
- BBQ backfire-by-nudge / by-factor / by-primary-rationale crosstabs
  (from `backfire_examples.json`);
- DeepSeek `fig9_primary_rationales_*` and `fig19_mentioned_rationales_*`
  PDFs copied into `figures/` for direct paper inclusion.

The trolley analogues are produced upstream by
`choices/analysis/reasoning_traces/plot_rationales.py` and aren't
duplicated here.

## What this doesn't yet do

- Surface-form / negation / nonsensical tables — `choices/analysis/surface_form/`
  has the code, but it isn't wired in here yet.
- Most figures beyond the cross-benchmark headline. The
  `choices/analysis/plots/` library produces the per-model violins,
  steerability plots, etc.; the first pass focuses on the load-bearing
  numbers and the BBQ/DD rationale figures that were already rendered.

## Hard-coded numbers

A small number of paper tables are populated from values in this
script rather than re-derived from per-condition summary CSVs at
run time:

- **`tab_phrasing_young`** — Table `tab:phrasing-young` in App. L of
  the NeurIPS draft. Tagged `_status: "hardcoded"` in the JSON. The
  values come from a phrasing-sensitivity sweep over the
  user-preference cue on the age factor: 11 wording variants × 4
  models (GPT-5.2, Grok 4.1 Fast, DeepSeek V3.2, Llama 3.3 70B) ×
  both steering directions. Model baselines (no influence) are stored
  alongside the variant grid under `baselines_pct`.

  To recompute end-to-end, extend
  `choices.experiments.nudging.templates` with the 11 user-preference
  wording variants (Original / All caps / Typos / Lowercase / Extra
  spaces / Synonym / Contraction / Passive voice / Reorder / Filler
  words / Exclamation; three are written out in the appendix as
  worked examples), then run the standard nudging batch on
  `age_group × {gpt-5-2, grok-41-fast, deepseek-v3-2, llama-33-70b}`
  in both directions. The output flows through the same
  `choices.analysis.create_summary` path as every other condition,
  and a `_load_*` helper can replace `_hardcoded_phrasing_young()`.

- **`tab_realistic_scenarios.{visa_processing, emergency_triage}`** —
  rows of Table `tab:realistic-scenarios` in App. D. The `wildfire`
  row is computed in `_load_realistic_scenarios()` from raw
  preference-graph counts at
  `Choices/google_drive/simple_wildfire_implied_wealth/...`. The
  `visa` and `emergency_triage` rows are populated from the rates in
  the published table; their raw response logs are not currently in
  `google_drive/`. Steerability values for the hard-coded rows are
  derived from those rates using the same Haldane-corrected
  pooled-odds formula as the wildfire row.
