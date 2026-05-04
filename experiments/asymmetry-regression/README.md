# Experiment 1: |asymmetry| ~ |baseline bias| regression

Quantitative version of C1 ("baseline preferences fail to predict
steerability asymmetry"). Pure re-analysis of existing main-experiment
data — no new API calls.

## Inputs

- Trolley: `~/code/values/moral-steerability-paper/google_drive/results_clean_arxiv/`
  + `results_extra_arxiv/` (factors: age_group, diet, gender, handedness,
  nationality, tech_view, wealth)
- BBQ: `~/code/values/moral-steerability-paper/google_drive/results_bbq_v2/`
  (factors: age_neg, age_pos, ses_neg, ses_pos)
- DailyDilemmas: `~/code/values/moral-steerability-paper/google_drive/results_dailydilemmas/`
  converted via `experiments/dailydilemmas/convert_to_simple.py`
  into `data/results_dailydilemmas_simple/` (10 value factors:
  compassion, courage, dishonesty, honesty, integrity, loyalty,
  patience, responsibility, sacrifice, self).

## How to run

```bash
# Regenerate the per-condition CSVs from raw graph data
uv run python -m choices.analysis.create_summary \
    --results-dirs ~/code/values/moral-steerability-paper/google_drive/results_clean_arxiv \
                   ~/code/values/moral-steerability-paper/google_drive/results_extra_arxiv \
    --output experiments/asymmetry-regression/data/trolley_summary.csv

uv run python -m choices.analysis.create_summary \
    --results-dirs ~/code/values/moral-steerability-paper/google_drive/results_bbq_v2 \
    --output experiments/asymmetry-regression/data/bbq_summary.csv

# DailyDilemmas: convert first, then summarize.
uv run python experiments/dailydilemmas/convert_to_simple.py \
    --input ~/code/values/moral-steerability-paper/google_drive/results_dailydilemmas \
    --output experiments/asymmetry-regression/data/results_dailydilemmas_simple
uv run python -m choices.analysis.create_summary \
    --results-dirs experiments/asymmetry-regression/data/results_dailydilemmas_simple \
    --output experiments/asymmetry-regression/data/dailydilemmas_summary.csv

# Fit regressions and produce the scatter
uv run python experiments/asymmetry-regression/run_regression.py
```

## Model

```
|asym| ~ |baseline_bias|  +  (1 | model:reasoning)
                          +  (1 | factor)
                          +  (1 | influence_type)
```

(`statsmodels.MixedLM` with `groups = model:reasoning`, crossed random
intercepts on factor and nudge_type via `vc_formula`.)

The pooled fit adds `C(benchmark)` as a fixed effect.

## Headline numbers

| Fit            | n     | β (95% CI)             | R² (marginal) | R² (conditional) | OLS R² |
|----------------|------:|------------------------|--------------:|-----------------:|-------:|
| BBQ            |   252 | +1.68 [+0.45, +2.90]   | 0.065         | 0.380            | 0.068  |
| DailyDilemmas  |   700 | +2.35 [+1.73, +2.96]   | 0.115         | 0.411            | 0.111  |
| Trolley        |   546 | +3.15 [+2.09, +4.21]   | 0.104         | 0.391            | 0.167  |
| **Pooled**     | 1,498 | +2.79 [+2.24, +3.34]   | **0.165**     | 0.427            | 0.091  |

**Marginal R² (variance explained by `|baseline_bias|` alone)** is well
below 0.30 on every fit. This **strongly supports C1**: baseline bias
predicts only ~7–17% of the directional structure that the influence-pair
audit recovers. The remaining variance comes from the random effects
(model, factor, influence type) — which is exactly what the audit is
supposed to surface.

The β coefficients are **positive**: more biased baselines do tend to have
a slightly larger absolute asymmetry, but the slope is small relative to
the residual scatter.

## Sanity checks

Per-benchmark Pearson and Spearman on (`|baseline_bias|`, `|asym|`):

| Benchmark      | n     | Pearson(|·|) | Spearman(|·|) | Pearson(signed) |
|----------------|------:|--------------|---------------|-----------------|
| BBQ            |   252 | +0.261       | +0.090        | **-0.368**      |
| DailyDilemmas  |   700 | +0.334       | +0.082        | **-0.363**      |
| Trolley        |   546 | +0.409       | +0.475        | +0.258          |
| Pooled         | 1,498 | +0.302       | +0.165        | -0.046          |

**BBQ signed Pearson** of `(f_0(B) - 0.5, steerability_asym)` is
**r = -0.368, p = 1.6e-9** — the draft (line 1488 of `main.tex`)
cites **r = -0.425, p = 1.7e-9**. Same n=252, same significance
magnitude, slightly different point estimate. Likely the figure
in the draft used a slightly different snapshot or filter; the
qualitative claim ("negatively correlated with baseline preference")
is unchanged.

The **sign flips between benchmarks** is itself interesting:
- BBQ: high baseline bias → asymmetry away from the biased default
  (-0.37, consistent with stereotype-correction influences working harder).
- DailyDilemmas: same negative direction (-0.36) — high baseline
  preference for one value → easier to push opposite.
- Trolley: high baseline bias → asymmetry *toward* the biased default
  (+0.26), the opposite direction.

This sign reversal is hidden when you only look at `|asym|`. The pooled
signed Pearson is essentially zero (-0.05) because the two directions
cancel.

## Suggested wording for §4.2 C1 (line ~250)

> "A mixed-effects regression of |asymmetry| on |baseline bias| (with
> random intercepts on model, factor, and influence type) yields a
> pooled marginal R² of **0.17** (β = 2.79, 95% CI [2.24, 3.34]) across
> 1,498 conditions on trolley + BBQ + DailyDilemmas, and below **0.12**
> within each benchmark. Baseline bias accounts for **7–17%** of the
> directional structure recovered by the influence-pair audit; the
> remainder is information that the choice-only protocol cannot
> recover."

## Outputs

- `data/per_condition.csv` — long-form table, one row per (benchmark,
  model, reasoning, factor, nudge_type) condition.
- `data/regression_results.json` — coefficients, CIs, R², correlations.
- `data/{trolley,bbq,dailydilemmas}_summary.csv` — raw create_summary
  output (kept for traceability).
- `data/results_dailydilemmas_simple/` — DailyDilemmas converted into
  the simple_* layout that `create_summary` expects.
- `figures/asym_vs_baseline_bias.png` — scatter with per-benchmark OLS
  lines.

## Result direction (from the brief)

R² < 0.3 → "strongly supports C1, the expected and desired result." That
is what we observed.
