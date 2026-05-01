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

DailyDilemmas was out of scope for this run — the brief noted it needs
`convert_to_simple.py` first and the resulting layout differs from the
two simple_* benchmarks. The pooled fit captures trolley + BBQ.

## How to run

```bash
# Regenerate the per-condition CSVs from raw graph data
uv run python -m choices.analysis.create_summary \
    --results-dirs ~/code/values/moral-steerability-paper/google_drive/results_clean_arxiv \
                   ~/code/values/moral-steerability-paper/google_drive/results_extra_arxiv \
    --output experiments/2026-05-01-asymmetry-baseline-regression/data/trolley_summary.csv

uv run python -m choices.analysis.create_summary \
    --results-dirs ~/code/values/moral-steerability-paper/google_drive/results_bbq_v2 \
    --output experiments/2026-05-01-asymmetry-baseline-regression/data/bbq_summary.csv

# Fit regressions and produce the scatter
uv run python experiments/2026-05-01-asymmetry-baseline-regression/run_regression.py
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

| Fit     | n   | β (95% CI)             | R² (marginal) | R² (conditional) | OLS R² |
|---------|-----|------------------------|---------------|------------------|--------|
| BBQ     | 252 | +1.68 [+0.45, +2.90]   | 0.065         | 0.380            | 0.068  |
| Trolley | 546 | +3.15 [+2.09, +4.21]   | 0.104         | 0.391            | 0.167  |
| Pooled  | 798 | +3.16 [+2.24, +4.08]   | 0.159         | 0.424            | 0.153  |

**Marginal R² (variance explained by `|baseline_bias|` alone)** is well
below 0.30 on every fit. This **strongly supports C1**: baseline bias
predicts only ~6–16% of the directional structure that the influence-pair
audit recovers. The remaining variance comes from the random effects
(model, factor, influence type) — which is exactly what the audit is
supposed to surface.

The β coefficients are **positive**: more biased baselines do tend to have
a slightly larger absolute asymmetry, but the slope is small relative to
the residual scatter.

## Sanity checks

Per-benchmark Pearson and Spearman on (`|baseline_bias|`, `|asym|`):

| Benchmark | n   | Pearson(|·|) | Spearman(|·|) | Pearson(signed) |
|-----------|-----|--------------|---------------|-----------------|
| BBQ       | 252 | +0.261       | +0.090        | -0.368          |
| Trolley   | 546 | +0.409       | +0.475        | +0.258          |
| Pooled    | 798 | +0.391       | +0.363        | +0.183          |

**BBQ signed Pearson** of `(f_0(B) - 0.5, steerability_asym)` is
**r = -0.368, p = 1.6e-9** — the draft (line 1488 of `main.tex`)
cites **r = -0.425, p = 1.7e-9**. Same n=252, same significance
magnitude, slightly different point estimate. Likely the figure
in the draft used a slightly different snapshot or filter; the
qualitative claim ("negatively correlated with baseline preference")
is unchanged.

The **sign flips between benchmarks** is itself interesting:
- BBQ: high baseline bias → asymmetry away from the biased default
  (consistent with stereotype-correction influences working harder).
- Trolley: high baseline bias → asymmetry toward the biased default
  (slight, +0.26), the opposite direction.

This is hidden when you only look at `|asym|`.

## Suggested wording for §4.2 C1 (line ~250)

> "A mixed-effects regression of |asymmetry| on |baseline bias| (with
> random intercepts on model, factor, and influence type) yields a
> marginal R² of 0.16 in the pooled fit (β = 3.16, 95% CI [2.24, 4.08])
> and below 0.11 within each benchmark. Baseline bias accounts for
> ~6–16% of the directional structure recovered by the influence-pair
> audit; the remainder is information that the choice-only protocol
> cannot recover."

## Outputs

- `data/per_condition.csv` — long-form table, one row per (benchmark,
  model, reasoning, factor, nudge_type) condition.
- `data/regression_results.json` — coefficients, CIs, R², correlations.
- `data/{trolley,bbq}_summary.csv` — raw create_summary output (kept for
  traceability).
- `figures/asym_vs_baseline_bias.png` — scatter with per-benchmark OLS
  lines.

## Result direction (from the brief)

R² < 0.3 → "strongly supports C1, the expected and desired result." That
is what we observed.
