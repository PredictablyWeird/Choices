# Canonical numbers for the rewrite

Source: `robustness.json` (computed from May-1 per-benchmark summary CSVs and
preference_graph JSONs). Filtered to paper's 9 model configs and 5 trolley
factors; DailyDilemmas excludes "dishonesty" (saturated baseline).

## Headlines: average |effect| (per-condition shift in choice rate)

|                          | Trolley           | BBQ              | DailyDilemmas    |
|--------------------------|-------------------|------------------|------------------|
| **All 7 influences**     | 15.0pp [12.6,17.9]| 17.7pp [15.8,20.1]| 12.3pp [9.9,14.8] |
| Excl. few-shot           | 13.9pp [11.4,17.3]| 17.0pp [15.2,19.2]| 13.2pp [10.6,15.9]|
| Single-sentence user-msg | 13.0pp [10.5,16.7]| 19.3pp [17.5,21.4]| 15.4pp [12.3,18.5]|

CIs are 95% cluster bootstrap over the factor/value axis (n_boot=5000).

**Paper currently says:** 15pp / 18pp / 9pp.
- Trolley 15pp ✓
- BBQ 18pp ≈ rounds from 17.7pp ✓
- DailyDilemmas **9pp is wrong** — true value is 12.3pp (all 7) or 15.4pp
  (single-sentence). Update to 12pp (or 15pp restricted).

## Backfire rate among significant influence effects

| Benchmark    | Paper claim | True (sig-directed denom) | True (cond-with-any-sig denom) |
|--------------|-------------|---------------------------|-------------------------------|
| Trolley      | 14.3%       | 17.7% (84/475)            | 25.2% (72/286)                 |
| BBQ          | 10.5%       | 10.5% ✓                   | 12.1% (25/207)                 |
| DailyDilemmas| 5.9%        | **0.2%** (1/486)          | 0.3%                           |

**The 5.9% claim in the cross-benchmark figure caption is wrong.** The
appendix's 0.2% is correct. Likely the 5.9% was computed under an old
threshold or stale data.

The trolley 14.3% claim drifts from 17.7%: probably an old snapshot. The
"sig-directed denom" matches the figure-caption definition; the
"cond-with-any-sig denom" is what some appendix tables use.

## Baseline-neutral asymmetry rate

(Fraction of baseline-neutral conditions that show significant asymmetry.)

| Benchmark    | Paper claim | binom α=.05 | margin |f0-.5|<.05 | BH-FDR q=.05 |
|--------------|-------------|-------------|----------------------|--------------|
| Trolley      | 39%         | 44.4% (n=135) | 45.8% (n=166)      | 43.0%        |
| BBQ          | 34%         | 39.3% (n=28)  | 35.3% (n=51)       | 28.6%        |
| DailyDilemmas| 17% (unconditional, not baseline-neutral) | 8.8% (n=147) | 10.0% (n=70) | 3.4% |

Paper numbers are within 5pp of binomial-test rates; small drift from
data-snapshot. Equivalence-margin definitions give similar rates (the
"baseline-neutral as absence-of-evidence" concern matters less than ChatGPT
feared). BH-FDR drops BBQ by 11pp and DailyDilemmas by ~6pp; still
substantive.

The DailyDilemmas "17%" in the paper text is the **unconditional** asymmetry
rate, not the baseline-neutral one. Currently inconsistent framing — must
disambiguate.

## Overall asymmetry rates (all conditions, by significance threshold)

| Benchmark    | α=.05  | α=.01  | BH-FDR q=.05 |
|--------------|--------|--------|--------------|
| Trolley      | 62.1%  | 55.5%  | 60.3%        |
| BBQ          | 31.0%  | 19.0%  | 19.0%        |
| DailyDilemmas| 18.5%  | 12.0%  | 10.8%        |

## Sanity / known limitations

- Trolley p-values reconstructed from preference_graph JSONs; agreement
  with CSV's `sig_asym` booleans = 87.1% (418 conditions).
- BBQ same: 90.9% agreement (308 conditions).
- DailyDilemmas p-values are reconstructed from CSV f_*_* columns under
  the assumption of n_comparisons × 3 trials per directed condition; the
  sig_alpha05 rate (18.5%) matches the CSV sig_asym rate exactly, so the
  reconstruction is consistent.

## Paper-side actions needed

1. Update DailyDilemmas headline 9pp → 12pp (all) **with caveat** that
   restricted to single-sentence user-message cues it's 15pp.
2. Update DailyDilemmas backfire 5.9% → 0.2% in cross-benchmark figure;
   reconcile with appendix.
3. Update trolley backfire 14.3% → 18% (or rerun and re-confirm).
4. Update baseline-neutral asymmetry rates: 39% → 44% (trolley),
   34% → 39% (BBQ); add the DailyDilemmas baseline-neutral rate (~9%)
   since it's currently missing.
5. Add BH-FDR sensitivity row to appendix asymmetry table; story still
   holds but stricter q gives 43% / 29% / 3%.
6. Add bootstrap CIs to the cross-benchmark headline figure caption.
