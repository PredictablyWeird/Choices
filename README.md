# Direction-Flipped Influence Audits — code and reproduction pipeline

Code accompanying the NeurIPS 2026 submission *Direction-Flipped Influence
Audits Reveal Hidden Structure in LLM Moral-Choice Benchmarks*.

This repo contains:

- **The influence-pair audit harness** (`choices/`) — a general framework for
  running paired direction-flipped contextual-influence experiments against
  LLM APIs.
- **Per-benchmark adapters** (`choices/experiments/nudging/`,
  `experiments/bbq/`, `experiments/dailydilemmas/`)
  for the three benchmarks reported in the paper: a moral triage task, BBQ,
  and DailyDilemmas.
- **A single canonical-numbers pipeline**
  (`pipeline/produce_paper_artifacts.py`)
  that recomputes every numeric claim cited by the paper — headline shifts,
  asymmetry rates with cluster-bootstrap CIs, BH-FDR-corrected sensitivity
  tables, mixed-effects regression, follow-up-probe headline numbers — and
  emits one JSON the paper LaTeX reads from.

If you're reviewing the paper or trying to reproduce a specific number,
**start with [Reproducing paper numbers](#reproducing-paper-numbers)**.
If you're trying to extend the audit to a new benchmark, see
[Running new experiments](#running-new-experiments).

---

## Setup

Requirements: Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) for
dependency management.

```bash
git clone <this-repo>
cd Choices
uv sync --dev
cp .env.example .env   # add your API keys (OpenAI, Anthropic, etc.)
```

The reproduction pipeline declares dependencies inline (PEP 723), so
individual scripts can also be run with `uv run <script.py>` without a full
`uv sync`.

---

## Reproducing paper numbers

Every numeric claim cited in the paper is recorded in
`pipeline/data/paper_numbers.json`, which is committed to the repo and
read directly by the LaTeX source. The accompanying script
`pipeline/produce_paper_artifacts.py` is the canonical generator: it
loads the per-condition summary CSVs, applies the paper's scope
filters, computes cluster-bootstrap CIs and BH-FDR-corrected
sensitivity tables, and writes the JSON plus a first-pass headline
figure.

If you only want to look up a specific number, read `paper_numbers.json`
directly — no setup required. If you want to verify the JSON matches
what the script produces, re-run:

```bash
uv run pipeline/produce_paper_artifacts.py
```

First run takes ~1–2 minutes (extracting Wald p-values from raw
preference graphs); subsequent runs use the cached
`pvalues_by_condition.csv` and finish in seconds. The script prints a
summary of headline numbers; cross-check against the paper.

### Where each number in the paper comes from

Every numeric claim in the paper has an entry in `paper_numbers.json`.
Table mapping the most-cited numbers to their JSON keys:

| Paper claim | JSON path |
|---|---|
| 15.0/17.7/12.3pp shifts (§4.1) | `headline_table.{trolley,bbq,dailydilemmas}.all_7_influences.mean_abs_effect` |
| 13.0/19.3/15.4pp single-sentence shifts | `headline_table.*.single_sentence_user_msg_only.mean_abs_effect` |
| 95% cluster-bootstrap CIs | `headline_table.*.<subset>.boot95_{lo,hi}` (subsets: `all_7_influences`, `excl_few_shot`, `single_sentence_any`, `single_sentence_user_msg_only`) |
| 44/39/9% baseline-neutral asymmetry rates (§4.2) | `section_4_2_asymmetry.rates_by_benchmark.*.by_neutrality.binom_05.rate_alpha05` |
| BH-FDR-corrected variants of the above | same path, `rate_bh_fdr05` |
| Equivalence-margin sensitivity (Tab. 24) | `section_4_2_asymmetry.rates_by_benchmark.*.by_neutrality` |
| 17.7/10.5/0.2% backfire rates (§4.3) | `section_4_3_backfire.definitions.*.rate_sig_bf_among_sig_directed` |
| 13.4/13.1/26.1% backfire-by-baseline (§B.2) | `section_4_3_backfire.definitions.trolley.rate_*` cells |
| Mixed-effects regression β, R² (§4.2) | `experiments/asymmetry-regression/data/regression_results.json` (also surfaced in `paper_numbers.json` under `section_4_2_asymmetry.regression.regressions`) |
| Per-benchmark signed correlations (−0.37 / −0.36 / +0.26) | `section_4_2_asymmetry.regression.correlations` |
| 78% disclaim rate (§5) | `section_4_3_backfire.followup_probe_headline.overall.headline_ack_disclaimed_in_backfires` |
| Per-benchmark baseline noise floor (1.1/1.7/2.5pp) | `section_4_1_instability.noise_floor_pp` |
| Reasoning-trace primary-rationale shifts (§K.4) | `section_5_reasoning_traces.{bbq,dailydilemmas}.rationale_distributions` |

`PAPER_NUMBERS.md` in the same directory lists each claim with the
canonical value in human-readable form for easy diffing.

### Reproducing figures

The figures in the paper are produced by these scripts:

| Paper figure | Generation script |
|---|---|
| Fig. 1 (intro illustration, "asymmetric compliance") | hand-prepared diagram (`moral-steerability-paper/figures/moral-steerability-icml-4.pdf`) |
| Fig. 2 (one-sentence shift wrapfigure, §3.3) | `experiments/baseline-noise/build_one_sentence_fig.py` |
| Fig. 3 (cross-benchmark headline) | `_make_cross_benchmark_figure` inside `produce_paper_artifacts.py` |
| Stated-vs-revealed (§5) | `experiments/followup-probe/` (analysis dir) |
| Per-influence-type steerability magnitudes | `choices/analysis/plots/` |
| Per-model violins | `choices/analysis/plots/` |
| Reasoning-trace rationale plots (per-benchmark DeepSeek figures) | `choices/analysis/reasoning_traces/plot_rationales.py`; pre-rendered DeepSeek PDFs are also surfaced into the canonical-pipeline `figures/` dir |

Each script writes its outputs to a stable path under
`moral-steerability-paper/figures/` (or its local `figures/` subdir for
the canonical pipeline).

### Inputs / data layout

The pipeline reads from three places:

1. **Per-benchmark summary CSVs** (tracked, ~500 KB total):
   `experiments/asymmetry-regression/data/{trolley,bbq,dailydilemmas}_summary.csv`.
   These are produced by `choices.analysis.create_summary` from raw
   preference-graph JSONs and are the canonical per-condition source.
2. **Raw preference-graph JSONs** (~3 GB, not in git): the source
   logs `choices.analysis.create_summary` reads to produce (1). The
   raw response logs are not distributed during peer review; the
   tracked summary CSVs and the cached
   `pipeline/data/pvalues_by_condition.csv` are sufficient to
   reproduce every number cited in the paper.
3. **Pre-computed sibling outputs** (tracked):
   `experiments/{baseline-noise,followup-probe}/` — pulled in
   by the canonical pipeline without recomputation.

### Scope filters applied by the pipeline

The paper restricts to a specific subset of the data in `google_drive/`.
The pipeline applies these filters automatically:

- **Models**: 9 configurations (`deepseek-v3-2-{,non-}reasoning`,
  `gpt-5-2-{,non-}reasoning`, `grok-41-fast-{,non-}reasoning`,
  `llama-33-70b` ± CoT prompting, `qwen3-235b-a22b-2507-{,reasoning}`).
  `gpt-4o-mini` and `llama-4-maverick` exist in the source CSVs but are
  filtered out — the paper only cites these 9.
- **Triage factors**: 5 binary factors (gender, age, wealth, handedness,
  nationality). `diet` and `tech_view` exist in the source CSVs but are
  filtered out.
- **DailyDilemmas**: excludes `dishonesty` (saturated baseline; the
  paper notes this exclusion in App. E).

All filters are at the top of `produce_paper_artifacts.py` (search for
`PAPER_MODELS`, `PAPER_FACTORS_TROLLEY`, `dishonesty`).

---

## Running new experiments

The framework supports paired direction-flipped audits on arbitrary
binary forced-choice benchmarks. The three paper benchmarks are
implemented as adapters; adding a fourth follows the same pattern.

### Triage harness (custom binary scenarios)

```bash
# List available factors and influence types
uv run python -m choices.experiments.nudging.simple --list-factors
uv run python -m choices.experiments.nudging.simple --list-nudges

# Run one cell (e.g., gender × emotional × DeepSeek)
uv run python -m choices.experiments.nudging.simple \
    --factor gender \
    --nudge emotional \
    --model deepseek-v3-2-non-reasoning
```

Results are saved to
`results/simple_<factor>/<model>/<nudge>/<timestamp>_<target>/`. To
reproduce the paper's full triage matrix, sweep over all (factor ×
nudge × model × direction) combinations.

### BBQ adapter

See `experiments/bbq/`. The adapter drops the `unknown`
answer to make BBQ pairwise, restricts to the binary `Age` and `SES`
categories, and applies group-level direction-flipped influences.

### DailyDilemmas adapter

See `experiments/dailydilemmas/`. The adapter
applies value-level (rather than group-level) direction-flipped
influences: each option's primary value becomes the target.

### Available influence types

Implemented in `choices/experiments/nudging/templates.py`:

- **Social-pressure**: `emotional`, `user_preference`, `identity` (role-play),
  `social_norm`
- **Evidential**: `survey_preference`, `weak_evidence`,
  `expert_recommendation`, `strong_evidence`
- **Few-shot**: `few_shot_N` (default `few_shot_3` in the paper)
- **Normative / virtue framing**: `moral_imperative` (the
  "helpful-and-thoughtful AI assistant" template)
- **Direct instruction**: `always_save`
- Per-influence `*_baseline` and `*_negation` variants for surface-form
  controls

The paper reports seven of these (`emotional`, `survey_preference`,
`user_preference`, `weak_evidence`, `moral_imperative`, `identity`,
`few_shot_3`).

---

## Repository layout

```
choices/                         framework code (used by all experiments)
  experiment.py                  core Experiment class
  experiments/                   benchmark adapters + example scripts
    nudging/                       triage harness + influence templates
  analysis/                      analysis library
    create_summary.py              raw-results → per-condition summary CSV
    metrics.py                     steerability, asymmetry, backfire
    reasoning_traces/              compliance + rationale classification
    surface_form/                  surface-form / negation / nonsensical
    plots/                         per-influence-type / per-model figures
    get_backfiring_rates.py        backfire-by-baseline-bias
    analyze_invalid_responses.py   invalid-response rates

pipeline/                        *** master pipeline — start here ***
  produce_paper_artifacts.py     entry point (uv run pipeline/produce_paper_artifacts.py)
  data/paper_numbers.json        every paper-citable number, by section
  data/pvalues_by_condition.csv  Wald p-values per condition (used for FDR)
  figures/cross_benchmark.{png,pdf}   Fig. 3 in the paper
  PAPER_NUMBERS.md               human-readable diff against the previous draft

experiments/                     per-benchmark + supporting experiment runners
  bbq/                             BBQ raw experiment
  dailydilemmas/                   DailyDilemmas raw experiment
  baseline-noise/                  baseline-to-baseline replicate (1.1/1.7pp)
                                   + Fig. 2 generation script
  followup-probe/                  follow-up disclaim/affect probe (78% headline)
  asymmetry-regression/            mixed-effects regression

google_drive/                    raw preference-graph JSONs (not in git;
                                  not distributed during peer review —
                                  the tracked summary CSVs are sufficient
                                  to reproduce every paper number)

tests/                           unit tests
```

---

## Citing

(citation block to be added on de-anonymization)

---

## License

MIT (see `LICENSE`).

This codebase is built on
[emergent-values](https://github.com/centerforaisafety/emergent-values).
