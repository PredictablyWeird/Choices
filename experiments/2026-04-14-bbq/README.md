# BBQ Nudging Experiments

Adaptation of the simple nudging pipeline to the
[BBQ (Bias Benchmark for QA)](https://arxiv.org/abs/2110.08193) dataset.

Each BBQ question is converted into a pairwise forced choice by dropping the
"unknown" answer option.  Nudges steer the model toward one social group,
and the results are stored in the standard `preference_graph` +
`utility_model` JSON format so the existing analysis code
(`create_summary`, steerability plots, etc.) works without modification.

## Quick Start

```bash
# Single experiment (age category, weak_evidence nudge)
uv run python experiments/2026-04-14-bbq/run.py \
    --category age --nudge weak_evidence --model gpt-4o-mini

# Batch run from config file
uv run python experiments/2026-04-14-bbq/run.py \
    --config experiments/2026-04-14-bbq/batch_config.yaml

# List what's available
uv run python experiments/2026-04-14-bbq/run.py --list-categories
uv run python experiments/2026-04-14-bbq/run.py --list-nudges
```

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--category` | *(required)* | BBQ category (e.g. `age`, `gender`, `ethnicity`) |
| `--nudge` | *(required)* | Nudge type (e.g. `weak_evidence`, `emotional`) |
| `--model` | `gpt-4o-mini` | Model key from `choices/config/models.yaml` |
| `--context-condition` | `ambig` | `ambig`, `disambig`, or `both` |
| `--max-questions` | `50` | Max BBQ questions to sample per category |
| `--requests-per-edge` | `4` | Repeated API calls per pairwise comparison |
| `--seed` | `42` | Random seed for sampling |
| `--reasoning` | `none` | `none`, `before`, or `after` |
| `--target-group` | *(all)* | Run only one condition (`base` or a group tag) |
| `--save-dir` | `experiments/2026-04-14-bbq/results` | Output directory |
| `--data-dir` | `experiments/2026-04-14-bbq/data` | BBQ JSONL cache |
| `--config` | — | Path to a YAML batch config (overrides other flags) |

## Context Conditions

BBQ provides two versions of each scenario:

- **`ambig`** (default) — The context is ambiguous and does not support either
  answer.  Any preference the model shows reveals latent bias.  This is the
  natural analog to the original "who to save" experiment, where there is no
  objectively correct answer.
- **`disambig`** — The context includes information that makes one answer
  clearly correct.  This tests whether nudges can override factual knowledge.
- **`both`** — Uses both context types.

## Result Format

Results land in:

```
results/simple_{category}/{model}/{nudge_type}/{timestamp}_{condition}/
    preference_graph_{model}.json
    utility_model_{model}.json
    example_prompt.txt
    summary_{model}.txt
```

The JSON files follow the exact schema produced by
`choices/experiments/nudging/simple.py`, so the standard analysis pipeline
(`choices/analysis/create_summary.py`, steerability plots, etc.) can load
them directly.

The factor variable is named after the BBQ category (e.g. `age` with values
`["nonOld", "old"]`).  A dummy `N=1` variable is included for structural
compatibility; `larger_group_rate` will be `None` since all comparisons
involve individuals.

## BBQ Dataset

The BBQ JSONL files are downloaded automatically on first run into the
`data/` subdirectory.  The 6 supported (binary) categories are:

age, disability, gender, nationality, appearance, ses.

Three categories with more than two social groups — ethnicity (6),
religion (8), sexuality (4) — are **not currently supported** because
the nudge text targets a single group that may not appear in every
question.  Future work could handle these by filtering questions per
nudge target or generating per-question nudge text.

Two intersectional categories (race\_x\_ses, race\_x\_gender) are also
not included in the default batch config.

## Nudge Types

BBQ-adapted versions of the standard nudge families:

| Nudge | Position | Brackets |
|---|---|---|
| `survey_preference` | start | none |
| `weak_evidence` | end | parentheses |
| `strong_evidence` | start | none |
| `expert_recommendation` | end | none |
| `emotional` | end | none |
| `virtue_appeal` | end | none |
| `user_preference` | end | parentheses |
| `social_norm` | end | none |
| `always` | end | none |
| `role_play` | system\_replace | none |
