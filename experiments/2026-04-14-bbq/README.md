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

## Question Polarity

BBQ questions come in two polarities:

- **`neg`** — *"Who was forgetful?"* — being chosen is a negative association.
- **`pos`** (nonneg) — *"Who acted responsibly?"* — being chosen is a positive association.

These are run as **separate sub-experiments** using pseudo-factor names
`{category}_neg` and `{category}_pos` (e.g. `age_neg`, `age_pos`).  This
ensures results are stored and analysed independently — a high selection
rate for a group in `age_neg` means the model associates that group with
negative traits, while a high rate in `age_pos` means positive traits.

`max_questions` is applied **per polarity split**, so each subset gets
up to that many questions independently.

## Result Format

Results land in:

```
results/simple_{category}_{polarity}/{model}/{nudge_type}/{timestamp}_{condition}/
    preference_graph_{model}.json
    utility_model_{model}.json
    example_prompt.txt
    summary_{model}.txt
```

For example: `results/simple_age_neg/llama-33-70b/weak_evidence/...`

The JSON files follow the exact schema produced by
`choices/experiments/nudging/simple.py`, so the standard analysis pipeline
(`choices/analysis/create_summary.py`, steerability plots, etc.) can load
them directly.

The factor variable is named `{category}_{polarity}` (e.g. `age_neg` with
values `["nonOld", "old"]`).  A dummy `N=1` variable is included for
structural compatibility; `larger_group_rate` will be `None` since all
comparisons involve individuals.

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
| `expert_recommendation` | end | none |
| `emotional` | end | none |
| `virtue_appeal` | end | none |
| `user_preference` | end | parentheses |
| `always` | end | none |
| `role_play` | system\_replace | none |
| `few_shot` | prepended | none |

### Few-shot nudge

The `few_shot` nudge prepends *k* completed examples (from the same
polarity) where the demonstrated answer always picks the target group.
Option order (A/B) is randomised per example so the model cannot learn a
positional shortcut.  The examples are drawn from questions **not** in the
test set.  Configure *k* with `--few-shot-k` (CLI) or `few_shot_k` in the
batch config (default: 3).
