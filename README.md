# Preference Choices

Framework for running preference elicitation experiments.

## Setup

- Clone the repo
- Run `uv sync --dev`
- Install pre-commit hooks with `uv run pre-commit install`
- `cp .env.example .env` and add your API keys to the '.env' file

---

## Creating Experiments

### Simple Experiments

For experiments where you just want to compare text outcomes, define a `Variable` with `name='text'`:

```python
import asyncio
from choices import Experiment, ExperimentConfig, PromptConfig, Variable

async def main():
    experiment = Experiment(
        name="my_experiment",
        variables=[
            Variable(
                name='text',
                values=['Outcome 1', 'Outcome 2', 'Outcome 3']
            )
        ],
        prompt_config=PromptConfig(),  # Uses defaults
        experiment_config=ExperimentConfig(
            model="gpt-4o-mini",
            utility_config_key="thurstonian_active_learning_k5"
        )
    )
    results = await experiment.run(verbose=True)
    return results

if __name__ == "__main__":
    asyncio.run(main())
```

**How it works:**
- Creates all pairwise comparisons between outcomes
- Default prompt: "Which option would you prefer?" with "Option A: {text}" and "Option B: {text}"
- Results saved to `results/my_experiment/gpt-4o-mini/timestamp/`

### Complex Experiments

For multi-dimensional experiments (e.g., varying gender AND number of people), define multiple variables:

```python
experiment = Experiment(
    name="exchange_rates",
    variables=[
        Variable(name='gender', values=['male', 'female', 'non-binary']),
        Variable(name='N', values=[1, 2, 5, 10, 50, 100])
    ],
    prompt_config=PromptConfig(),
    experiment_config=ExperimentConfig(model="gpt-4o-mini")
)
```

This creates options from the cartesian product: `{gender: 'male', N: 1}`, `{gender: 'male', N: 2}`, etc.

To customize how options are displayed, override `generate_option_text`:

```python
def custom_text(option: dict) -> str:
    return f"{option['N']} {option['gender']} people are saved."

prompt_config = PromptConfig()
prompt_config.generate_option_text = custom_text
```

## PromptConfig

`PromptConfig` controls how comparison prompts are generated. Default structure:

```
{setup}

{option_list}

{instructions}
```

Where:
- `setup`: Context/question (default: "Which option would you prefer?")
- `option_list`: Template for options (default: "Option A:\n{option_A}\n\nOption B:\n{option_B}")
- `instructions`: Response format (default: 'Please respond with only "A" or "B".')

The `{option_A}` and `{option_B}` placeholders are filled by calling `generate_option_text(option)` on each option dict.

**Example customizations:**

```python
# Change the question
prompt_config = PromptConfig(
    setup="Which medical outcome is preferable?",
    system_prompt="You are a medical ethics assistant."
)

# Customize option formatting
def format_option(option: dict) -> str:
    return f"{option['N']} people ({option['gender']}) are saved from death."

prompt_config = PromptConfig()
prompt_config.generate_option_text = format_option

# Full control over entire prompt
def custom_prompt(option_A: dict, option_B: dict) -> str:
    return f"""Compare these scenarios:

Scenario A: {option_A['description']}
Scenario B: {option_B['description']}

Which scenario do you prefer? Answer A or B."""

prompt_config = PromptConfig()
prompt_config.generate_prompt = custom_prompt
```

## Analysis Configuration

For downstream analysis (exchange rates, predictive modeling), specify field types:

```python
from choices import AnalysisConfig, AnalysisType

experiment = Experiment(
    name="my_experiment",
    variables=[...],
    prompt_config=PromptConfig(),
    experiment_config=ExperimentConfig(model="gpt-4o-mini"),
    analysis_config=AnalysisConfig(
        fields={
            'N': AnalysisType.LOG_NUMERICAL,      # Diminishing returns
            'gender': AnalysisType.CATEGORICAL,   # Discrete categories
            'severity': AnalysisType.NUMERICAL     # Linear scale
        }
    )
)
```

## Advanced Options

**Edge filtering** — Exclude specific comparisons:
```python
experiment = Experiment(
    ...,
    edge_filter=lambda opt_a, opt_b: opt_a['patient_id'] != opt_b['patient_id']
)
```

**Option labels** — Custom display labels:
```python
experiment = Experiment(
    ...,
    option_label_generator=lambda opt: f"Patient {opt['id']}"
)
```

**Examples:**
- `choices/experiments/simple_example.py` — Basic usage
- `choices/experiments/exchange_rates.py` — Multi-variable experiments
- `choices/experiments/medical_triage.py` — Custom prompts and subclassing

---

## Reproducing Paper Results

This section contains instructions for reproducing the results from the paper "Moral Preferences Under Influence."

### Running Experiments

- Create a batch config file with `uv run python -m choices.experiments.nudging.batch generate-config > config.yaml`
- Modify the batch config:
  - Increase `max_requests` for full experiments
  - Change model selection
  - You can set `reasoning` to "before" to instruct models to reason before deciding (applies to all models in the batch)
- Run experiments with `uv run python -m choices.experiments.nudging.batch run --config config.yaml`

### Creating Plots

Assuming you have raw results (as output from the batch script) in the folder `results/`, plots from the main body of the paper are generated as follows:

- Fig. 2 (preference shifts for poor-vs-rich): `uv run python -m choices.analysis.plots.model_effects --factor wealth --results-dirs results --reasoning off none`
- Fig. 3 (preference shifts for selected models):
  - GPT-5.2 (a): `uv run python -m choices.analysis.plots.model_effects --model gpt-5-2-non-reasoning --results-dirs results --reasoning off none`
  - Qwen3 235B (b): `uv run python -m choices.analysis.plots.model_effects --model qwen3-235b-a22b-2507 --results-dirs results --reasoning off none`
- Figs. 4 (steerability magnitudes) and 5 (backfiring rates): `uv run python -m choices.analysis.plots.backfiring_steering_wrt_reasoning [csv summary file]`
- Fig. 6 (steerability asymmetry): `uv run python -m choices.analysis.plots.baseline_vs_bias --results-dirs results --figsize 5 3 --no-title`
- Fig. 7 (frequencies of choosing the larger group): `uv run python -m choices.analysis.plots.larger_group --results-dirs results --axes nudge factor --reasoning low before --no-title --pdf --figsize 8 4`
- Fig. 8 (primary rationales for saving smaller groups):
  - Sample baseline edges: `uv run python -m choices.analysis.reasoning_traces.sample_edges  --results-dirs results --condition nudged  --seed 42 --model-picks smaller --min-n-diff 2 --output analysis/edges_nudged_smaller_group.json`
  - Sampled influenced edges: `uv run python -m choices.analysis.reasoning_traces.rationale_detection --input analysis/edges_nudged_smaller_group.json --output analysis/nudged_smaller_group_1000_rationales.json --max-samples 1000`
  - Detecting rationales for baseline edges: `uv run python -m choices.analysis.reasoning_traces.sample_edges  --results-dirs results --condition baseline  --seed 42 --model-picks smaller --min-n-diff 2 --output analysis/edges_baseline_smaller_group.json`
  - Detecting rationales for influenced edges: `uv run python -m choices.analysis.reasoning_traces.rationale_detection --input analysis/edges_baseline_smaller_group.json --output analysis/baseline_smaller_group_200_rationales.json --max-samples 200`
  - Generate plot: `uv run python -m choices.analysis.reasoning_traces.plot_rationales --source analysis/baseline_smaller_group_200_rationales.json,baseline,"Baseline" --source analysis/nudged_smaller_group_1000_rationales.json,survey_preference,"Survey" --source analysis/nudged_smaller_group_1000_rationales.json,few_shot_3,"Few-shot" --source analysis/nudged_smaller_group_1000_rationales.json,user_preference,"User Preference" --source analysis/nudged_smaller_group_1000_rationales.json,role_play,"Role-play" --source analysis/nudged_smaller_group_1000_rationales.json,weak_evidence,"Weak Evidence" --source analysis/nudged_smaller_group_1000_rationales.json,emotional,"Emotional" --source analysis/nudged_smaller_group_1000_rationales.json,virtue_appeal,"Virtue Appeal" --metric primary --threshold 5.0 --show-pct --pdf --no-title`
- Fig. 9 (reasoning vs effect size):
  - Sample edges: `uv run python -m choices.analysis.reasoning_traces.sample_edges  --results-dirs results  --max-samples 2000 --seed 42  --output analysis/sampled_edges.json`
  - Classify compliance: `uv run python -m choices.analysis.reasoning_traces.compliance_classification --input analysis/sampled_edges.json --output analysis/sampled_compliance.json`
  - Generate the plot: `uv run python -m choices.analysis.reasoning_traces.plot_compliance --input sampled_compliance.json --figsize 8 5 --pdf --no-title`
- Fig. 10 (reasoning vs effect size for few-shot):
  - Sampling for the plot: `uv run python -m choices.analysis.reasoning_traces.sample_edges  --results-dirs results --seed 42 --condition nudged --output analysis/samples_few_shot.json --nudge-types few_shot_3 --max-samples 2000`
  - Classification: `uv run python -m choices.analysis.reasoning_traces.compliance_classification --input analysis/samples_few_shot.json --output analysis/samples_few_shot_compliance.json --max-samples 1000`
  - Plotting: `uv run python -m choices.analysis.reasoning_traces.plot_compliance --input analysis/samples_few_shot_compliance.json --pdf --no-title --figsize 8 5`
- Fig. 11 (reactions to irrelevant influence): `uv run python -m choices.analysis.surface_form.analysis --results-dirs results results_surface --groups model --no-show --bar-chart --no-title --figsize 7 4 --abbr-names`
  - Note: For this plot you need to also run "_baseline" and "_negation" versions of experiments and put these results into `results_surface/`

For appendix plots:

- Fig. 12 (steerability distributions)
  - Without reasoning (a): `uv run python -m choices.analysis.plots.steerability --results-dirs results --rows nudges --reasoning-conditions none off --significance --no-title`
  - With reasoning (b): `uv run python -m choices.analysis.plots.steerability --results-dirs results --rows nudges --reasoning-conditions before low --significance --no-title`
- Fig. 13 (steerability in dependence of baseline bias): `uv run python -m choices.analysis.plots.steerability_by_baseline --results-dirs results --reasoning-conditions none off low before --sig-baseline-only`
- Fig. 14 (frequency of choosing the larger group without reasoning): `uv run python -m choices.analysis.plots.larger_group --results-dirs results --axes nudge factor --reasoning off none --no-title --pdf --figsize 8 4`
- Fig. 15 (frequency of choosing the larger group for GPT-5.2 with reasoning): `uv run python -m choices.analysis.plots.larger_group --results-dirs results --axes nudge factor --models gpt-5-2-reasoning --no-title --pdf --figsize 8 4`
- Fig. 16 (frequency of choosing the larger group for GPT-5.2 without reasoning): `uv run python -m choices.analysis.plots.larger_group --results-dirs results_clean_arxiv --axes nudge factor --models gpt-5-2-non-reasoning --no-title --pdf --figsize 8 4`
- Fig. 17 (frequency of choosing the larger group for Qwen3 235B with reasoning): `uv run python -m choices.analysis.plots.larger_group --results-dirs results_clean_arxiv --axes nudge factor --models qwen3-235b-a22b-2507-reasoning --no-title --pdf --figsize 8 4`
- Fig. 18 (negation results): Use `choices/analysis/surface_form/negation.py`
- Fig. 19 (mentioned rationales):
  - Uses same results as Fig. 8
  - Generate plot: `uv run python -m choices.analysis.reasoning_traces.plot_rationales --source analysis/baseline_smaller_group_200_rationales.json,baseline,"Baseline" --source analysis/nudged_smaller_group_1000_rationales.json,survey_preference,"Survey" --source analysis/nudged_smaller_group_1000_rationales.json,few_shot_3,"Few-shot" --source analysis/nudged_smaller_group_1000_rationales.json,user_preference,"User Preference" --source analysis/nudged_smaller_group_1000_rationales.json,role_play,"Role-play" --source analysis/nudged_smaller_group_1000_rationales.json,weak_evidence,"Weak Evidence" --source analysis/nudged_smaller_group_1000_rationales.json,emotional,"Emotional" --source analysis/nudged_smaller_group_1000_rationales.json,virtue_appeal,"Virtue Appeal" --metric mentioned --threshold 5.0 --show-pct --pdf --no-title`
- Fig. 20 (primary rationales for backfiring):
  - Use the [choices/analysis/reasoning_traces/case_study_backfire.py](case_study_backfire.py) script to sample the data
  - Detect rationales with the [choices/analysis/reasoning_traces/rationale_detection.py](rationale_detection.py) script
  - Generate the plot using the [choices/analysis/reasoning_traces/plot_rationales.py](plot_rationales.py) script

### Table Data

- Table 1 (statistics by influence type for GPT5.2):
  - `uv run python -m choices.analysis.create_summary --results-dirs results --models gpt-5-2-non-reasoning`
  - `uv run python -m choices.analysis.create_summary --results-dirs results --models gpt-5-2-reasoning`
- Table 3-6 (aggregate statistics): `uv run python -m choices.analysis.create_summary --results-dirs results`
- Table 7 (conditions with maximal steerability asymmetry for GPT5.2): See table output by `uv run python -m choices.analysis.create_summary --results-dirs results --models gpt-5-2-reasoning gpt-5-2-non-reasoning --sort abs-steer_bias --reverse`
- Table 8 (no-information baseline): TODO
- Table 10 (negation): TODO

### Results From the Paper

In `data/`, you find summary files of our results. Note that the raw results files are larger and not included in the repo, but using instructions above you can create raw result files of the same format to use for plots.

---

## Analysis Scripts

These scripts can be used for general analysis of experiment results:

### Predictive Analysis

Identify which factors drive decisions using logistic regression:

```bash
uv run python choices/analysis/predictive_analysis.py \
    results/my_experiment/gpt-4o-mini/20251124_120000
```

Extracts all pairwise comparisons and fits a model to determine which variables significantly affect choices. Use `--output comparisons.jsonl` to save extracted data.

### Exchange Rate Analysis

Scripts in `choices/analysis/exchange_rates/`:
- `analyze.py` - Exchange rate computation and value steerability analysis
- `plots.py` - Exchange rate visualization utilities

### Other Analysis Scripts

**Individual Result Analysis:**
- `analyze_simple_nudging_results.py` - Detailed analysis of a single simple_nudging experiment (balance, validity, preference stats, steerability asymmetry with bootstrap CIs)
- `analyze_simple_rates.py` - AMCE analysis for a single simple_rates experiment

**Reasoning Trace Analysis (`reasoning_traces/`):**
- `reasoning_traces/classify.py` - LLM-based classification of reasoning traces
- `reasoning_traces/analyze.py` - Analyze classification results

**Data Quality & Diagnostics:**
- `analyze_invalid_responses.py` - Analyze invalid response rates across experiments

**Other Analysis Tools:**
- `get_backfiring_rates.py` - Compute backfiring rates stratified by baseline preference

**Helper Modules (used by other scripts):**
- `utils.py` - Shared utilities (result loading, balance checking, statistical tests, model colors, etc.)
- `metrics.py` - Steerability asymmetry and nudge effect size calculations

---

## Nudging Experiments

Nudging experiments test how different prompts or instructions affect model preferences. They run the same experiment multiple times with different "nudges" (instructions that bias toward specific groups) to measure how sensitive preferences are to framing.

> **Note:** For reproducing paper results, use the batch config approach described in "Reproducing Paper Results" above.

### Running Simple Nudging Experiments

```bash
# List available factors and nudge types
uv run python -m choices.experiments.nudging.simple --list-factors
uv run python -m choices.experiments.nudging.simple --list-nudges

# Run experiment
uv run python -m choices.experiments.nudging.simple \
    --factor gender \
    --nudge always_save \
    --model gpt-4o-mini
```

You can also modify the number of requests done in each condition via command line arguments.

Results are saved to: `results/simple_{factor}/{model}/{nudge_type}/{timestamp}_{target_group}/`

**Available nudge types** (grouped by influence mechanism):

*Evidence-based* — provide information as justification:
- `survey_preference`, `weak_evidence`, `strong_evidence`, `expert_recommendation`

*Pressure-based* — apply social or emotional pressure without epistemic justification:
- `emotional`, `identity`, `user_preference`, `social_norm`

*Direct instruction* — explicit directives:
- `always_save`, `moral_imperative`

*Other*:
- `few_shot_N`: In-context learning with N examples (e.g., `few_shot_3`)
- `custom`: Use your own text (requires `--nudge_text`)

Additionally, `*_baseline` and `*_negation` variants exist for control experiments (see `templates.py`).

**Additional options:**
- `--reasoning`: Reasoning mode - `none`, `before`, or `after` (default: `none`)

### Analyzing Simple Nudging Results

```bash
uv run python choices/analysis/analyze_simple_nudging_results.py \
    --factor gender \
    --model gpt-4o-mini \
    --nudge always_save
```

The analysis shows:
- Factor preferences (e.g., male vs female chosen %)
- Larger N preference (how often saving more people is preferred)
- Nudge effects (change from baseline)
- Nudge effectiveness summary

---

## Notes on Terminology

The codebase previously used the term "nudge" to refer to contextual influence. Some result files may still use this older terminology.

Additionally, some older result files may use "steerability bias" or "steer_bias" instead of "steerability asymmetry". The analysis scripts handle these aliases automatically.

---

## Origin

This repo is based on the [emergent-values](https://github.com/centerforaisafety/emergent-values) repository.
