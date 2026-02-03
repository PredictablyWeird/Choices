# Moral Preferences Under Influence

## Setup

- Clone the repo
- Run `uv sync --dev`
- Install pre-commit hooks with `uv run pre-commit install`
- Add API keys to an '.env' file

## Running Experiments

- Create a batch config file with `uv run python -m choices.experiments.nudging.batch generate-config > config.yaml`
- Modify the batch config
  - Increase `max_requests` for full experiments
  - Change model selection
  - You can set `reasoning` to "before" to instruct models to reason before deciding (applies to all models in the batch)
- Run experiments with `uv run python -m choices.experiments.nudging.batch run --config config.yaml`


## Creating Plots

Assuming you have raw results (as output from the batch script) in the folder `results/`, plots from the main body of the paper are generated as follows:

- Fig. 2: `uv run python -m choices.analysis.plot_model_effects_v2 --factor wealth --results-dirs results --reasoning off none`
- Fig. 3 (a): `uv run python -m choices.analysis.plot_model_effects_v2 --model gpt-5-2-non-reasoning --results-dirs results --reasoning off none`
- Fig. 3 (b): `uv run python -m choices.analysis.plot_model_effects_v2 --model qwen3-235b-a22b-2507 --results-dirs results --reasoning off none`
- Figs. 4 and 5: `uv run python -m choices.analysis.plot_backfiring_steering_wrt_reasoning`
- Fig. 6: `uv run python -m choices.analysis.plot_baseline_vs_bias --results-dirs results --figsize 5 3 --no-title`
- Fig. 7: `uv run python -m choices.analysis.surface_form_analysis --results-dirs results results_surface --groups model --no-show --bar-chart --no-title --figsize 7 4 --abbr-names`
  - Note: For this plot you need to also run "_baseline" and "_negation" versions of experiments and put these results into `results_surface/`

For appendix plots:

- Fig. 8 (a): `uv run python -m choices.analysis.plot_steerability --results-dirs results --rows nudges --reasoning-conditions none off --significance --no-title`
- Fig. 8 (b): `uv run python -m choices.analysis.plot_steerability --results-dirs results --rows nudges --reasoning-conditions before low --significance --no-title`
- Fig. 9 (steerability in dependence of baseline bias): `uv run python -m choices.analysis.plot_steerability_by_baseline --results-dirs results --reasoning-conditions none off low before --sig-baseline-only`
- Fig. 10 (negation results): Use `choices/analysis/analyze_negation_results.py`
- FIgs. 11-18: Script `choices/analysis/plot_classification_results.py`


## Table Data

- Table 1 (statistics by influence type for GPT5.2):
  - `uv run python -m choices.analysis.create_summary  --results-dirs results --models gpt-5-2-non-reasoning`
  - `uv run python -m choices.analysis.create_summary  --results-dirs results --models gpt-5-2-reasoning`
- Table 3-6 (aggregate statistics): `uv run python -m choices.analysis.create_summary  --results-dirs results`
- Table 7 (conditions with maximal steerability asymmetry for GPT5.2): See table output by `uv run python -m choices.analysis.create_summary  --results-dirs results --models gpt-5-2-reasoning gpt-5-2-non-reasoning --sort abs-steer_bias --reverse`
- Table 8 (no-information baseline): TODO
- Table 10 (negation): TODO


## Results From the Paper

In `data/`, you find summary files of our results.
Note that the raw results files are larger and not included in the repo, but using instructions above you can create raw result files of the same format to use for plots.


## Notes on Terminology

Initially we used the term "nudge" to refer to contextual influence, and we used a term "steerability bias" or sometimes "moral steerability bias (MSB)" to refer to steerability asymmetry. In many places, the code and result files currently still use this older terminology.


## Analysis Scripts Not Used in Paper

The following scripts in `choices/analysis/` are **not** used to generate figures or tables in the paper, but are kept for utility or potential future use:

**Individual Result Analysis:**
- `analyze_simple_nudging_results.py` - Detailed analysis of a single simple_nudging experiment (balance, validity, preference stats, steerability asymmetry with bootstrap CIs)
- `analyze_simple_rates.py` - AMCE analysis for a single simple_rates experiment

**Exchange Rate Analysis:**
- `analyze_nudging_results.py` - Exchange rate computation and value steerability analysis
- `create_exchange_rates_plots.py` - Exchange rate visualization utilities

**Data Quality & Diagnostics:**
- `analyze_invalid_responses.py` - Analyze invalid response rates across experiments
- `analyze_classifications.py` - Analyze classification results from reasoning trace classification

**Other Analysis Tools:**
- `get_backfiring_rates.py` - Compute backfiring rates stratified by baseline preference
- `predictive_analysis.py` - Logistic regression analysis of decision factors

**Helper Modules (used by other scripts):**
- `utils.py` - Shared utilities (result loading, balance checking, statistical tests, model colors, etc.)
- `steerability_metric.py` - Steerability asymmetry computation functions
- `nudge_effect_size.py` - Nudge effect size dataclass and helpers
