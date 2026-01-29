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


## Results From the Paper

In `data/`, you find summary files of our results.
Note that the raw results files are larger and not included in the repo, but using instructions above you can create raw result files of the same format to use for plots.


## Notes on Terminology

Initially we used the term "nudge" to refer to contextual influence, and we used a term "steerability bias" or sometimes "moral steerability bias (MSB)" to refer to steerability asymmetry. In many places, the code and result files currently still use this older terminology.