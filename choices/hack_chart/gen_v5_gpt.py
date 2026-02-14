"""Generate v5 chart for GPT-5.2 with survey_preference first ordering for all factors."""

import sys

sys.path.insert(0, ".")

from choices.analysis.create_summary import (
    discover_experiments,
    compute_frequency_results,
)

MODEL = "gpt-5-2-non-reasoning"
RESULTS_DIRS = ["results/results_main0", "results/results_main1"]
FACTORS = ["age_group", "gender", "handedness", "nationality", "wealth"]

# Get all experiments
experiments = discover_experiments(RESULTS_DIRS, [MODEL], FACTORS, None)


# Sort: survey_preference first for ALL factors
def sort_key(exp):
    results_dir, factor, model, nudge_type = exp
    dir_order = 0 if "main0" in results_dir else 1
    nudge_order = 0 if nudge_type == "survey_preference" else 1
    return (dir_order, factor, nudge_order, nudge_type)


sorted_experiments = sorted(experiments, key=sort_key)

# Compute results in this specific order
results = []
for results_base_dir, factor_name, model, nudge_type in sorted_experiments:
    experiment_results = compute_frequency_results(
        factor_name, model, nudge_type, results_base_dir
    )
    results.extend(experiment_results)

# Now import and use the plotting code from cc19791
# We extract it inline since the module doesn't exist on this branch
import subprocess  # noqa: E402

plotting_code = subprocess.check_output(
    ["git", "show", "cc19791:choices/analysis/plot_model_effects_v2.py"], text=True
)

# Patch imports - replace get_nudge_display_name and get_nudge_marker with
# local definitions matching cc19791's utils.py (current branch has different markers)
patched_code = plotting_code.replace(
    "from choices.analysis.utils import (\n    get_model_display_name,\n    get_nudge_display_name,\n    get_nudge_marker,\n    is_reasoning_model,\n)",
    """from choices.analysis.utils import (
    get_model_display_name,
    is_reasoning_model,
)

# Marker definitions from cc19791 utils.py (current branch differs)
_NUDGE_MARKERS_V4 = {
    "survey_preference": "o",  # circle
    "user_preference": "s",  # square
    "emotional": "P",  # plus (filled)
    "weak_evidence": "v",  # triangle down
    "few_shot_3": "D",  # diamond
}
_EXTRA_MARKERS_V4 = ["X", "*", "^", "h", "p", "H", "8", "d", "<", ">", "1", "2", "3", "4"]
_dynamic_nudge_markers_v4 = {}

def get_nudge_marker(nudge_type):
    if nudge_type in _NUDGE_MARKERS_V4:
        return _NUDGE_MARKERS_V4[nudge_type]
    if nudge_type not in _dynamic_nudge_markers_v4:
        idx = len(_dynamic_nudge_markers_v4) % len(_EXTRA_MARKERS_V4)
        _dynamic_nudge_markers_v4[nudge_type] = _EXTRA_MARKERS_V4[idx]
    return _dynamic_nudge_markers_v4[nudge_type]

NUDGE_TYPE_ABBREVIATIONS = {
    "survey_preference": "Survey",
    "weak_evidence": "Weak Ev.",
    "strong_evidence": "Strong Ev.",
    "expert_recommendation": "Expert",
    "emotional": "Emotional",
    "identity": "Identity",
    "user_preference": "User Pref.",
    "social_norm": "Social",
    "always_save": "Always",
    "moral_imperative": "Moral",
    "few_shot": "Few-shot",
}

def get_nudge_display_name(nudge_type: str) -> str:
    if nudge_type in NUDGE_TYPE_ABBREVIATIONS:
        return NUDGE_TYPE_ABBREVIATIONS[nudge_type]
    if nudge_type.startswith("few_shot"):
        return "Few-shot"
    return nudge_type.replace("_", " ").title()""",
)

# Patch 2: Move "Influence:" label outside the legend box
patched_code = patched_code.replace(
    """            # Legend 2 (top-right): Influence types in 2 rows
            n_nudges = len(nudge_elements)
            # Calculate ncol to get ~2 rows (ceiling division)
            ncol_influence = (n_nudges + 1 + 1) // 2  # +1 for header, +1 for ceiling

            influence_elements = [influence_header] + nudge_elements
            ax.legend(
                handles=influence_elements,
                loc="lower left",
                bbox_to_anchor=(0.2, 1.02),
                ncol=ncol_influence,
                fontsize=10,
                framealpha=0.9,
                columnspacing=1.2,
            )""",
    """            # Legend 2 (top-right): Influence types in 2 rows
            n_nudges = len(nudge_elements)
            # Calculate ncol to get ~2 rows (ceiling division)
            ncol_influence = (n_nudges + 1) // 2  # +1 for ceiling

            leg2 = ax.legend(
                handles=nudge_elements,
                loc="lower left",
                bbox_to_anchor=(0.2, 1.02),
                ncol=ncol_influence,
                fontsize=10,
                framealpha=0.9,
                columnspacing=1.2,
                title=r"$\bf{Influence}$",
                title_fontsize=10,
                alignment="left",
            )""",
)

# Remove the if __name__ == "__main__" block
main_idx = patched_code.find("\nif __name__")
if main_idx > 0:
    patched_code = patched_code[:main_idx]

# Execute the patched module code to define functions
exec(compile(patched_code, "plot_model_effects_v2.py", "exec"), globals())

# Now use the functions

reasoning = get_default_reasoning_condition(MODEL)  # noqa: F821
data = collect_data_for_single_model(results, MODEL, reasoning)  # noqa: F821
filtered_data = {f: d for f, d in data.items() if f in FACTORS}

# Print f_0_B values for verification
print(
    "f_0_B values (targets: age_group=0.42, gender=0.50, handedness=0.51, nationality=0.51, wealth=0.15):"
)
for f in FACTORS:
    if f in filtered_data:
        print(
            f"  {f}: {filtered_data[f]['f_0_B']:.4f} -> {filtered_data[f]['f_0_B']:.2f}"
        )

import matplotlib.pyplot as plt  # noqa: E402

# Build row_labels and row_keys for the plot
row_keys = sorted(filtered_data.keys())
row_labels = row_keys  # factor names as labels

fig = create_steerability_plot(  # noqa: F821
    row_data=filtered_data,
    row_labels=row_labels,
    row_keys=row_keys,
    single_model_mode=True,
)

output_path = "/home/rob/Downloads/steerability_gpt-5-2-non-reasoning_v5.pdf"
if fig is not None:
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_path}")
else:
    print("ERROR: No figure generated")
