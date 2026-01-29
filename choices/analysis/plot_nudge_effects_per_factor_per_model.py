#!/usr/bin/env python3
"""
Plot nudge type effects comparison for a single model and factor.
Shows 5 nudge types on y-axis, with left side (red) = choosing A, right side (blue) = choosing B.

Reads from main_results_jan26.csv file.
Only plots models with reasoning condition "off" or "none".

Y-axis: Nudge types (5 types)
X-axis: Probability/Frequency
Left side (red shaded): Choosing A
Right side (blue shaded): Choosing B

Usage:
    python plot_nudge_effects_comparison.py --file my_plot --model "DeepSeek V3.2" --factor "age_group"

    # Show in log odds space
    python plot_nudge_effects_comparison.py --file my_plot --model "DeepSeek V3.2" --factor "age_group" --log-odds
"""

import argparse
import math
import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_csv_file(filepath):
    """Parse the CSV results file into a pandas DataFrame"""
    df = pd.read_csv(filepath)

    # Filter for reasoning condition "off" or "none"
    df = df[df["reasoning_condition"].isin(["off", "none"])]

    return df


def freq_to_log_odds(
    freq: float,
    pseudo_n: float = 100.0,
) -> float:
    """
    Convert frequency to log odds with Haldane-Anscombe correction.

    Uses pseudo-counts to handle frequencies at or near 0 and 1.
    The correction adds 0.5 to both wins and losses before computing odds.

    Args:
        freq: Frequency (probability) in [0, 1]
        pseudo_n: Pseudo sample size for correction (default 100)

    Returns:
        Log10 odds ratio
    """
    # Convert frequency to pseudo-counts
    pseudo_wins = freq * pseudo_n
    pseudo_losses = (1 - freq) * pseudo_n

    # Apply Haldane-Anscombe correction
    odds = (pseudo_wins + 0.5) / (pseudo_losses + 0.5)

    return math.log10(odds)


# Nudge type labels
NUDGE_LABELS = {
    "emotional": "Emotional",
    "few_shot_3": "Few Shot 3",
    "survey_preference": "Survey Preference",
    "user_preference": "User Preference",
    "weak_evidence": "Weak Evidence",
}

# Order of nudge types for display
NUDGE_ORDER = [
    "emotional",
    "few_shot_3",
    "survey_preference",
    "user_preference",
    "weak_evidence",
]

# Colors
COLOR_CHOOSE_A = "#E63946"  # Red - choosing A
COLOR_CHOOSE_B = "#457B9D"  # Blue - choosing B
COLOR_BASELINE = "#95A5A6"  # Grey - baseline
COLOR_BASELINE_DIAMOND = "#2A9D8F"  # Green - baseline diamond


def format_factor_label(factor: str, level_A: str = None, level_B: str = None) -> str:
    """Format factor name with level labels."""
    factor_display = factor.replace("_", " ").title()
    if level_A and level_B:
        return f"{factor_display} ({level_A} vs {level_B})"
    return factor_display


def create_nudge_effects_plot(
    df: pd.DataFrame,
    model: str,
    factor: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    log_odds: bool = False,
) -> plt.Figure:
    """
    Create plot showing nudge type effects for a single model and factor.
    Y-axis shows nudge types, X-axis shows probability.
    Left side (red) = choosing A, Right side (blue) = choosing B.

    Args:
        df: DataFrame with the data (already filtered for reasoning="off")
        model: Model name to plot
        factor: Factor to plot
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size (width, height)
        log_odds: If True, data is in log odds space

    Returns:
        The matplotlib Figure object
    """
    # Filter by model and factor
    df = df[(df["model_display_name"] == model) & (df["factor"] == factor)]

    if len(df) == 0:
        print(f"No data to plot for model: {model}, factor: {factor}")
        return None

    # Verify we only have one model (should be true after filtering, but double-check)
    unique_models = df["model_display_name"].unique()
    if len(unique_models) > 1:
        print(
            f"WARNING: Multiple models found: {unique_models}. Using first one: {unique_models[0]}"
        )
        df = df[df["model_display_name"] == unique_models[0]]

    # Ensure we only have one reasoning condition - prefer "off" over "none"
    reasoning_conditions = df["reasoning_condition"].unique()
    if "off" in reasoning_conditions:
        df = df[df["reasoning_condition"] == "off"]
        reasoning = "off"
    elif "none" in reasoning_conditions:
        df = df[df["reasoning_condition"] == "none"]
        reasoning = "none"
    else:
        # If neither exists, take the first one (shouldn't happen after parse_csv_file filter)
        reasoning = df["reasoning_condition"].iloc[0]
        df = df[df["reasoning_condition"] == reasoning]
        print(f"WARNING: Using reasoning condition '{reasoning}' (not 'off' or 'none')")

    if len(df) == 0:
        print(
            f"No data to plot for model: {model}, factor: {factor} with reasoning 'off' or 'none'"
        )
        return None

    print(
        f"Selected model: {model}, reasoning: {reasoning}, factor: {factor}, rows: {len(df)}"
    )

    # Get level labels for this factor
    level_A = df["level_A"].iloc[0] if "level_A" in df.columns else "A"
    level_B = df["level_B"].iloc[0] if "level_B" in df.columns else "B"

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Transform data if needed
    if log_odds:
        df = df.copy()
        # First convert frequencies to log odds
        f_0_B_freq = df["f_0_B"].values
        f_A_B_freq = df["f_A_B"].values
        f_B_B_freq = df["f_B_B"].values

        # Convert to log odds
        df["f_0_B"] = [freq_to_log_odds(f) for f in f_0_B_freq]
        df["f_A_B"] = [freq_to_log_odds(f) for f in f_A_B_freq]
        df["f_B_B"] = [freq_to_log_odds(f) for f in f_B_B_freq]

        # For choosing A, convert (1 - freq) to log odds
        df["f_0_A"] = [freq_to_log_odds(1 - f) for f in f_0_B_freq]
        df["f_A_A"] = [freq_to_log_odds(1 - f) for f in f_A_B_freq]
        df["f_B_A"] = [freq_to_log_odds(1 - f) for f in f_B_B_freq]
    else:
        df = df.copy()
        df["f_0_A"] = 1 - df["f_0_B"]
        df["f_A_A"] = 1 - df["f_A_B"]  # Probability of choosing A when nudged towards A
        df["f_B_A"] = 1 - df["f_B_B"]  # Probability of choosing A when nudged towards B

    # Y positions for each nudge type (reduced spacing for more compact plot)
    n_nudges = len(NUDGE_ORDER)
    y_positions = np.arange(n_nudges) * 0.7  # Reduce spacing between rows

    # Track labels for legend
    labels_added = set()

    # Process each nudge type
    for i, nudge_type in enumerate(NUDGE_ORDER):
        nudge_df = df[df["nudge_type"] == nudge_type]

        if len(nudge_df) == 0:
            continue

        y_pos = y_positions[i]

        # Get values for this nudge type (take mean if multiple rows)
        # f_0_A = nudge_df["f_0_A"].mean()
        f_0_B = nudge_df["f_0_B"].mean()
        # f_A_A = nudge_df["f_A_A"].mean()
        f_A_B = nudge_df["f_A_B"].mean()
        # f_B_A = nudge_df["f_B_A"].mean()
        f_B_B = nudge_df["f_B_B"].mean()

        # Use fixed jitter for reproducibility (reduced for tighter spacing)
        np.random.seed(42 + i)
        jitter_base = np.random.uniform(-0.05, 0.05)
        jitter_A = np.random.uniform(-0.08, -0.03)
        jitter_B = np.random.uniform(0.03, 0.08)

        # Plot baseline (neutral) - probability of choosing B
        label_key_base = "baseline"
        ax.scatter(
            [f_0_B],
            [y_pos + jitter_base],
            color=COLOR_BASELINE,
            marker="o",
            alpha=0.8,
            s=150,
            edgecolors="white",
            linewidths=1.5,
            label="Baseline (neutral)" if label_key_base not in labels_added else "",
            zorder=4,
        )
        if label_key_base not in labels_added:
            labels_added.add(label_key_base)

        # Plot nudge towards A - probability of choosing B when nudged towards A
        # Red color because nudge is →A
        label_key_A = "nudge_A"
        ax.scatter(
            [f_A_B],
            [y_pos + jitter_A],
            color=COLOR_CHOOSE_A,
            marker="s",
            alpha=0.8,
            s=150,
            edgecolors="white",
            linewidths=1.5,
            label=f"Nudge →{level_A}" if label_key_A not in labels_added else "",
            zorder=3,
        )
        if label_key_A not in labels_added:
            labels_added.add(label_key_A)

        # Plot nudge towards B - probability of choosing B when nudged towards B
        # Blue color because nudge is →B
        label_key_B = "nudge_B"
        ax.scatter(
            [f_B_B],
            [y_pos + jitter_B],
            color=COLOR_CHOOSE_B,
            marker="^",
            alpha=0.8,
            s=150,
            edgecolors="white",
            linewidths=1.5,
            label=f"Nudge →{level_B}" if label_key_B not in labels_added else "",
            zorder=3,
        )
        if label_key_B not in labels_added:
            labels_added.add(label_key_B)

    # Add shaded background regions
    # Left side (red) for lower frequencies (more choosing A)
    # Right side (blue) for higher frequencies (more choosing B)
    if not log_odds:
        ax.axvspan(-0.05, 0.5, alpha=0.15, color=COLOR_CHOOSE_A, zorder=0)
        ax.axvspan(0.5, 1.05, alpha=0.15, color=COLOR_CHOOSE_B, zorder=0)

    # Add reference line at 0.5 (or 0 for log odds)
    if not log_odds:
        ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)
    else:
        ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1)

    # Set y-axis labels - nudge types
    nudge_labels = [NUDGE_LABELS.get(nt, nt) for nt in NUDGE_ORDER]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(nudge_labels, fontsize=14)
    ax.set_ylabel("Nudge types", fontsize=15)

    # Build x-axis label - clarify it's frequency of choosing level_B
    if log_odds:
        ax.set_xlabel(f"Log₁₀ Odds of Choosing {level_B}", fontsize=15)
    else:
        ax.set_xlabel(f"Frequency of Choosing {level_B}", fontsize=15)

    # Set x-axis limits
    if log_odds:
        all_values = []
        all_values.extend(df["f_0_A"].values)
        all_values.extend(df["f_0_B"].values)
        all_values.extend(df["f_A_A"].values)
        all_values.extend(df["f_A_B"].values)
        all_values.extend(df["f_B_A"].values)
        all_values.extend(df["f_B_B"].values)
        if len(all_values) > 0:
            min_val, max_val = min(all_values), max(all_values)
            padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
    else:
        ax.set_xlim(-0.05, 1.05)

    # Create simplified legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = []

    # Shaded regions (most prominent - explain what they represent)
    if not log_odds:
        legend_elements.append(
            Patch(facecolor=COLOR_CHOOSE_A, alpha=0.15, label=f"Prefers {level_A}")
        )
        legend_elements.append(
            Patch(facecolor=COLOR_CHOOSE_B, alpha=0.15, label=f"Prefers {level_B}")
        )

    # Nudge towards A (square marker)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=COLOR_CHOOSE_A,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=0,
            label=f"Nudge →{level_A}",
        )
    )

    # Nudge towards B (triangle marker)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=COLOR_CHOOSE_B,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=0,
            label=f"Nudge →{level_B}",
        )
    )

    # Baseline circle (neutral/grey)
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLOR_BASELINE,
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            linewidth=0,
            label="Baseline (neutral)",
        )
    )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=12,
        framealpha=0.9,
        ncol=1,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=13)

    # Set y-axis limits (tighter margins for more compact plot)
    # Adjust for reduced spacing (0.7 multiplier)
    max_y_pos = (n_nudges - 1) * 0.7
    ax.set_ylim(-0.1, max_y_pos + 0.3)

    # Add horizontal grid lines to separate nudge types (adjusted for reduced spacing)
    for i in range(n_nudges - 1):
        ax.axhline(
            y=(i + 0.5) * 0.7,  # Midpoint between rows with 0.7 spacing
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

    # Invert y-axis so first nudge type is at top
    ax.invert_yaxis()

    plt.tight_layout(pad=1.5)

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create plot showing nudge type effects for a single model and factor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python plot_nudge_effects_comparison.py --file my_plot --model "DeepSeek V3.2" --factor "age_group"

    # Show in log odds space
    python plot_nudge_effects_comparison.py --file my_plot --model "DeepSeek V3.2" --factor "age_group" --log-odds
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Output filename (without extension, will be saved in plots/ directory)",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model name to plot (e.g., 'DeepSeek V3.2')",
    )

    parser.add_argument(
        "--factor",
        type=str,
        required=True,
        help="Factor to plot (e.g., 'age_group', 'wealth')",
    )

    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title",
    )

    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[10, 8],
        help="Figure size (width height)",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (only save to file)",
    )

    parser.add_argument(
        "--log-odds",
        action="store_true",
        help="Show plot in log odds space instead of frequency space",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="main_results_jan26.csv",
        help="Input CSV file path (default: main_results_jan26.csv)",
    )

    args = parser.parse_args()

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Determine output path
    output_path = f"plots/{args.file}.pdf"

    # Print header
    print("=" * 70)
    print("Nudge Type Effects Comparison Plot")
    print("=" * 70)
    print(f"Input file: {args.input}")
    print(f"Model: {args.model}")
    print(f"Factor: {args.factor}")
    print(f"Space: {'Log Odds' if args.log_odds else 'Frequency'}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Load data from CSV file
    print(f"Loading data from {args.input}...")
    df = parse_csv_file(args.input)
    print(f"Loaded {len(df)} conditions (reasoning=off/none)")
    print()

    if len(df) == 0:
        print("ERROR: No data remaining after filtering!")
        return

    # Print summary
    print("Data Summary:")
    print("-" * 70)
    print(f"  Models: {sorted(df['model_display_name'].unique())}")
    print(f"  Factors: {sorted(df['factor'].unique())}")
    print(f"  Nudge types: {sorted(df['nudge_type'].unique())}")
    print()

    # Create plot
    figsize = tuple(args.figsize)

    fig = create_nudge_effects_plot(
        df,
        model=args.model,
        factor=args.factor,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        log_odds=args.log_odds,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
