#!/usr/bin/env python3
"""
Plot steerability distributions with violin plots for each factor.

For each factor, shows two horizontal violin plots:
- Left: distribution of f(B) across all models/nudge types when nudging towards A
- Right: distribution of f(B) across all models/nudge types when nudging towards B

The average baseline preference f_0(B) is visually indicated on both plots.

Usage:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --factors gender age_group wealth

    # Save to file
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --output steerability_factors.pdf

    # Show in log odds space
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --log-odds

    # Show median and IQR instead of mean
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --percentiles

    # Show effects relative to baseline
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --relative

    # Combine: relative effects in log odds space
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --log-odds --relative

    # Show one row per model (forces log odds space)
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --rows models

    # Show one row per nudge type (forces log odds space)
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --rows nudges

    # Show rows by baseline bias magnitude bins
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --rows baseline --n-bins 5

    # Filter by reasoning conditions
    uv run python -m choices.analysis.plot_steerability \
        --results-dirs results \
        --reasoning-conditions none before after
"""

import argparse
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
    discover_experiments,
)


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


def transform_data_to_log_odds(
    data_by_factor: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Transform all frequency data to log odds space.

    Args:
        data_by_factor: Dictionary mapping factor -> frequency data

    Returns:
        Same structure with frequencies converted to log odds
    """
    transformed = {}
    for factor, factor_data in data_by_factor.items():
        transformed[factor] = {
            "f_A_B": [freq_to_log_odds(f) for f in factor_data["f_A_B"]],
            "f_B_B": [freq_to_log_odds(f) for f in factor_data["f_B_B"]],
            "f_0_B": [freq_to_log_odds(f) for f in factor_data["f_0_B"]],
            "level_A": factor_data.get("level_A"),
            "level_B": factor_data.get("level_B"),
            # Preserve significance data
            "sig_A": factor_data.get("sig_A", []),
            "sig_B": factor_data.get("sig_B", []),
            "sig_bias": factor_data.get("sig_bias", []),
        }
    return transformed


def transform_data_to_relative(
    data_by_factor: Dict[str, Dict[str, List[float]]],
) -> Dict[str, Dict[str, List[float]]]:
    """
    Transform data to be relative to baseline (subtract baseline from each nudge condition).

    For each experiment, computes:
    - f_A_B - f_0_B (effect of nudging towards A)
    - f_B_B - f_0_B (effect of nudging towards B)
    - f_0_B remains as 0 (reference point)

    This works in both frequency space (giving frequency differences) and
    log odds space (giving log odds ratios).

    Args:
        data_by_factor: Dictionary mapping factor -> data (frequency or log odds)

    Returns:
        Same structure with values relative to baseline
    """
    transformed = {}
    for factor, factor_data in data_by_factor.items():
        # Compute relative values (subtract corresponding baseline)
        relative_f_A_B = [
            f_A - f_0 for f_A, f_0 in zip(factor_data["f_A_B"], factor_data["f_0_B"])
        ]
        relative_f_B_B = [
            f_B - f_0 for f_B, f_0 in zip(factor_data["f_B_B"], factor_data["f_0_B"])
        ]
        # Baseline becomes zero (reference point)
        relative_f_0_B = [0.0] * len(factor_data["f_0_B"])

        transformed[factor] = {
            "f_A_B": relative_f_A_B,
            "f_B_B": relative_f_B_B,
            "f_0_B": relative_f_0_B,
            "level_A": factor_data.get("level_A"),
            "level_B": factor_data.get("level_B"),
            # Preserve significance data
            "sig_A": factor_data.get("sig_A", []),
            "sig_B": factor_data.get("sig_B", []),
            "sig_bias": factor_data.get("sig_bias", []),
        }
    return transformed


def collect_data_by_factor(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by factor.

    Returns:
        Dictionary mapping factor -> {
            'f_A_B': list of f_A(B) values (freq of B when nudged towards A),
            'f_B_B': list of f_B(B) values (freq of B when nudged towards B),
            'f_0_B': list of f_0(B) values (baseline freq of B),
            'level_A': name of level A (e.g., 'poor'),
            'level_B': name of level B (e.g., 'rich'),
            'sig_A': list of significance flags for nudge towards A,
            'sig_B': list of significance flags for nudge towards B,
            'sig_bias': list of significance flags for steerability bias,
        }
    """
    data_by_factor: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
            "sig_A": [],
            "sig_B": [],
            "sig_bias": [],
        }
    )

    for r in results:
        data_by_factor[r.factor]["f_A_B"].append(r.f_A_B)
        data_by_factor[r.factor]["f_B_B"].append(r.f_B_B)
        data_by_factor[r.factor]["f_0_B"].append(r.f_0_B)
        data_by_factor[r.factor]["sig_A"].append(r.sig_A)
        data_by_factor[r.factor]["sig_B"].append(r.sig_B)
        data_by_factor[r.factor]["sig_bias"].append(r.sig_bias)
        # Store level names (they should be consistent within a factor)
        if data_by_factor[r.factor]["level_A"] is None:
            data_by_factor[r.factor]["level_A"] = r.level_A
            data_by_factor[r.factor]["level_B"] = r.level_B

    return dict(data_by_factor)


def normalize_direction(
    f_A_B: float,
    f_B_B: float,
    f_0_B: float,
    sig_A: bool = False,
    sig_B: bool = False,
    sig_bias: bool = False,
) -> Tuple[float, float, float, bool, bool, bool]:
    """
    Normalize frequencies so that A corresponds to the less-preferred option at baseline.

    This ensures that when aggregating across factors:
    - f_A_B represents nudging AWAY from baseline preference
    - f_B_B represents nudging TOWARDS baseline preference
    - f_0_B is always >= 0.5 (B is always the baseline-preferred option)

    If baseline prefers A (f_0_B < 0.5), we swap the labels:
    - new_f_A_B = 1 - old_f_B_B (nudge towards new A = nudge towards old B, measure new B = old A)
    - new_f_B_B = 1 - old_f_A_B (nudge towards new B = nudge towards old A, measure new B = old A)
    - new_f_0_B = 1 - old_f_0_B

    Args:
        f_A_B: Frequency of choosing B when nudged towards A
        f_B_B: Frequency of choosing B when nudged towards B
        f_0_B: Baseline frequency of choosing B
        sig_A: Significance flag for nudge towards A
        sig_B: Significance flag for nudge towards B
        sig_bias: Significance flag for steerability bias

    Returns:
        Tuple of (normalized_f_A_B, normalized_f_B_B, normalized_f_0_B,
                  normalized_sig_A, normalized_sig_B, sig_bias)
    """
    if f_0_B >= 0.5:
        # B is already the preferred option, no change needed
        return f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias
    else:
        # A is preferred, swap labels (also swap significance flags)
        return 1 - f_B_B, 1 - f_A_B, 1 - f_0_B, sig_B, sig_A, sig_bias


def collect_data_by_model(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by model and reasoning condition.

    Models with different reasoning conditions are treated as separate entries.
    Direction is normalized so A always corresponds to the less-preferred option
    at baseline (i.e., f_A_B shows nudging AWAY from baseline preference).

    Returns:
        Dictionary mapping "model (reasoning)" -> {
            'f_A_B': list of f_A(B) values across all factors/nudge types,
            'f_B_B': list of f_B(B) values across all factors/nudge types,
            'f_0_B': list of f_0(B) values (baseline),
            'level_A': None (not applicable for model grouping),
            'level_B': None (not applicable for model grouping),
            'sig_A': list of significance flags for nudge towards A,
            'sig_B': list of significance flags for nudge towards B,
            'sig_bias': list of significance flags for steerability bias,
        }
    """
    from choices.analysis.utils import get_model_display_name

    data_by_model: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
            "sig_A": [],
            "sig_B": [],
            "sig_bias": [],
        }
    )

    for r in results:
        # Normalize direction so A is always the less-preferred option
        f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias = normalize_direction(
            r.f_A_B, r.f_B_B, r.f_0_B, r.sig_A, r.sig_B, r.sig_bias
        )

        # Include reasoning condition in key to separate different conditions
        display_name = get_model_display_name(r.model)
        model_key = f"{display_name} ({r.reasoning_condition})"
        data_by_model[model_key]["f_A_B"].append(f_A_B)
        data_by_model[model_key]["f_B_B"].append(f_B_B)
        data_by_model[model_key]["sig_A"].append(sig_A)
        data_by_model[model_key]["sig_B"].append(sig_B)
        data_by_model[model_key]["sig_bias"].append(sig_bias)
        data_by_model[model_key]["f_0_B"].append(f_0_B)

    return dict(data_by_model)


def collect_data_by_nudge_type(
    results: List[FrequencyResult],
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by nudge type.

    Direction is normalized so A always corresponds to the less-preferred option
    at baseline (i.e., f_A_B shows nudging AWAY from baseline preference).

    Returns:
        Dictionary mapping nudge_type -> {
            'f_A_B': list of f_A(B) values across all factors/models,
            'f_B_B': list of f_B(B) values across all factors/models,
            'f_0_B': list of f_0(B) values (baseline),
            'level_A': None (not applicable for nudge type grouping),
            'level_B': None (not applicable for nudge type grouping),
            'sig_A': list of significance flags for nudge towards A,
            'sig_B': list of significance flags for nudge towards B,
            'sig_bias': list of significance flags for steerability bias,
        }
    """
    data_by_nudge: Dict[str, Dict[str, any]] = defaultdict(
        lambda: {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
            "sig_A": [],
            "sig_B": [],
            "sig_bias": [],
        }
    )

    for r in results:
        # Normalize direction so A is always the less-preferred option
        f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias = normalize_direction(
            r.f_A_B, r.f_B_B, r.f_0_B, r.sig_A, r.sig_B, r.sig_bias
        )

        # Format nudge type for display
        nudge_key = r.nudge_type.replace("_", " ").title()
        data_by_nudge[nudge_key]["f_A_B"].append(f_A_B)
        data_by_nudge[nudge_key]["f_B_B"].append(f_B_B)
        data_by_nudge[nudge_key]["f_0_B"].append(f_0_B)
        data_by_nudge[nudge_key]["sig_A"].append(sig_A)
        data_by_nudge[nudge_key]["sig_B"].append(sig_B)
        data_by_nudge[nudge_key]["sig_bias"].append(sig_bias)

    return dict(data_by_nudge)


def collect_data_by_baseline_bin(
    results: List[FrequencyResult],
    n_bins: int = 5,
) -> Dict[str, Dict[str, any]]:
    """
    Collect frequency data grouped by baseline bias magnitude bins.

    Direction is normalized so A always corresponds to the less-preferred option
    at baseline. Then samples are binned by f_0_B (which is always >= 0.5 after
    normalization, representing the frequency of the baseline-preferred option).

    Bins are created using quantiles so each bin has approximately equal samples.

    Args:
        results: List of FrequencyResult objects
        n_bins: Number of bins to create (each with ~equal sample count)

    Returns:
        Dictionary mapping bin_label -> {
            'f_A_B': list of f_A(B) values,
            'f_B_B': list of f_B(B) values,
            'f_0_B': list of f_0(B) values (baseline),
            'level_A': None (not applicable for baseline grouping),
            'level_B': None (not applicable for baseline grouping),
            'sig_A': list of significance flags for nudge towards A,
            'sig_B': list of significance flags for nudge towards B,
            'sig_bias': list of significance flags for steerability bias,
        }
    """
    import numpy as np

    # First pass: normalize all data to compute quantile-based bin edges
    normalized_data = []
    for r in results:
        f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias = normalize_direction(
            r.f_A_B, r.f_B_B, r.f_0_B, r.sig_A, r.sig_B, r.sig_bias
        )
        normalized_data.append((f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias))

    all_f_0_B = np.array([d[2] for d in normalized_data])

    # Create quantile-based bin edges (ensures ~equal samples per bin)
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(all_f_0_B, quantiles)

    # Ensure edges are unique (can happen with many identical values)
    bin_edges = np.unique(bin_edges)
    actual_n_bins = len(bin_edges) - 1

    if actual_n_bins < n_bins:
        print(
            f"Note: Reduced to {actual_n_bins} bins due to data distribution "
            f"(many samples have identical baseline values)"
        )

    # Initialize data structure for each bin
    data_by_bin: Dict[str, Dict[str, any]] = {}
    for i in range(actual_n_bins):
        # Create label showing the interval
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        # Use [lower, upper) notation, except for last bin which is [lower, upper]
        if i == actual_n_bins - 1:
            bin_label = f"[{lower:.2f}, {upper:.2f}]"
        else:
            bin_label = f"[{lower:.2f}, {upper:.2f})"
        data_by_bin[bin_label] = {
            "f_A_B": [],
            "f_B_B": [],
            "f_0_B": [],
            "level_A": None,
            "level_B": None,
            "sig_A": [],
            "sig_B": [],
            "sig_bias": [],
            "_bin_index": i,  # For sorting
        }

    # Second pass: assign each result to a bin
    for f_A_B, f_B_B, f_0_B, sig_A, sig_B, sig_bias in normalized_data:
        # Find which bin this belongs to
        bin_idx = np.searchsorted(bin_edges[1:], f_0_B, side="right")
        bin_idx = min(bin_idx, actual_n_bins - 1)  # Clamp to last bin

        # Find the corresponding bin label
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == actual_n_bins - 1:
            bin_label = f"[{lower:.2f}, {upper:.2f}]"
        else:
            bin_label = f"[{lower:.2f}, {upper:.2f})"

        data_by_bin[bin_label]["f_A_B"].append(f_A_B)
        data_by_bin[bin_label]["f_B_B"].append(f_B_B)
        data_by_bin[bin_label]["f_0_B"].append(f_0_B)
        data_by_bin[bin_label]["sig_A"].append(sig_A)
        data_by_bin[bin_label]["sig_B"].append(sig_B)
        data_by_bin[bin_label]["sig_bias"].append(sig_bias)

    # Remove empty bins (shouldn't happen with quantile-based edges, but just in case)
    non_empty = {k: v for k, v in data_by_bin.items() if len(v["f_A_B"]) > 0}

    # Sort by bin index and remove the helper field
    sorted_bins = dict(
        sorted(non_empty.items(), key=lambda x: x[1].get("_bin_index", 0))
    )
    for v in sorted_bins.values():
        v.pop("_bin_index", None)

    return sorted_bins


def format_factor_label(factor: str, level_A: str, level_B: str) -> str:
    """Format factor name with level labels."""
    factor_display = factor.replace("_", " ").title()
    return f"{factor_display}\n({level_A} vs {level_B})"


def _split_violin_halves(parts, side: str, y_position: float):
    """
    Modify violin plot to show only one half (left or right).

    Args:
        parts: The violinplot collection
        side: "left" or "right"
        y_position: The y position of the violin
    """
    for vp in parts["bodies"]:
        # Get the path vertices
        paths = vp.get_paths()
        if not paths:
            continue
        path = paths[0]
        vertices = path.vertices.copy()

        # For horizontal violins, we clip on y-axis
        # side="left" means keep points below y_position
        # side="right" means keep points above y_position
        if side == "left":
            # Keep only the lower half (below center line)
            vertices[vertices[:, 1] > y_position, 1] = y_position
        else:
            # Keep only the upper half (above center line)
            vertices[vertices[:, 1] < y_position, 1] = y_position

        path.vertices = vertices


def create_steerability_violin_plot(
    data_by_row: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, None),
    log_odds: bool = False,
    percentiles: bool = False,
    relative: bool = False,
    row_type: str = "factors",
    show_bias: bool = False,
    show_significance: bool = False,
    single_model: bool = False,
) -> plt.Figure:
    """
    Create violin plot showing steerability distributions.

    For each row, shows a split violin:
    - Left half (below row center): distribution when nudged towards A
    - Right half (above row center): distribution when nudged towards B

    Args:
        data_by_row: Dictionary mapping row key -> frequency data
        output_path: Optional path to save the figure
        title: Optional custom title
        figsize: Figure size (width, height). If height is None, auto-calculated.
        log_odds: If True, data is in log odds space
        percentiles: If True, show median and 25/75 percentiles instead of mean
        relative: If True, values are relative to baseline
        row_type: Type of rows - "factors", "models", or "nudges"
        show_bias: If True, add a column showing steerability bias as violin plot
        show_significance: If True, color non-significant points in grey
        single_model: If True, only show baseline average (not range) since data is from one model

    Returns:
        The matplotlib Figure object
    """
    rows = sorted(data_by_row.keys())
    n_rows = len(rows)

    if n_rows == 0:
        print("No data to plot.")
        return None

    # Calculate figure height based on number of rows
    height = figsize[1] if figsize[1] else max(4, n_rows * 1.5)

    # Create figure with optional bias column
    if show_bias:
        # Use gridspec for width ratio: violin plot gets 3 parts, bias gets 1 part
        fig, (ax, ax_bias) = plt.subplots(
            1,
            2,
            figsize=(figsize[0] * 1.35, height),
            gridspec_kw={"width_ratios": [3, 1], "wspace": 0.12},
        )
    else:
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        ax_bias = None

    # Colors
    color_nudge_A = "#E63946"  # Red - nudging towards A
    color_nudge_B = "#457B9D"  # Blue - nudging towards B
    color_baseline = "#2A9D8F"  # Teal - baseline marker
    color_nonsig = "#A0A0A0"  # Grey - non-significant points

    # Y positions for each row
    y_positions = np.arange(n_rows)

    # Process each row separately to create split violins
    for i, row_key in enumerate(rows):
        row_data = data_by_row[row_key]
        y_pos = y_positions[i]

        # Create violin for nudge towards A (left/lower half)
        if len(row_data["f_A_B"]) >= 2:
            parts_A = ax.violinplot(
                [row_data["f_A_B"]],
                positions=[y_pos],
                vert=False,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )

            # Style and clip to left half
            for pc in parts_A["bodies"]:
                pc.set_facecolor(color_nudge_A)
                pc.set_edgecolor(color_nudge_A)
                pc.set_alpha(0.3)  # Lighter background

            _split_violin_halves(parts_A, "left", y_pos)

        # Create violin for nudge towards B (right/upper half)
        if len(row_data["f_B_B"]) >= 2:
            parts_B = ax.violinplot(
                [row_data["f_B_B"]],
                positions=[y_pos],
                vert=False,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                widths=0.7,
            )

            # Style and clip to right half
            for pc in parts_B["bodies"]:
                pc.set_facecolor(color_nudge_B)
                pc.set_edgecolor(color_nudge_B)
                pc.set_alpha(0.3)  # Lighter background

            _split_violin_halves(parts_B, "right", y_pos)

        # Add individual data points as dots
        # Scatter points for nudge A (below center line)
        n_A = len(row_data["f_A_B"])
        jitter_A = np.random.uniform(-0.25, -0.05, n_A)
        if show_significance and row_data.get("sig_A"):
            # Color by significance: grey for non-significant, colored for significant
            sig_A_flags = row_data["sig_A"]
            colors_A = [color_nudge_A if sig else color_nonsig for sig in sig_A_flags]
            ax.scatter(
                row_data["f_A_B"],
                y_pos + jitter_A,
                c=colors_A,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
        else:
            ax.scatter(
                row_data["f_A_B"],
                y_pos + jitter_A,
                color=color_nudge_A,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Scatter points for nudge B (above center line)
        n_B = len(row_data["f_B_B"])
        jitter_B = np.random.uniform(0.05, 0.25, n_B)
        if show_significance and row_data.get("sig_B"):
            # Color by significance: grey for non-significant, colored for significant
            sig_B_flags = row_data["sig_B"]
            colors_B = [color_nudge_B if sig else color_nonsig for sig in sig_B_flags]
            ax.scatter(
                row_data["f_B_B"],
                y_pos + jitter_B,
                c=colors_B,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
        else:
            ax.scatter(
                row_data["f_B_B"],
                y_pos + jitter_B,
                color=color_nudge_B,
                alpha=0.7,
                s=25,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

        # Add central tendency markers for nudge conditions
        if percentiles:
            # Use median and show 25/75 percentiles
            center_nudge_A = np.median(row_data["f_A_B"])
            center_nudge_B = np.median(row_data["f_B_B"])
            p25_nudge_A = np.percentile(row_data["f_A_B"], 25)
            p75_nudge_A = np.percentile(row_data["f_A_B"], 75)
            p25_nudge_B = np.percentile(row_data["f_B_B"], 25)
            p75_nudge_B = np.percentile(row_data["f_B_B"], 75)
        else:
            # Use mean
            center_nudge_A = np.mean(row_data["f_A_B"])
            center_nudge_B = np.mean(row_data["f_B_B"])

        # Central marker for nudge A (in lower half)
        # Add black outline for visibility (drawn first, behind)
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color="black",
            marker="|",
            s=550,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_A],
            [y_pos - 0.15],
            color=color_nudge_A,
            marker="|",
            s=500,
            linewidths=2.5,
            zorder=7,
        )

        # Central marker for nudge B (in upper half)
        # Add black outline for visibility (drawn first, behind)
        ax.scatter(
            [center_nudge_B],
            [y_pos + 0.15],
            color="black",
            marker="|",
            s=550,
            linewidths=3,
            zorder=6,
        )
        ax.scatter(
            [center_nudge_B],
            [y_pos + 0.15],
            color=color_nudge_B,
            marker="|",
            s=500,
            linewidths=2.5,
            zorder=7,
        )

        # Add percentile markers if enabled
        if percentiles:
            # 25th and 75th percentile markers for nudge A - more visible
            ax.scatter(
                [p25_nudge_A, p75_nudge_A],
                [y_pos - 0.15, y_pos - 0.15],
                color=color_nudge_A,
                marker="|",
                s=250,
                linewidths=2.5,
                zorder=6,
            )

            # 25th and 75th percentile markers for nudge B - more visible
            ax.scatter(
                [p25_nudge_B, p75_nudge_B],
                [y_pos + 0.15, y_pos + 0.15],
                color=color_nudge_B,
                marker="|",
                s=250,
                linewidths=2.5,
                zorder=6,
            )

            # Connect percentiles with a horizontal line (IQR) - thicker
            ax.plot(
                [p25_nudge_A, p75_nudge_A],
                [y_pos - 0.15, y_pos - 0.15],
                color=color_nudge_A,
                linewidth=2.5,
                alpha=0.8,
                zorder=5,
            )
            ax.plot(
                [p25_nudge_B, p75_nudge_B],
                [y_pos + 0.15, y_pos + 0.15],
                color=color_nudge_B,
                linewidth=2.5,
                alpha=0.8,
                zorder=5,
            )

        # Add baseline marker (median or mean f_0(B)) - skip in relative mode (always 0)
        if not relative:
            if percentiles:
                center_baseline = np.median(row_data["f_0_B"])
            else:
                center_baseline = np.mean(row_data["f_0_B"])

            # Check if there's variation in baseline (aggregating multiple models)
            # Only show range if we have multiple models (not single_model mode)
            baseline_min = np.min(row_data["f_0_B"])
            baseline_max = np.max(row_data["f_0_B"])
            has_baseline_range = baseline_min != baseline_max and not single_model

            # Draw green handle bars (error bars) if there's baseline variation
            if has_baseline_range:
                # Draw horizontal error bar line
                ax.plot(
                    [baseline_min, baseline_max],
                    [y_pos, y_pos],
                    color=color_baseline,
                    linewidth=2,
                    alpha=0.9,
                    zorder=7,
                )
                # Draw vertical "handles" at the ends
                handle_height = 0.15
                ax.plot(
                    [baseline_min, baseline_min],
                    [y_pos - handle_height, y_pos + handle_height],
                    color=color_baseline,
                    linewidth=2,
                    alpha=0.9,
                    zorder=7,
                )
                ax.plot(
                    [baseline_max, baseline_max],
                    [y_pos - handle_height, y_pos + handle_height],
                    color=color_baseline,
                    linewidth=2,
                    alpha=0.9,
                    zorder=7,
                )

            # Draw vertical line at baseline spanning the row (high zorder to be visible)
            ax.plot(
                [center_baseline, center_baseline],
                [y_pos - 0.35, y_pos + 0.35],
                color=color_baseline,
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=8,
            )

            # Add diamond marker at center (highest zorder to always be on top)
            ax.scatter(
                [center_baseline],
                [y_pos],
                color=color_baseline,
                marker="D",
                s=100,
                edgecolors="black",
                linewidths=1.5,
                zorder=9,
            )

    # Add reference line at appropriate value
    # - relative mode: 0 (no effect)
    # - log odds mode: 0 (equal odds)
    # - frequency mode: 0.5 (no preference)
    if relative:
        ref_value = 0.0
    elif log_odds:
        ref_value = 0.0
    else:
        ref_value = 0.5
    ax.axvline(
        x=ref_value, color="gray", linestyle=":", linewidth=1, alpha=0.5, zorder=1
    )

    # Remove y-axis tick labels - options will be shown on sides instead
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * len(rows))

    # Build x-axis label based on mode
    if relative and log_odds:
        ax.set_xlabel("Δ Log₁₀ Odds (relative to baseline)", fontsize=12)
    elif relative:
        ax.set_xlabel("Δ Frequency (relative to baseline)", fontsize=12)
    elif log_odds:
        ax.set_xlabel("Log₁₀ Odds of Choosing B", fontsize=12)
    else:
        ax.set_xlabel("Frequency of Choosing B", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    else:
        # Build space label
        if relative and log_odds:
            space_label = "(Relative Log Odds)"
        elif relative:
            space_label = "(Relative Frequency)"
        elif log_odds:
            space_label = "(Log Odds Space)"
        else:
            space_label = "(Frequency Space)"

        # Build row type label
        if row_type == "factors":
            row_label = "Factor"
            dist_label = "models and nudge types"
        elif row_type == "models":
            row_label = "Model"
            dist_label = "factors and nudge types"
        elif row_type == "nudges":
            row_label = "Nudge Type"
            dist_label = "factors and models"
        else:  # baseline
            row_label = "Baseline Preference"
            dist_label = "factors, models, and nudge types"

        ax.set_title(
            f"Steerability by {row_label} {space_label}\n(Distribution across {dist_label})",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

    # Set x-axis limits
    if log_odds or relative:
        # Auto-scale for log odds or relative mode, with some padding
        all_values = []
        for rd in data_by_row.values():
            all_values.extend(rd["f_A_B"])
            all_values.extend(rd["f_B_B"])
            all_values.extend(rd["f_0_B"])
        if all_values:
            min_val, max_val = min(all_values), max(all_values)
            padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
            ax.set_xlim(min_val - padding, max_val + padding)
    else:
        ax.set_xlim(-0.05, 1.05)

    # Add option labels on left and right sides of each row (only for factors)
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    label_offset = x_range * 0.02  # Small offset from plot edge

    if row_type == "factors":
        for i, row_key in enumerate(rows):
            y_pos = y_positions[i]
            rd = data_by_row[row_key]
            # Get level names from data (extracted from FrequencyResult)
            level_A = rd.get("level_A") or "A"
            level_B = rd.get("level_B") or "B"

            # Left label (level A - towards which nudging decreases f(B))
            ax.text(
                x_min - label_offset,
                y_pos,
                level_A,
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#E63946",  # Same as nudge A color
                clip_on=False,  # Allow text outside plot area
            )

            # Right label (level B - towards which nudging increases f(B))
            # Skip right label if bias column is shown (would overlap)
            if not show_bias:
                ax.text(
                    x_max + label_offset,
                    y_pos,
                    level_B,
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="#457B9D",  # Same as nudge B color
                    clip_on=False,  # Allow text outside plot area
                )
    else:
        # For models/nudges, show row labels on the left side
        for i, row_key in enumerate(rows):
            y_pos = y_positions[i]
            ax.text(
                x_min - label_offset,
                y_pos,
                row_key,
                ha="right",
                va="center",
                fontsize=10,
                clip_on=False,
            )

    # Draw steerability bias column if enabled
    if show_bias and ax_bias is not None:
        # Compute steerability bias for each row
        # steerability_A = log10(odds(A|nudge_A)) - log10(odds(A|baseline))
        # steerability_B = log10(odds(B|nudge_B)) - log10(odds(B|baseline))
        # bias = steerability_B - steerability_A
        # Positive bias = more steerable towards B (right side of violin)

        def compute_steerability_bias_from_freq(
            f_A_B: float, f_B_B: float, f_0_B: float
        ) -> float:
            """Compute steerability bias from frequency data."""
            # Convert to frequencies of A
            f_A_A = 1.0 - f_A_B  # freq of A when nudged towards A
            f_0_A = 1.0 - f_0_B  # baseline freq of A
            # Compute log odds for each condition
            log_odds_A_nudged = freq_to_log_odds(f_A_A)
            log_odds_A_baseline = freq_to_log_odds(f_0_A)
            log_odds_B_nudged = freq_to_log_odds(f_B_B)
            log_odds_B_baseline = freq_to_log_odds(f_0_B)
            # Steerability = change in log odds when nudged
            steerability_A = log_odds_A_nudged - log_odds_A_baseline
            steerability_B = log_odds_B_nudged - log_odds_B_baseline
            # Bias = differential steerability
            return steerability_B - steerability_A

        def compute_steerability_bias_from_log_odds(
            lo_A_B: float, lo_B_B: float, lo_0_B: float
        ) -> float:
            """Compute steerability bias from log odds data.

            When data is in log odds space (log10(odds of B)):
            - steerability_A = log_odds(A|nudge_A) - log_odds(A|baseline)
                            = -lo_A_B - (-lo_0_B) = lo_0_B - lo_A_B
            - steerability_B = log_odds(B|nudge_B) - log_odds(B|baseline)
                            = lo_B_B - lo_0_B
            - bias = steerability_B - steerability_A
            """
            steerability_A = lo_0_B - lo_A_B
            steerability_B = lo_B_B - lo_0_B
            return steerability_B - steerability_A

        def compute_steerability_bias_from_relative(
            rel_A_B: float, rel_B_B: float
        ) -> float:
            """Compute steerability bias from relative log odds data.

            When data is relative to baseline (baseline = 0):
            - rel_A_B = log_odds(B|nudge_A) - log_odds(B|baseline)
            - rel_B_B = log_odds(B|nudge_B) - log_odds(B|baseline) = steerability_B
            - steerability_A = -rel_A_B (since log_odds(A) = -log_odds(B))
            - bias = steerability_B - steerability_A = rel_B_B + rel_A_B
            """
            return rel_B_B + rel_A_B

        # Compute bias values for each row and collect all values for axis limits
        bias_by_row = {}
        sig_bias_by_row = {}
        all_bias_values = []
        for row_key in rows:
            rd = data_by_row[row_key]
            if relative:
                bias_values = [
                    compute_steerability_bias_from_relative(f_A, f_B)
                    for f_A, f_B in zip(rd["f_A_B"], rd["f_B_B"])
                ]
            elif log_odds:
                bias_values = [
                    compute_steerability_bias_from_log_odds(f_A, f_B, f_0)
                    for f_A, f_B, f_0 in zip(rd["f_A_B"], rd["f_B_B"], rd["f_0_B"])
                ]
            else:
                bias_values = [
                    compute_steerability_bias_from_freq(f_A, f_B, f_0)
                    for f_A, f_B, f_0 in zip(rd["f_A_B"], rd["f_B_B"], rd["f_0_B"])
                ]
            bias_by_row[row_key] = bias_values
            sig_bias_by_row[row_key] = rd.get("sig_bias", [])
            all_bias_values.extend(bias_values)

        # Determine x-axis limits with padding
        all_bias = np.array(all_bias_values)
        bias_min, bias_max = np.min(all_bias), np.max(all_bias)
        bias_range = bias_max - bias_min if bias_max != bias_min else 0.1
        bias_padding = bias_range * 0.1
        bias_xlim = (bias_min - bias_padding, bias_max + bias_padding)

        # Bias violin color
        bias_color = "#7B68EE"  # Medium slate blue

        # Draw violin and scatter for each row
        for i, row_key in enumerate(rows):
            bias_values = bias_by_row[row_key]
            sig_bias_flags = sig_bias_by_row[row_key]
            y_pos = y_positions[i]

            # Draw violin if enough data points
            if len(bias_values) >= 2:
                parts = ax_bias.violinplot(
                    [bias_values],
                    positions=[y_pos],
                    vert=False,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    widths=0.7,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(bias_color)
                    pc.set_edgecolor(bias_color)
                    pc.set_alpha(0.3)

            # Draw scatter points with jitter
            n_points = len(bias_values)
            jitter = np.random.uniform(-0.2, 0.2, n_points)
            if show_significance and sig_bias_flags:
                # Color by significance: grey for non-significant
                colors_bias = [
                    bias_color if sig else color_nonsig for sig in sig_bias_flags
                ]
                ax_bias.scatter(
                    bias_values,
                    y_pos + jitter,
                    c=colors_bias,
                    alpha=0.7,
                    s=20,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )
            else:
                ax_bias.scatter(
                    bias_values,
                    y_pos + jitter,
                    color=bias_color,
                    alpha=0.7,
                    s=20,
                    edgecolors="white",
                    linewidths=0.5,
                    zorder=3,
                )

            # Add mean marker
            mean_bias = np.mean(bias_values)
            ax_bias.scatter(
                [mean_bias],
                [y_pos],
                color="black",
                marker="|",
                s=400,
                linewidths=2,
                zorder=5,
            )

        # Configure bias axis
        ax_bias.axvline(x=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_bias.set_xlim(bias_xlim)
        ax_bias.set_ylim(ax.get_ylim())
        ax_bias.set_yticks([])
        ax_bias.invert_yaxis()

        # Steerability bias is always in log odds space
        ax_bias.set_xlabel("Steerability Bias\n(Δ Log₁₀ Odds)", fontsize=10)

        # Style bias axis
        ax_bias.spines["top"].set_visible(False)
        ax_bias.spines["right"].set_visible(False)
        ax_bias.spines["left"].set_visible(False)
        ax_bias.tick_params(axis="x", which="major", labelsize=9)

        # Add grid lines to match main plot
        for i in range(n_rows - 1):
            ax_bias.axhline(
                y=i + 0.5,
                color="lightgray",
                linestyle="-",
                linewidth=0.5,
                alpha=0.5,
                zorder=0,
            )

    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    central_label = "Median" if percentiles else "Mean"

    # Use different legend labels when aggregating across factors
    if row_type == "factors":
        label_A = "Nudged towards A"
        label_B = "Nudged towards B"
    else:
        # When aggregating, A = less preferred at baseline, B = more preferred
        label_A = "Nudge away from baseline pref."
        label_B = "Nudge towards baseline pref."

    legend_elements = [
        Patch(facecolor=color_nudge_A, alpha=0.3, label=label_A),
        Patch(facecolor=color_nudge_B, alpha=0.3, label=label_B),
        Line2D(
            [0],
            [0],
            marker="|",
            color=color_nudge_A,
            markersize=14,
            linewidth=0,
            markeredgewidth=2.5,
            label=f"{central_label} (nudged)",
        ),
    ]

    # Only show baseline in legend when not in relative mode
    if not relative:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor=color_baseline,
                markeredgecolor="black",
                markersize=10,
                label=f"{central_label} Baseline f₀(B)",
            )
        )
        # Check if any row has baseline variation to show range legend
        # Only show if not in single_model mode
        has_any_baseline_range = not single_model and any(
            np.min(rd["f_0_B"]) != np.max(rd["f_0_B"]) for rd in data_by_row.values()
        )
        if has_any_baseline_range:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=color_baseline,
                    linewidth=2,
                    marker="|",
                    markersize=8,
                    markeredgewidth=2,
                    label="Baseline Range (min-max)",
                )
            )

    if percentiles:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color="gray",
                linewidth=2.5,
                alpha=0.8,
                label="IQR (25-75%)",
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Add light horizontal grid lines to separate rows
    for i in range(n_rows - 1):
        ax.axhline(
            y=i + 0.5,
            color="lightgray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=0,
        )

    # Invert y-axis so first row is at top
    ax.invert_yaxis()

    # Adjust margins to make room for option labels on left and right
    if show_bias:
        # With bias column, we need different margins and more space between plots
        plt.subplots_adjust(left=0.10, right=0.98, wspace=0.15)
        plt.tight_layout(rect=[0.06, 0, 1.0, 1])
    else:
        plt.subplots_adjust(left=0.12, right=0.88)
        plt.tight_layout(rect=[0.08, 0, 0.92, 1])

    # Save figure
    if output_path:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved plot to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Create violin plots showing steerability distributions by factor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Discover all results from results directories
    uv run python -m choices.analysis.plot_steerability --results-dirs results

    # Specify multiple results directories
    uv run python -m choices.analysis.plot_steerability \\
        --results-dirs results results_anthropic

    # Filter by factors
    uv run python -m choices.analysis.plot_steerability \\
        --results-dirs results \\
        --factors gender age_group wealth
        """,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="List of results directories to search",
    )

    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="List of factors to include (default: all discovered)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="List of models to include (default: all discovered)",
    )

    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="List of nudge types to include (default: all discovered)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: steerability_violins_<rows>.pdf)",
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
        default=[12, None],
        help="Figure size (width height). Height auto-calculated if not provided.",
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
        "--percentiles",
        action="store_true",
        help="Show median and IQR (25-75%%) instead of mean",
    )

    parser.add_argument(
        "--relative",
        action="store_true",
        help="Compute effects relative to baseline (subtract baseline from each condition)",
    )

    parser.add_argument(
        "--rows",
        type=str,
        choices=["factors", "models", "nudges", "baseline"],
        default="factors",
        help="What to show as rows: factors (default), models, nudges, or baseline. "
        "Log odds space is forced for non-factor rows. "
        "For 'baseline', rows are bins of baseline bias magnitude.",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of quantile-based bins when --rows=baseline (default: 5). "
        "Each bin will have approximately equal sample counts.",
    )

    parser.add_argument(
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include (e.g., 'before', 'none', 'after', 'low', 'medium', 'high')",
    )

    parser.add_argument(
        "--bias",
        action="store_true",
        help="Add a column showing steerability bias distribution as violin plot with data points",
    )

    parser.add_argument(
        "--significance",
        action="store_true",
        help="Show non-significant data points in grey (uses z-test for nudges, Wald test for bias)",
    )

    args = parser.parse_args()

    # Determine output path (append row type to default filename)
    if args.output:
        output_path = args.output
    else:
        output_path = f"steerability_violins_{args.rows}.pdf"

    # Force log odds and relative mode for model/nudge rows (but not baseline)
    use_log_odds = args.log_odds
    use_relative = args.relative
    if args.rows in ("models", "nudges"):
        if not use_log_odds:
            print(f"Note: Forcing log odds space for --rows={args.rows}")
            use_log_odds = True
        if not use_relative:
            print(f"Note: Forcing relative mode for --rows={args.rows}")
            use_relative = True

    # Print header
    print("=" * 70)
    print("Steerability Violin Plot")
    print("=" * 70)
    print(f"Results directories: {args.results_dirs}")
    print(f"Rows: {args.rows}")
    if args.rows == "baseline":
        print(f"Number of bins: {args.n_bins}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning_conditions:
        print(f"Reasoning filter: {args.reasoning_conditions}")
    print(f"Space: {'Log Odds' if use_log_odds else 'Frequency'}")
    print(f"Relative: {'Yes' if use_relative else 'No'}")
    print(f"Statistics: {'Median + IQR' if args.percentiles else 'Mean'}")
    print(f"Bias column: {'Yes' if args.bias else 'No'}")
    print(f"Significance: {'Yes' if args.significance else 'No'}")
    print(f"Output: {output_path}")
    print("=" * 70)
    print()

    # Discover and compute results
    print("Discovering experiments...")
    experiments = discover_experiments(
        args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if not experiments:
        print("No experiments found matching the filters.")
        return

    print(f"Found {len(experiments)} experiment(s)")
    print()

    # Compute frequency results
    print("Computing frequency results...")
    results = compute_all_results(
        args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if not results:
        print("No results found.")
        return

    print(f"Computed {len(results)} result(s)")

    # Filter by reasoning condition if specified
    if args.reasoning_conditions:
        results = [
            r for r in results if r.reasoning_condition in args.reasoning_conditions
        ]
        print(f"After reasoning filter: {len(results)} result(s)")

    if not results:
        print("No results after filtering.")
        return

    print()

    # Collect data based on row type
    if args.rows == "factors":
        data_by_row = collect_data_by_factor(results)
        row_type_label = "Factor"
    elif args.rows == "models":
        data_by_row = collect_data_by_model(results)
        row_type_label = "Model"
    elif args.rows == "nudges":
        data_by_row = collect_data_by_nudge_type(results)
        row_type_label = "Nudge Type"
    else:  # baseline
        data_by_row = collect_data_by_baseline_bin(results, n_bins=args.n_bins)
        row_type_label = "Baseline Bias"

    # Print summary (always in frequency space for clarity)
    print(f"Data Summary by {row_type_label} (Frequency Space):")
    print("-" * 70)
    for row_key in sorted(data_by_row.keys()):
        rd = data_by_row[row_key]
        n_experiments = len(rd["f_0_B"])
        avg_baseline = np.mean(rd["f_0_B"])
        avg_nudge_A = np.mean(rd["f_A_B"])
        avg_nudge_B = np.mean(rd["f_B_B"])
        print(
            f"  {row_key:<25} n={n_experiments:>3}  "
            f"Baseline={avg_baseline:.3f}  "
            f"Nudge→A={avg_nudge_A:.3f}  "
            f"Nudge→B={avg_nudge_B:.3f}"
        )
    print()

    # Transform to log odds if needed (do this before relative transform)
    if use_log_odds:
        print("Transforming data to log odds space...")
        data_by_row = transform_data_to_log_odds(data_by_row)
        print()

    # Transform to relative values if requested (after log odds transform)
    if use_relative:
        print("Computing effects relative to baseline...")
        data_by_row = transform_data_to_relative(data_by_row)
        print()

    # Check if we have a single model (don't show baseline range in that case)
    unique_models = set(r.model for r in results)
    is_single_model = len(unique_models) == 1
    if is_single_model:
        print(f"Single model detected: {list(unique_models)[0]}")
        print()

    # Create plot
    figsize = (args.figsize[0], args.figsize[1] if args.figsize[1] else None)
    fig = create_steerability_violin_plot(
        data_by_row,
        output_path=output_path,
        title=args.title,
        figsize=figsize,
        log_odds=use_log_odds,
        percentiles=args.percentiles,
        relative=use_relative,
        row_type=args.rows,
        show_bias=args.bias,
        show_significance=args.significance,
        single_model=is_single_model,
    )

    if fig is None:
        return

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
