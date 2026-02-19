#!/usr/bin/env python3
"""
Plot compliance category vs actual steerability (effect size) at the edge level.

For each edge (case), the majority compliance category is determined from its
nudged traces, and the signed effect is computed as the change in probability
of choosing the nudged option (positive = nudge worked, negative = backfire).

Produces a combined strip + violin plot with compliance categories on the
x-axis and signed effect on the y-axis.

Usage:
    uv run python -m choices.analysis.reasoning_traces.plot_compliance \
        --input compliance_results.json

    # Custom output
    uv run python -m choices.analysis.reasoning_traces.plot_compliance \
        --input compliance_results.json \
        --output compliance_vs_effect.png

    # Colour by factor, nudge type, or model
    uv run python -m choices.analysis.reasoning_traces.plot_compliance \
        --input compliance_results.json \
        --color-by factor

    uv run python -m choices.analysis.reasoning_traces.plot_compliance \
        --input compliance_results.json \
        --color-by model
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.reasoning_traces.compliance_classification import (
    ComplianceCategory,
)
from choices.analysis.utils import (
    PLOTS_OUTPUT_DIR,
    get_factor_color,
    get_model_color,
    get_nudge_color,
)

# ── Style ────────────────────────────────────────────────────────────────────

FONT_SIZES = {
    "title": 16,
    "axes_label": 14,
    "tick_label": 12,
    "legend": 11,
    "annotation": 9,
}

# Ordered from most-compliant to least-engaged
CATEGORY_ORDER = [
    ComplianceCategory.GOING_ALONG.value,
    ComplianceCategory.MENTIONING_NO_RESOLUTION.value,
    ComplianceCategory.CLAIMING_TO_IGNORE.value,
    ComplianceCategory.REJECTING.value,
    ComplianceCategory.NOT_MENTIONING.value,
]

CATEGORY_DISPLAY_NAMES = {
    "going_along": "Going along",
    "mentioning_no_resolution": "Mentioning,\nno resolution",
    "claiming_to_ignore": "Claiming\nto ignore",
    "rejecting": "Rejecting",
    "not_mentioning": "Not\nmentioning",
}

CATEGORY_COLORS = {
    "going_along": "#2ecc71",
    "mentioning_no_resolution": "#f39c12",
    "claiming_to_ignore": "#3498db",
    "rejecting": "#e74c3c",
    "not_mentioning": "#95a5a6",
}

# Color-getter dispatch for --color-by options
_COLOR_GETTERS: dict[str, callable] = {
    "factor": get_factor_color,
    "nudge_type": get_nudge_color,
    "model": get_model_color,
}


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.size": FONT_SIZES["tick_label"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axes_label"],
            "xtick.labelsize": FONT_SIZES["tick_label"],
            "ytick.labelsize": FONT_SIZES["tick_label"],
            "legend.fontsize": FONT_SIZES["legend"],
            "figure.titlesize": FONT_SIZES["title"],
        }
    )


# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class EdgePoint:
    """One edge's aggregated compliance + effect data."""

    edge_key: str
    majority_category: str
    signed_effect: float  # nudged_freq - baseline_freq
    n_traces: int
    unanimity: float  # fraction of traces in the majority category
    factor: str
    nudge_type: str
    model: str


def load_compliance_data(filepath: str) -> tuple[dict, list[dict]]:
    """Load compliance-classification output JSON."""
    with open(filepath) as f:
        data = json.load(f)
    metadata = data.get("original_metadata") or data.get("metadata", {})
    return metadata, data.get("cases", [])


def majority_compliance(traces: list[dict]) -> tuple[str, float]:
    """Return (majority_category, unanimity) from a list of nudged traces."""
    categories = [
        t["compliance"]["compliance_category"]
        for t in traces
        if t.get("compliance") and t["compliance"].get("compliance_category")
    ]
    if not categories:
        return "unknown", 0.0

    counts = Counter(categories)
    majority, majority_count = counts.most_common(1)[0]
    return majority, majority_count / len(categories)


def build_edge_points(cases: list[dict]) -> list[EdgePoint]:
    """Build one EdgePoint per case from compliance results."""
    points: list[EdgePoint] = []
    for case in cases:
        nudged_traces = [
            t
            for t in case.get("condition_b_traces", [])
            if t.get("compliance") is not None
        ]
        if not nudged_traces:
            continue

        cat, unanimity = majority_compliance(nudged_traces)
        if cat == "unknown":
            continue

        # Signed effect: positive means nudge pushed in intended direction
        signed_effect = case.get("nudged_freq", 0.0) - case.get("baseline_freq", 0.0)

        points.append(
            EdgePoint(
                edge_key=case.get("edge_key", ""),
                majority_category=cat,
                signed_effect=signed_effect,
                n_traces=len(nudged_traces),
                unanimity=unanimity,
                factor=case.get("factor", ""),
                nudge_type=case.get("nudge_type", ""),
                model=case.get("model", ""),
            )
        )
    return points


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_compliance_vs_effect(
    points: list[EdgePoint],
    output_path: str,
    color_by: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 7),
):
    """
    Create a strip + violin plot of compliance category vs signed effect.

    Args:
        points: List of EdgePoint objects.
        output_path: Where to save the figure.
        color_by: Optional grouping for point colours (``"factor"``, ``"nudge_type"``, or ``"model"``).
        title: Optional custom title.
        figsize: Figure size as ``(width, height)`` in inches.
    """
    setup_plot_style()

    # Group points by category (in display order)
    categories_present = [
        c for c in CATEGORY_ORDER if any(p.majority_category == c for p in points)
    ]
    cat_to_x = {c: i for i, c in enumerate(categories_present)}

    fig, ax = plt.subplots(figsize=figsize)

    # ── Violin bodies ────────────────────────────────────────────────────
    for cat in categories_present:
        effects = [p.signed_effect * 100 for p in points if p.majority_category == cat]
        if len(effects) < 2:
            continue
        x_pos = cat_to_x[cat]
        parts = ax.violinplot(
            effects,
            positions=[x_pos],
            showextrema=False,
            showmedians=False,
            widths=0.7,
        )
        for body in parts["bodies"]:
            body.set_facecolor(CATEGORY_COLORS.get(cat, "#cccccc"))
            body.set_alpha(0.25)
            body.set_edgecolor("none")

    # ── Strip points ─────────────────────────────────────────────────────
    color_getter = _COLOR_GETTERS.get(color_by) if color_by else None

    # Collect legend handles
    legend_handles: dict[str, plt.Artist] = {}
    rng = np.random.default_rng(42)

    for pt in points:
        x = cat_to_x.get(pt.majority_category)
        if x is None:
            continue

        jitter = rng.uniform(-0.2, 0.2)
        alpha = 0.2 + 0.3 * pt.unanimity

        if color_getter is not None:
            group_val = getattr(pt, color_by)
            color = color_getter(group_val)
            label = group_val.replace("_", " ").title()
        else:
            color = CATEGORY_COLORS.get(pt.majority_category, "#555555")
            label = None

        scatter = ax.scatter(
            x + jitter,
            pt.signed_effect * 100,
            s=30,
            color=color,
            alpha=alpha,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

        if label and label not in legend_handles:
            legend_handles[label] = scatter

    # ── Medians ───────────────────────────────────────────────────────────
    for cat in categories_present:
        effects = [p.signed_effect * 100 for p in points if p.majority_category == cat]
        x_pos = cat_to_x[cat]
        if effects:
            median = np.median(effects)
            ax.plot(
                [x_pos - 0.25, x_pos + 0.25],
                [median, median],
                color="black",
                linewidth=2,
                zorder=4,
            )

    # ── Reference line at zero ───────────────────────────────────────────
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

    # ── Axes ─────────────────────────────────────────────────────────────
    ax.set_xticks(range(len(categories_present)))
    ax.set_xticklabels([CATEGORY_DISPLAY_NAMES.get(c, c) for c in categories_present])
    ax.set_ylabel("Effect on P(influenced option)  (pp)")
    ax.set_xlabel("")

    if title is not None:
        ax.set_title(title)

    if legend_handles:
        ax.legend(
            legend_handles.values(),
            legend_handles.keys(),
            loc="upper right",
            framealpha=0.9,
        )

    # ── Count annotations (placed after ylim is settled) ─────────────────
    ylim = ax.get_ylim()
    for cat in categories_present:
        effects = [p.signed_effect * 100 for p in points if p.majority_category == cat]
        x_pos = cat_to_x[cat]
        if effects:
            ax.text(
                x_pos,
                ylim[0] + (ylim[1] - ylim[0]) * 0.02,
                f"n={len(effects)}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["tick_label"],
                color="0.4",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot compliance category vs actual steerability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --input compliance_results.json\n"
            "  %(prog)s --input compliance_results.json --color-by factor\n"
            "  %(prog)s --input compliance_results.json --color-by nudge_type --pdf"
        ),
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSON file with compliance annotations",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: <plots_dir>/compliance_vs_effect.<fmt>)",
    )
    parser.add_argument(
        "--color-by",
        choices=["factor", "nudge_type", "model"],
        default=None,
        help="Colour points by this grouping variable",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the plot title",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Figure size in inches, e.g. --figsize 10 6",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save as PDF instead of PNG",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading compliance data from {args.input}...")
    metadata, cases = load_compliance_data(args.input)
    print(f"Loaded {len(cases)} cases")

    # Build edge-level points
    points = build_edge_points(cases)
    print(f"Built {len(points)} edge-level points")

    # Summary
    cat_counts = Counter(p.majority_category for p in points)
    for cat in CATEGORY_ORDER:
        if cat in cat_counts:
            effects = [p.signed_effect for p in points if p.majority_category == cat]
            median_eff = np.median(effects) * 100
            print(
                f"  {cat}: {cat_counts[cat]} edges, "
                f"median effect = {median_eff:+.1f}pp"
            )

    # Output path
    fmt = "pdf" if args.pdf else "png"
    output_path = args.output
    if not output_path:
        out_dir = Path(PLOTS_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"compliance_vs_effect.{fmt}")

    title = None if args.no_title else "Compliance Category vs Actual Steerability"

    kwargs = {}
    if args.figsize:
        kwargs["figsize"] = tuple(args.figsize)

    plot_compliance_vs_effect(
        points,
        output_path,
        color_by=args.color_by,
        title=title,
        **kwargs,
    )


if __name__ == "__main__":
    main()
