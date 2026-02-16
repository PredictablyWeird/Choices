#!/usr/bin/env python3
"""
Plot rationale mention and acted-on rates across multiple result files / conditions.

Produces a horizontal bar chart with rationale codes on the y-axis and grouped
bars for each input source (file + condition combination).

Each input source is specified as a triple ``file,condition,label`` where:
- ``file`` is the path to a rationale-detection JSON output,
- ``condition`` is one of ``both``, ``baseline``, or ``nudged``,
- ``label`` is the legend label for this source.

Usage:
    uv run python -m choices.analysis.reasoning_traces.plot_rationales \
        --source rationale_results.json,baseline,"Baseline" \
        --source rationale_results.json,nudged,"Nudged" \
        --output rationale_comparison.png

    # Or compare two different model files
    uv run python -m choices.analysis.reasoning_traces.plot_rationales \
        --source rationale_gpt52.json,both,"GPT-5.2" \
        --source rationale_claude.json,both,"Claude" \
        --metric acted_on
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.reasoning_traces.rationale_detection import (
    CONDITION_KEYS,
    RATIONALE_CODES,
)
from choices.analysis.utils import PLOTS_OUTPUT_DIR

# ── Style ────────────────────────────────────────────────────────────────────

FONT_SIZES = {
    "title": 16,
    "axes_label": 14,
    "tick_label": 12,
    "legend": 11,
    "annotation": 9,
}

COLOR_PALETTE = [
    "#3498db",  # blue
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#9b59b6",  # purple
    "#f39c12",  # orange
    "#1abc9c",  # teal
    "#e67e22",  # dark orange
    "#34495e",  # dark grey-blue
]

RATIONALE_DISPLAY_NAMES = {
    "save_more_lives": "Save more lives",
    "life_years_or_potential": "Life-years / potential",
    "quality_of_life": "Quality of life",
    "equal_moral_worth": "Equal moral worth",
    "anti_discrimination": "Anti-discrimination",
    "equity_for_disadvantaged": "Equity for disadvantaged",
    "societal_contribution": "Societal contribution",
    "external_evidence_or_authority": "External evidence / authority",
    "social_norms": "Social norms",
    "emotional_or_compassion": "Emotional / compassion",
    "random_or_arbitrary": "Random / arbitrary",
    "task_compliance": "Task compliance",
    "feels_right": "Feels right",
    "context": "Context (nudge reference)",
    "other": "Other",
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


# ── Data helpers ─────────────────────────────────────────────────────────────


def load_rationale_data(filepath: str) -> tuple[dict, list[dict]]:
    """Load rationale-detection output JSON and return (metadata, cases)."""
    with open(filepath) as f:
        data = json.load(f)
    metadata = data.get("original_metadata") or data.get("metadata", {})
    return metadata, data.get("cases", [])


def _trace_condition_keys(condition: str) -> tuple[str, ...]:
    keys = CONDITION_KEYS.get(condition)
    if keys is None:
        raise ValueError(
            f"Unknown condition {condition!r}; choose from {list(CONDITION_KEYS)}"
        )
    return keys


def collect_traces(cases: list[dict], condition: str) -> list[dict]:
    """Return all trace dicts matching *condition* that have rationale annotations."""
    keys = _trace_condition_keys(condition)
    traces: list[dict] = []
    for case in cases:
        for key in keys:
            for trace in case.get(key, []):
                if trace.get("rationales") is not None:
                    traces.append(trace)
    return traces


def compute_rationale_rates(
    traces: list[dict],
    metric: str = "mentioned",
) -> dict[str, float]:
    """
    Compute the rate of each rationale across *traces*.

    Args:
        traces: List of trace dicts with ``rationales`` annotations.
        metric: One of
            ``"mentioned"`` – mentioned_but_not_acted_on OR mentioned_and_acted_on,
            ``"acted_on"`` – only mentioned_and_acted_on,
            ``"primary"``  – fraction where this rationale is the primary one.

    Returns:
        Dict mapping rationale code to rate (0–1).
    """
    n = len(traces)
    if n == 0:
        return {code: 0.0 for code in RATIONALE_CODES}

    rates: dict[str, float] = {}
    for code in RATIONALE_CODES:
        if metric == "primary":
            count = sum(
                1 for t in traces if t["rationales"].get("primary_rationale") == code
            )
        elif metric == "acted_on":
            count = sum(
                1
                for t in traces
                if t["rationales"].get(code, {}).get("status")
                == "mentioned_and_acted_on"
            )
        else:  # mentioned (any mention)
            count = sum(
                1
                for t in traces
                if t["rationales"].get(code, {}).get("status")
                in ("mentioned_but_not_acted_on", "mentioned_and_acted_on")
            )
        rates[code] = count / n

    return rates


# ── Source parsing ───────────────────────────────────────────────────────────


def parse_source(source_str: str) -> tuple[str, str, str]:
    """Parse a ``file,condition,label`` string."""
    parts = [p.strip().strip('"').strip("'") for p in source_str.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected file,condition,label but got: {source_str!r}"
        )
    filepath, condition, label = parts
    if condition not in CONDITION_KEYS:
        raise argparse.ArgumentTypeError(
            f"Unknown condition {condition!r}; choose from {list(CONDITION_KEYS)}"
        )
    return filepath, condition, label


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_rationale_comparison(
    sources: list[tuple[str, str, str]],
    output_path: str,
    metric: str = "mentioned",
    title: str | None = None,
):
    """
    Create a horizontal grouped bar chart comparing rationale rates.

    Args:
        sources: List of (filepath, condition, label) tuples.
        output_path: Where to save the figure.
        metric: ``"mentioned"``, ``"acted_on"``, or ``"primary"``.
        title: Optional custom title.
    """
    setup_plot_style()

    # Collect rates for each source
    all_rates: list[dict[str, float]] = []
    labels: list[str] = []
    trace_counts: list[int] = []

    for filepath, condition, label in sources:
        _meta, cases = load_rationale_data(filepath)
        traces = collect_traces(cases, condition)
        rates = compute_rationale_rates(traces, metric=metric)
        all_rates.append(rates)
        labels.append(label)
        trace_counts.append(len(traces))

    n_sources = len(sources)
    n_rationales = len(RATIONALE_CODES)

    # Sort rationales by the average rate across sources (descending)
    avg_rates = {
        code: np.mean([r[code] for r in all_rates]) for code in RATIONALE_CODES
    }
    sorted_codes = sorted(RATIONALE_CODES, key=lambda c: avg_rates[c], reverse=True)

    display_names = [RATIONALE_DISPLAY_NAMES.get(c, c) for c in sorted_codes]

    # Bar geometry
    bar_height = 0.8 / n_sources
    y = np.arange(n_rationales)

    fig, ax = plt.subplots(figsize=(12, max(7, n_rationales * 0.55)))

    for i, (rates, label, n_traces) in enumerate(zip(all_rates, labels, trace_counts)):
        offsets = y - 0.4 + bar_height * (i + 0.5)
        values = [rates[code] * 100 for code in sorted_codes]
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        bars = ax.barh(
            offsets,
            values,
            height=bar_height * 0.9,
            label=f"{label} (n={n_traces})",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        # Value labels for bars > 3%
        for bar in bars:
            width = bar.get_width()
            if width > 3:
                ax.text(
                    width + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=FONT_SIZES["annotation"],
                    color=color,
                )

    # Axes
    metric_label = {
        "mentioned": "Mention rate",
        "acted_on": "Acted-on rate",
        "primary": "Primary rationale rate",
    }.get(metric, metric)

    ax.set_xlabel(f"{metric_label} (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Rationale {metric_label} Comparison")

    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)  # room for labels

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Plot rationale rates across multiple result files / conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s \\\n"
            '    --source rationale_results.json,baseline,"Baseline" \\\n'
            '    --source rationale_results.json,nudged,"Nudged"\n'
            "\n"
            "  %(prog)s \\\n"
            '    --source gpt52_rationales.json,both,"GPT-5.2" \\\n'
            '    --source claude_rationales.json,both,"Claude" \\\n'
            "    --metric acted_on --pdf"
        ),
    )
    parser.add_argument(
        "--source",
        "-s",
        action="append",
        required=True,
        metavar="FILE,CONDITION,LABEL",
        help=(
            "A source triple: file,condition,label.  "
            "condition is one of: both, baseline, nudged.  "
            "Can be repeated for multiple sources."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: <plots_dir>/rationale_comparison.<fmt>)",
    )
    parser.add_argument(
        "--metric",
        "-m",
        choices=["mentioned", "acted_on", "primary"],
        default="mentioned",
        help="Which metric to plot (default: mentioned)",
    )
    parser.add_argument(
        "--title",
        "-t",
        default=None,
        help="Custom plot title",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save as PDF instead of PNG",
    )

    args = parser.parse_args()

    # Parse sources
    sources = [parse_source(s) for s in args.source]
    print(f"Plotting {len(sources)} source(s), metric={args.metric}")
    for fp, cond, label in sources:
        print(f"  {label}: {fp} [{cond}]")

    # Output path
    fmt = "pdf" if args.pdf else "png"
    output_path = args.output
    if not output_path:
        out_dir = Path(PLOTS_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"rationale_comparison.{fmt}")

    plot_rationale_comparison(
        sources,
        output_path,
        metric=args.metric,
        title=args.title,
    )


if __name__ == "__main__":
    main()
