#!/usr/bin/env python3
"""
Plot rationale mention and acted-on rates across multiple result files / conditions.

Produces a horizontal bar chart with rationale codes on the y-axis and grouped
bars for each input source (file + condition combination).

Each input source is specified as a triple ``file,condition,label`` where:
- ``file`` is the path to a rationale-detection JSON output,
- ``condition`` is one of ``both``, ``baseline``, ``nudged``, or a specific
  nudge type name (e.g. ``survey_preference``) to select only nudged traces
  from cases with that nudge type,
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
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.reasoning_traces.rationale_detection import (
    CONDITION_KEYS,
    RATIONALE_CODES,
    is_nudge_type_condition,
)
from choices.analysis.utils import PLOTS_OUTPUT_DIR

# ── Style ────────────────────────────────────────────────────────────────────

FONT_SIZES = {
    "title": 16,
    "base": 14,
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
    "identity_based": "Identity-based",
    "influence_resistance": "Influence resistance",
    "random_or_arbitrary": "Random / arbitrary",
    "task_compliance": "Task compliance",
    "feels_right": "Feels right",
    "legal_compliance": "Legal / rule compliance",
    "honesty_integrity": "Honesty / integrity",
    "self_interest": "Self-interest",
    "harm_care_safety": "Harm / care / safety",
    "deontology_duty": "Deontology / duty",
    "context": "Contextual information",
    "utilitarian_consequentialist": "Utilitarian / consequentialist",
    "other": "Other",
}


def setup_plot_style():
    base = FONT_SIZES["base"]
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base,
            "axes.labelsize": base,
            "xtick.labelsize": base,
            "ytick.labelsize": base,
            "legend.fontsize": base,
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
    if is_nudge_type_condition(condition):
        return ("condition_b_traces",)
    keys = CONDITION_KEYS.get(condition)
    if keys is None:
        raise ValueError(
            f"Unknown condition {condition!r}; choose from {list(CONDITION_KEYS)}"
        )
    return keys


def collect_traces(cases: list[dict], condition: str) -> list[dict]:
    """Return all trace dicts matching *condition* that have rationale annotations.

    If *condition* is a nudge type name (e.g. ``"survey_preference"``), only
    nudged traces from cases whose ``nudge_type`` matches are returned.
    """
    keys = _trace_condition_keys(condition)
    nudge_type_filter = condition if is_nudge_type_condition(condition) else None
    traces: list[dict] = []
    for case in cases:
        if nudge_type_filter and case.get("nudge_type") != nudge_type_filter:
            continue
        for key in keys:
            for trace in case.get(key, []):
                if trace.get("rationales") is not None:
                    traces.append(trace)
    return traces


@dataclass
class RationaleRate:
    """Rate and 95 % Wilson-score CI for a single rationale."""

    rate: float
    ci_low: float
    ci_high: float
    count: int
    n: int


def _wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = count / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def compute_rationale_rates(
    traces: list[dict],
    metric: str = "mentioned",
) -> dict[str, RationaleRate]:
    """
    Compute the rate and 95 % CI of each rationale across *traces*.

    Args:
        traces: List of trace dicts with ``rationales`` annotations.
        metric: One of
            ``"mentioned"`` – mentioned_but_not_acted_on OR mentioned_and_acted_on,
            ``"acted_on"`` – only mentioned_and_acted_on,
            ``"primary"``  – fraction where this rationale is the primary one.

    Returns:
        Dict mapping rationale code to :class:`RationaleRate`.
    """
    n = len(traces)
    if n == 0:
        return {
            code: RationaleRate(rate=0.0, ci_low=0.0, ci_high=0.0, count=0, n=0)
            for code in RATIONALE_CODES
        }

    rates: dict[str, RationaleRate] = {}
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
        ci_low, ci_high = _wilson_ci(count, n)
        rates[code] = RationaleRate(
            rate=count / n, ci_low=ci_low, ci_high=ci_high, count=count, n=n
        )

    return rates


def _compute_merged_rate(
    traces: list[dict],
    codes: set[str],
    metric: str,
) -> RationaleRate:
    """Compute a single rate for traces matching *any* code in *codes*."""
    n = len(traces)
    if n == 0:
        return RationaleRate(rate=0.0, ci_low=0.0, ci_high=0.0, count=0, n=0)

    count = 0
    for t in traces:
        rationales = t["rationales"]
        for code in codes:
            if metric == "primary":
                if rationales.get("primary_rationale") == code:
                    count += 1
                    break
            elif metric == "acted_on":
                if rationales.get(code, {}).get("status") == "mentioned_and_acted_on":
                    count += 1
                    break
            else:
                if rationales.get(code, {}).get("status") in (
                    "mentioned_but_not_acted_on",
                    "mentioned_and_acted_on",
                ):
                    count += 1
                    break

    ci_low, ci_high = _wilson_ci(count, n)
    return RationaleRate(
        rate=count / n, ci_low=ci_low, ci_high=ci_high, count=count, n=n
    )


# ── Source parsing ───────────────────────────────────────────────────────────


def parse_source(source_str: str) -> tuple[str, str, str]:
    """Parse a ``file,condition,label`` string.

    *condition* may be a built-in key (``both``, ``baseline``, ``nudged``) or a
    specific nudge type name (e.g. ``survey_preference``), in which case only
    nudged traces from cases with that nudge type are selected.
    """
    parts = [p.strip().strip('"').strip("'") for p in source_str.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected file,condition,label but got: {source_str!r}"
        )
    filepath, condition, label = parts
    return filepath, condition, label


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_rationale_comparison(
    sources: list[tuple[str, str, str]],
    output_path: str,
    metric: str = "mentioned",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    threshold: float | None = None,
    show_pct: bool = False,
):
    """
    Create a faceted grid of horizontal bar charts comparing rationale rates.

    Each source gets its own panel in a grid layout, making the figure compact
    enough for paper inclusion even with many sources.

    Args:
        sources: List of (filepath, condition, label) tuples.
        output_path: Where to save the figure.
        metric: ``"mentioned"``, ``"acted_on"``, or ``"primary"``.
        title: Optional custom title (shown as ``suptitle``).
        figsize: Figure size as ``(width, height)`` in inches.
        threshold: If set, only keep rationales where at least one source has
            a rate >= this percentage.  All others are merged into "Other".
        show_pct: If True, display the percentage value at the end of each bar.
    """
    setup_plot_style()

    # Collect rates and traces for each source
    all_rates: list[dict[str, RationaleRate]] = []
    all_traces: list[list[dict]] = []
    labels: list[str] = []
    trace_counts: list[int] = []

    for filepath, condition, label in sources:
        _meta, cases = load_rationale_data(filepath)
        traces = collect_traces(cases, condition)
        rates = compute_rationale_rates(traces, metric=metric)
        all_rates.append(rates)
        all_traces.append(traces)
        labels.append(label)
        trace_counts.append(len(traces))

    n_sources = len(sources)

    # ── Optional threshold filtering ──────────────────────────────────────
    active_codes: list[str] = list(RATIONALE_CODES)

    if threshold is not None:
        codes_above = {
            code
            for code in RATIONALE_CODES
            if code != "other"
            and any(r[code].rate * 100 >= threshold for r in all_rates)
        }
        codes_below = set(RATIONALE_CODES) - codes_above - {"other"}

        if codes_below:
            codes_to_merge = codes_below | {"other"}
            for i in range(n_sources):
                all_rates[i]["other"] = _compute_merged_rate(
                    all_traces[i], codes_to_merge, metric
                )
                for code in codes_below:
                    del all_rates[i][code]

        active_codes = list(codes_above) + ["other"]

    # Sort rationales by the average rate across sources (descending),
    # keeping only those with at least one non-zero bar.
    avg_rates = {
        code: np.mean([r[code].rate for r in all_rates]) for code in active_codes
    }
    sorted_codes = [
        c
        for c in sorted(active_codes, key=lambda c: avg_rates[c], reverse=True)
        if any(r[c].rate > 0 for r in all_rates)
    ]
    if threshold is not None and "other" not in sorted_codes:
        sorted_codes.append("other")
    if threshold is not None and "other" in sorted_codes:
        sorted_codes = [c for c in sorted_codes if c != "other"] + ["other"]

    display_names = [RATIONALE_DISPLAY_NAMES.get(c, c) for c in sorted_codes]

    # ── Faceted grid layout ───────────────────────────────────────────────
    n_rationales = len(sorted_codes)
    y = np.arange(n_rationales)

    ncols = min(n_sources, 4)
    nrows = math.ceil(n_sources / ncols)

    effective_figsize = figsize or (
        3.5 * ncols,
        0.35 * n_rationales * nrows + 1,
    )
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharey=True,
        sharex=True,
        figsize=effective_figsize,
    )
    axes_flat = np.atleast_1d(axes).flatten()

    # Global x-limit across all sources
    x_max = max(
        r[code].ci_high * 100
        for rates in all_rates
        for code in sorted_codes
        for r in [rates]
    )

    metric_label = {
        "mentioned": "Mention rate",
        "acted_on": "Acted-on rate",
        "primary": "Primary rationale rate",
    }.get(metric, metric)

    for i, (rates, label, n_traces) in enumerate(zip(all_rates, labels, trace_counts)):
        ax = axes_flat[i]
        values = [rates[code].rate * 100 for code in sorted_codes]

        ci_lo = [rates[code].ci_low * 100 for code in sorted_codes]
        ci_hi = [rates[code].ci_high * 100 for code in sorted_codes]
        xerr_low = [max(0.0, v - lo) for v, lo in zip(values, ci_lo)]
        xerr_high = [max(0.0, hi - v) for v, hi in zip(values, ci_hi)]

        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        ax.barh(
            y,
            values,
            height=0.7,
            xerr=[xerr_low, xerr_high],
            error_kw={"linewidth": 0.8, "capsize": 2, "color": "0.4"},
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )

        if show_pct:
            for yi, (v, hi) in enumerate(zip(values, ci_hi)):
                ax.text(
                    hi + x_max * 0.01,
                    yi,
                    f"{v:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=FONT_SIZES["base"],
                    color="0.3",
                )

        ax.set_title(f"{label} (n={n_traces})", fontsize=FONT_SIZES["base"])
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_xlim(0, x_max * (1.18 if show_pct else 1.08))

    # Shared axis configuration
    axes_flat[0].set_yticks(y)
    axes_flat[0].set_yticklabels(display_names)
    axes_flat[0].invert_yaxis()
    for idx in range(n_sources):
        panel = axes_flat[idx]
        if idx // ncols == nrows - 1:
            panel.set_xlabel(f"{metric_label} (%)", fontsize=FONT_SIZES["base"])

    # Hide unused panels
    for j in range(n_sources, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if title is not None:
        fig.suptitle(title, fontsize=FONT_SIZES["title"], y=1.01)

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
            "condition is one of: both, baseline, nudged, or a specific "
            "nudge type name (e.g. survey_preference).  "
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
        "--threshold",
        "-t",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Minimum percentage for a rationale to be shown individually. "
            "Rationales where no source reaches this threshold are merged "
            "into 'Other'."
        ),
    )
    parser.add_argument(
        "--show-pct",
        action="store_true",
        help="Display percentage values next to each bar",
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

    metric_label = {
        "mentioned": "Mention rate",
        "acted_on": "Acted-on rate",
        "primary": "Primary rationale rate",
    }.get(args.metric, args.metric)

    title = None if args.no_title else f"Rationale {metric_label} Comparison"

    kwargs = {}
    if args.figsize:
        kwargs["figsize"] = tuple(args.figsize)

    plot_rationale_comparison(
        sources,
        output_path,
        metric=args.metric,
        title=title,
        threshold=args.threshold,
        show_pct=args.show_pct,
        **kwargs,
    )


if __name__ == "__main__":
    main()
