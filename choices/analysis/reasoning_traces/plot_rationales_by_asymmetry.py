#!/usr/bin/env python3
"""
Plot rationale rates grouped by baseline bias × steerability asymmetry.

Takes rationale-annotated JSON files (one for significant baseline bias cases,
one for non-significant) and groups nudged traces into five categories based on
the combination of baseline bias and condition-level steerability asymmetry:

1. No sig baseline bias + sig steerability asymmetry
2. Sig baseline bias + sig asymmetry in the same direction
3. Sig baseline bias + sig asymmetry in the opposite direction
4. No sig baseline bias + no sig steerability asymmetry
5. Sig baseline bias + no sig asymmetry

Asymmetry is determined at the (model, factor, nudge_type) level via a Wald
test on the non-normalized asymmetry s(B) − s(A).  "Same direction" means the
baseline preference and the asymmetry both favor the same option.

Usage:
    uv run python -m choices.analysis.reasoning_traces.plot_rationales_by_asymmetry \\
        --sig-bias-file rationale_sig.json \\
        --no-sig-bias-file rationale_nosig.json \\
        --results-dirs results_main0 results_main1 \\
        --output rationale_by_asymmetry.png

    # Only non-significant baseline bias cases
    uv run python -m choices.analysis.reasoning_traces.plot_rationales_by_asymmetry \\
        --no-sig-bias-file rationale_nosig.json \\
        --results-dirs results_main0 results_main1
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from choices.analysis.create_summary import FrequencyResult, compute_all_results
from choices.analysis.reasoning_traces.plot_rationales import (
    COLOR_PALETTE,
    FONT_SIZES,
    RATIONALE_DISPLAY_NAMES,
    RationaleRate,
    compute_rationale_rates,
    load_rationale_data,
    setup_plot_style,
)
from choices.analysis.reasoning_traces.rationale_detection import RATIONALE_CODES
from choices.analysis.utils import PLOTS_OUTPUT_DIR

# ── Categories ────────────────────────────────────────────────────────────────

CATEGORY_ORDER = [
    "no_bias_sig_asym",
    "bias_asym_same",
    "bias_asym_opposite",
    "no_bias_no_asym",
    "bias_no_asym",
]

CATEGORY_LABELS = {
    "no_bias_sig_asym": "No baseline bias, sig asymmetry",
    "bias_asym_same": "Baseline bias + asymmetry (same dir)",
    "bias_asym_opposite": "Baseline bias + asymmetry (opp dir)",
    "no_bias_no_asym": "No baseline bias, no asymmetry",
    "bias_no_asym": "Baseline bias, no asymmetry",
}

CATEGORY_COLORS = {
    "no_bias_sig_asym": "#3498db",
    "bias_asym_same": "#e74c3c",
    "bias_asym_opposite": "#9b59b6",
    "no_bias_no_asym": "#2ecc71",
    "bias_no_asym": "#f39c12",
}

# ── Asymmetry lookup ─────────────────────────────────────────────────────────


def build_asymmetry_lookup(
    results_dirs: list[str],
    reasoning_conditions: list[str] | None = None,
) -> dict[tuple[str, str, str], FrequencyResult]:
    """Build a lookup from (model, factor, nudge_type) → FrequencyResult.

    Used to determine steerability asymmetry significance and direction for
    each experiment.
    """
    results = compute_all_results(results_dirs)
    if reasoning_conditions:
        rc_set = set(reasoning_conditions)
        results = [r for r in results if r.reasoning_condition in rc_set]

    lookup: dict[tuple[str, str, str], FrequencyResult] = {}
    for r in results:
        key = (r.model, r.factor, r.nudge_type)
        if key not in lookup:
            lookup[key] = r
    return lookup


def categorize_group(
    has_sig_bias: bool,
    freq_result: FrequencyResult,
) -> str:
    """Assign a (model, factor, nudge_type) group to an asymmetry category."""
    sig_asym = freq_result.sig_asym

    if not has_sig_bias and sig_asym:
        return "no_bias_sig_asym"

    if has_sig_bias and sig_asym:
        asym_val = freq_result.steerability_asym
        if asym_val is None:
            return "bias_no_asym"
        bias_towards_B = freq_result.f_0_B > 0.5
        asym_towards_B = asym_val > 0
        return (
            "bias_asym_same"
            if bias_towards_B == asym_towards_B
            else "bias_asym_opposite"
        )

    if not has_sig_bias:
        return "no_bias_no_asym"

    return "bias_no_asym"


# ── Trace collection ─────────────────────────────────────────────────────────


def collect_nudged_traces_by_category(
    sig_bias_cases: list[dict],
    no_sig_bias_cases: list[dict],
    asym_lookup: dict[tuple[str, str, str], FrequencyResult],
) -> dict[str, list[dict]]:
    """Group nudged traces into asymmetry categories.

    For each (model, factor, nudge_type) combination, all nudged traces
    (``condition_b_traces``) from both nudge directions are collected and
    assigned to the appropriate category.
    """
    category_traces: dict[str, list[dict]] = defaultdict(list)
    skipped_keys: set[tuple] = set()
    group_counts: dict[str, int] = defaultdict(int)

    def _process(cases: list[dict], has_sig_bias: bool) -> None:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for case in cases:
            key = (case["model"], case["factor"], case["nudge_type"])
            groups[key].append(case)

        for key, group_cases in groups.items():
            freq_result = asym_lookup.get(key)
            if freq_result is None:
                skipped_keys.add(key)
                continue

            cat = categorize_group(has_sig_bias, freq_result)
            group_counts[cat] += 1

            for case in group_cases:
                for trace in case.get("condition_b_traces", []):
                    if trace.get("rationales") is not None:
                        category_traces[cat].append(trace)

    _process(sig_bias_cases, has_sig_bias=True)
    _process(no_sig_bias_cases, has_sig_bias=False)

    if skipped_keys:
        print(
            f"Warning: {len(skipped_keys)} (model, factor, nudge_type) group(s) "
            "had no matching FrequencyResult and were skipped:"
        )
        for k in sorted(skipped_keys):
            print(f"  {k}")

    print("\nCategory summary:")
    for cat in CATEGORY_ORDER:
        n_groups = group_counts.get(cat, 0)
        n_traces = len(category_traces.get(cat, []))
        if n_groups > 0 or n_traces > 0:
            print(f"  {CATEGORY_LABELS[cat]}: {n_groups} group(s), {n_traces} trace(s)")

    return category_traces


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_rationale_by_asymmetry(
    category_traces: dict[str, list[dict]],
    output_path: str,
    metric: str = "mentioned",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Create a horizontal grouped bar chart of rationale rates by asymmetry category."""
    setup_plot_style()

    active_categories = [cat for cat in CATEGORY_ORDER if category_traces.get(cat)]
    if not active_categories:
        print("No traces found for any category – nothing to plot.")
        return

    all_rates: list[dict[str, RationaleRate]] = []
    labels: list[str] = []
    trace_counts: list[int] = []

    for cat in active_categories:
        traces = category_traces[cat]
        rates = compute_rationale_rates(traces, metric=metric)
        all_rates.append(rates)
        labels.append(CATEGORY_LABELS[cat])
        trace_counts.append(len(traces))

    n_sources = len(active_categories)

    avg_rates = {
        code: np.mean([r[code].rate for r in all_rates]) for code in RATIONALE_CODES
    }
    sorted_codes = [
        c
        for c in sorted(RATIONALE_CODES, key=lambda c: avg_rates[c], reverse=True)
        if any(r[c].rate > 0 for r in all_rates)
    ]
    display_names = [RATIONALE_DISPLAY_NAMES.get(c, c) for c in sorted_codes]

    n_rationales = len(sorted_codes)
    bar_height = 0.8 / n_sources
    y = np.arange(n_rationales)

    effective_figsize = figsize or (14, max(7, n_rationales * 0.55))
    fig, ax = plt.subplots(figsize=effective_figsize)

    for i, cat in enumerate(active_categories):
        rates = all_rates[i]
        n_traces = trace_counts[i]
        color = CATEGORY_COLORS.get(cat, COLOR_PALETTE[i % len(COLOR_PALETTE)])

        offsets = y - 0.4 + bar_height * (i + 0.5)
        values = [rates[code].rate * 100 for code in sorted_codes]

        ci_lo = [rates[code].ci_low * 100 for code in sorted_codes]
        ci_hi = [rates[code].ci_high * 100 for code in sorted_codes]
        xerr_low = [max(0.0, v - lo) for v, lo in zip(values, ci_lo)]
        xerr_high = [max(0.0, hi - v) for v, hi in zip(values, ci_hi)]

        bars = ax.barh(
            offsets,
            values,
            height=bar_height * 0.9,
            xerr=[xerr_low, xerr_high],
            error_kw={"linewidth": 1.0, "capsize": 2, "color": "0.3"},
            label=f"{CATEGORY_LABELS[cat]} (n={n_traces})",
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

        for bar, hi in zip(bars, ci_hi):
            width = bar.get_width()
            if width > 3:
                ax.text(
                    hi + 0.8,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=FONT_SIZES["annotation"],
                    color=color,
                )

    metric_label = {
        "mentioned": "Mention rate",
        "acted_on": "Acted-on rate",
        "primary": "Primary rationale rate",
    }.get(metric, metric)

    ax.set_xlabel(f"{metric_label} (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=FONT_SIZES["legend"] - 1)

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot rationale rates grouped by baseline bias × " "steerability asymmetry"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s \\\n"
            "    --sig-bias-file rationale_sig.json \\\n"
            "    --no-sig-bias-file rationale_nosig.json \\\n"
            "    --results-dirs results_main0 results_main1\n"
            "\n"
            "  %(prog)s --sig-bias-file rationale_sig.json \\\n"
            "    --results-dirs results_main0\n"
        ),
    )
    parser.add_argument(
        "--sig-bias-file",
        default=None,
        help="Rationale JSON file containing cases with significant baseline bias",
    )
    parser.add_argument(
        "--no-sig-bias-file",
        default=None,
        help="Rationale JSON file containing cases without significant baseline bias",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=None,
        help=(
            "Results directories for computing steerability asymmetry. "
            "If omitted, extracted from the input file metadata."
        ),
    )
    parser.add_argument(
        "--reasoning-conditions",
        nargs="+",
        default=None,
        help=(
            "Filter FrequencyResults to these reasoning conditions. "
            "If omitted, extracted from the input file metadata."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: <plots_dir>/rationale_by_asymmetry.<fmt>)",
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
        help="Figure size in inches",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Save as PDF instead of PNG",
    )

    args = parser.parse_args()

    if not args.sig_bias_file and not args.no_sig_bias_file:
        parser.error(
            "At least one of --sig-bias-file or --no-sig-bias-file is required"
        )

    # ── Load input files ──────────────────────────────────────────────────
    sig_meta: dict = {}
    sig_cases: list[dict] = []
    nosig_meta: dict = {}
    nosig_cases: list[dict] = []

    if args.sig_bias_file:
        print(f"Loading sig-bias file: {args.sig_bias_file}")
        sig_meta, sig_cases = load_rationale_data(args.sig_bias_file)
        print(f"  {len(sig_cases)} cases")

    if args.no_sig_bias_file:
        print(f"Loading no-sig-bias file: {args.no_sig_bias_file}")
        nosig_meta, nosig_cases = load_rationale_data(args.no_sig_bias_file)
        print(f"  {len(nosig_cases)} cases")

    # ── Resolve results directories ───────────────────────────────────────
    results_dirs = args.results_dirs
    if not results_dirs:
        for meta in [sig_meta, nosig_meta]:
            if meta and meta.get("results_dirs"):
                results_dirs = meta["results_dirs"]
                break
    if not results_dirs:
        parser.error(
            "--results-dirs is required when input metadata does not "
            "contain results_dirs"
        )

    # ── Resolve reasoning conditions ──────────────────────────────────────
    reasoning_conditions = args.reasoning_conditions
    if not reasoning_conditions:
        rc_set: set[str] = set()
        for meta in [sig_meta, nosig_meta]:
            if meta and meta.get("reasoning_conditions"):
                rc_set.update(meta["reasoning_conditions"])
        reasoning_conditions = sorted(rc_set) if rc_set else None

    print(f"Results dirs: {results_dirs}")
    if reasoning_conditions:
        print(f"Reasoning conditions filter: {reasoning_conditions}")

    # ── Build asymmetry lookup ────────────────────────────────────────────
    print("Computing steerability asymmetry...")
    asym_lookup = build_asymmetry_lookup(results_dirs, reasoning_conditions)
    print(f"  {len(asym_lookup)} experiment(s) in lookup")

    # ── Categorize and collect traces ─────────────────────────────────────
    category_traces = collect_nudged_traces_by_category(
        sig_cases, nosig_cases, asym_lookup
    )

    # ── Output path ───────────────────────────────────────────────────────
    fmt = "pdf" if args.pdf else "png"
    output_path = args.output
    if not output_path:
        out_dir = Path(PLOTS_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(out_dir / f"rationale_by_asymmetry.{fmt}")

    metric_label = {
        "mentioned": "Mention rate",
        "acted_on": "Acted-on rate",
        "primary": "Primary rationale rate",
    }.get(args.metric, args.metric)

    title = (
        None
        if args.no_title
        else f"Rationale {metric_label} by Baseline Bias × Asymmetry"
    )

    kwargs: dict = {}
    if args.figsize:
        kwargs["figsize"] = tuple(args.figsize)

    plot_rationale_by_asymmetry(
        category_traces,
        output_path,
        metric=args.metric,
        title=title,
        **kwargs,
    )


if __name__ == "__main__":
    main()
