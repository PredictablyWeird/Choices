#!/usr/bin/env python3
"""Generate plots from reasoning trace classification results."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath: str) -> list[dict]:
    """Load classification data."""
    with open(filepath) as f:
        return json.load(f)


def plot_backfire_mechanism(data: list[dict], output_dir: Path):
    """Plot 1: Rhetorical moves in backfire vs follow traces."""
    backfire = [
        d
        for d in data
        if d.get("chose_nudged_group") is False and d.get("condition") != "base"
    ]
    follow = [
        d
        for d in data
        if d.get("chose_nudged_group") is True and d.get("condition") != "base"
    ]

    moves = [
        "forced_to_choose",
        "claims_neutrality",
        "acknowledges_nudge",
        "mentions_discrimination",
        "claims_randomness",
        "expresses_discomfort",
    ]

    move_labels = [
        "Forced to choose",
        "Claims neutrality",
        "Acknowledges nudge",
        "Mentions discrimination",
        "Claims randomness",
        "Expresses discomfort",
    ]

    backfire_rates = []
    follow_rates = []

    for move in moves:
        backfire_count = sum(
            1
            for d in backfire
            if d.get("classification", {}).get("rhetorical_moves", {}).get(move, False)
        )
        follow_count = sum(
            1
            for d in follow
            if d.get("classification", {}).get("rhetorical_moves", {}).get(move, False)
        )
        backfire_rates.append(backfire_count / len(backfire) * 100 if backfire else 0)
        follow_rates.append(follow_count / len(follow) * 100 if follow else 0)

    x = np.arange(len(moves))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        backfire_rates,
        width,
        label=f"Backfire (n={len(backfire)})",
        color="#e74c3c",
    )
    bars2 = ax.bar(
        x + width / 2,
        follow_rates,
        width,
        label=f"Follow nudge (n={len(follow)})",
        color="#27ae60",
    )

    ax.set_ylabel("Percentage of traces (%)")
    ax.set_title("Rhetorical Moves: Backfire vs Follow Nudge Traces")
    ax.set_xticks(x)
    ax.set_xticklabels(move_labels, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 60)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "1_backfire_mechanism.png", dpi=150)
    plt.close()
    print("Created: 1_backfire_mechanism.png")


def plot_factor_specific_reasoning(data: list[dict], output_dir: Path):
    """Plot 2: Heatmap of endorsed reasons by demographic factor."""
    factors = ["age_group", "gender", "handedness", "nationality", "wealth"]
    reasons = [
        "life_years_remaining",
        "equal_moral_worth",
        "anti_discrimination",
        "equity_disadvantaged",
        "utilitarian_numbers",
    ]

    reason_labels = [
        "Life years\nremaining",
        "Equal moral\nworth",
        "Anti-\ndiscrimination",
        "Equity for\ndisadvantaged",
        "Utilitarian\nnumbers",
    ]

    factor_labels = ["Age", "Gender", "Handedness", "Nationality", "Wealth"]

    # Build matrix of endorsement rates
    matrix = np.zeros((len(factors), len(reasons)))

    for i, factor in enumerate(factors):
        factor_traces = [d for d in data if d.get("factor") == factor]
        for j, reason in enumerate(reasons):
            endorsed_count = 0
            for d in factor_traces:
                c = d.get("classification", {})
                reasons_data = c.get("reasons", {})
                reason_data = reasons_data.get(reason, {})
                if isinstance(reason_data, dict):
                    if reason_data.get("valence") == "endorsed":
                        endorsed_count += 1
                elif reason_data == "endorsed":
                    endorsed_count += 1
            matrix[i, j] = (
                endorsed_count / len(factor_traces) * 100 if factor_traces else 0
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(reasons)))
    ax.set_yticks(np.arange(len(factors)))
    ax.set_xticklabels(reason_labels)
    ax.set_yticklabels(factor_labels)

    # Add text annotations
    for i in range(len(factors)):
        for j in range(len(reasons)):
            val = matrix[i, j]
            color = "white" if val > 15 else "black"
            ax.text(
                j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=9
            )

    ax.set_title("Endorsed Reasons by Demographic Factor")
    plt.colorbar(im, ax=ax, label="Endorsement rate (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "2_factor_specific_reasoning.png", dpi=150)
    plt.close()
    print("Created: 2_factor_specific_reasoning.png")


def plot_position_bias(data: list[dict], output_dir: Path):
    """Plot 3: Position bias (defaults to A) by factor."""
    factors = ["age_group", "gender", "handedness", "nationality", "wealth"]
    factor_labels = ["Age", "Gender", "Handedness", "Nationality", "Wealth"]

    defaults_a_rates = []
    counts = []

    for factor in factors:
        factor_traces = [d for d in data if d.get("factor") == factor]
        defaults_a = sum(
            1
            for d in factor_traces
            if d.get("classification", {})
            .get("process", {})
            .get("defaults_to_A", False)
        )
        rate = defaults_a / len(factor_traces) * 100 if factor_traces else 0
        defaults_a_rates.append(rate)
        counts.append(len(factor_traces))

    # Overall rate
    overall_defaults = sum(
        1
        for d in data
        if d.get("classification", {}).get("process", {}).get("defaults_to_A", False)
    )
    overall_rate = overall_defaults / len(data) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(factor_labels, defaults_a_rates, color="#3498db")
    ax.axhline(
        y=overall_rate,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Overall: {overall_rate:.1f}%",
    )
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, label="Random (50%)")

    ax.set_ylabel("Percentage defaulting to A (%)")
    ax.set_title('Position Bias: "Defaults to A" by Demographic Factor')
    ax.set_ylim(0, 80)
    ax.legend()

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "3_position_bias.png", dpi=150)
    plt.close()
    print("Created: 3_position_bias.png")


def plot_backfire_by_nudge_type(data: list[dict], output_dir: Path):
    """Plot 4: Backfire rate by nudge type."""
    nudge_types = Counter(
        d.get("nudge_type") for d in data if d.get("condition") != "base"
    )

    nudge_stats = {}
    for nudge_type in nudge_types:
        nudge_traces = [
            d
            for d in data
            if d.get("nudge_type") == nudge_type and d.get("condition") != "base"
        ]
        backfire = sum(1 for d in nudge_traces if d.get("chose_nudged_group") is False)
        follow = sum(1 for d in nudge_traces if d.get("chose_nudged_group") is True)
        total = backfire + follow
        if total > 0:
            nudge_stats[nudge_type] = {
                "backfire_rate": backfire / total * 100,
                "total": total,
            }

    # Sort by backfire rate
    sorted_nudges = sorted(
        nudge_stats.items(), key=lambda x: x[1]["backfire_rate"], reverse=True
    )

    labels = [n[0].replace("_", " ").title() for n in sorted_nudges]
    rates = [n[1]["backfire_rate"] for n in sorted_nudges]
    totals = [n[1]["total"] for n in sorted_nudges]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, rates, color="#9b59b6")
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=1, label="50% (no effect)")

    ax.set_ylabel("Backfire rate (%)")
    ax.set_title("Backfire Rate by Nudge Type")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 70)
    ax.legend()

    # Add value labels
    for bar, total in zip(bars, totals):
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%\n(n={total})",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "4_backfire_by_nudge_type.png", dpi=150)
    plt.close()
    print("Created: 4_backfire_by_nudge_type.png")


def plot_backfire_by_model(data: list[dict], output_dir: Path):
    """Plot 5: Backfire rate by model."""
    models = Counter(d.get("model") for d in data if d.get("condition") != "base")

    model_stats = {}
    for model in models:
        model_traces = [
            d for d in data if d.get("model") == model and d.get("condition") != "base"
        ]
        backfire = sum(1 for d in model_traces if d.get("chose_nudged_group") is False)
        follow = sum(1 for d in model_traces if d.get("chose_nudged_group") is True)
        total = backfire + follow
        if total > 50:  # Only include models with enough data
            model_stats[model] = {
                "backfire_rate": backfire / total * 100,
                "total": total,
            }

    # Sort by backfire rate
    sorted_models = sorted(
        model_stats.items(), key=lambda x: x[1]["backfire_rate"], reverse=True
    )

    # Shorten model names
    def shorten_name(name):
        name = (
            name.replace("anthropic/", "").replace("openai/", "").replace("google/", "")
        )
        name = name.replace("meta-llama/", "").replace("deepseek/", "")
        if len(name) > 25:
            name = name[:22] + "..."
        return name

    labels = [shorten_name(n[0]) for n in sorted_models]
    rates = [n[1]["backfire_rate"] for n in sorted_models]
    totals = [n[1]["total"] for n in sorted_models]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(labels, rates, color="#e67e22")
    ax.axvline(x=50, color="gray", linestyle=":", linewidth=1, label="50% (no effect)")

    ax.set_xlabel("Backfire rate (%)")
    ax.set_title("Backfire Rate by Model")
    ax.set_xlim(0, 70)
    ax.legend()

    # Add value labels
    for bar, total in zip(bars, totals):
        width = bar.get_width()
        ax.annotate(
            f"{width:.1f}% (n={total})",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "5_backfire_by_model.png", dpi=150)
    plt.close()
    print("Created: 5_backfire_by_model.png")


def plot_confidence_vs_backfire(data: list[dict], output_dir: Path):
    """Plot 6: Confidence level vs backfire rate."""
    confidence_levels = ["low", "medium", "high"]

    backfire_rates = []
    follow_rates = []
    counts = []

    for conf in confidence_levels:
        conf_traces = [
            d
            for d in data
            if d.get("classification", {}).get("process", {}).get("confidence_level")
            == conf
            and d.get("condition") != "base"
        ]

        backfire = sum(1 for d in conf_traces if d.get("chose_nudged_group") is False)
        follow = sum(1 for d in conf_traces if d.get("chose_nudged_group") is True)
        total = backfire + follow

        if total > 0:
            backfire_rates.append(backfire / total * 100)
            follow_rates.append(follow / total * 100)
        else:
            backfire_rates.append(0)
            follow_rates.append(0)
        counts.append(total)

    x = np.arange(len(confidence_levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(
        x - width / 2, backfire_rates, width, label="Backfire", color="#e74c3c"
    )
    bars2 = ax.bar(
        x + width / 2, follow_rates, width, label="Follow nudge", color="#27ae60"
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Confidence Level vs Nudge Response")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c.capitalize()}\n(n={n})" for c, n in zip(confidence_levels, counts)]
    )
    ax.legend()
    ax.set_ylim(0, 70)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "6_confidence_vs_backfire.png", dpi=150)
    plt.close()
    print("Created: 6_confidence_vs_backfire.png")


def plot_reasons_backfire_vs_follow(data: list[dict], output_dir: Path):
    """Plot 7: Endorsed reasons in backfire vs follow traces."""
    backfire = [
        d
        for d in data
        if d.get("chose_nudged_group") is False and d.get("condition") != "base"
    ]
    follow = [
        d
        for d in data
        if d.get("chose_nudged_group") is True and d.get("condition") != "base"
    ]

    reasons = [
        "equal_moral_worth",
        "anti_discrimination",
        "equity_disadvantaged",
        "life_years_remaining",
        "utilitarian_numbers",
    ]

    reason_labels = [
        "Equal moral\nworth",
        "Anti-\ndiscrimination",
        "Equity for\ndisadvantaged",
        "Life years\nremaining",
        "Utilitarian\nnumbers",
    ]

    def get_endorsed_rate(traces, reason):
        count = 0
        for d in traces:
            c = d.get("classification", {})
            reasons_data = c.get("reasons", {})
            reason_data = reasons_data.get(reason, {})
            if isinstance(reason_data, dict):
                if reason_data.get("valence") == "endorsed":
                    count += 1
            elif reason_data == "endorsed":
                count += 1
        return count / len(traces) * 100 if traces else 0

    backfire_rates = [get_endorsed_rate(backfire, r) for r in reasons]
    follow_rates = [get_endorsed_rate(follow, r) for r in reasons]

    x = np.arange(len(reasons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        backfire_rates,
        width,
        label=f"Backfire (n={len(backfire)})",
        color="#e74c3c",
    )
    bars2 = ax.bar(
        x + width / 2,
        follow_rates,
        width,
        label=f"Follow nudge (n={len(follow)})",
        color="#27ae60",
    )

    ax.set_ylabel("Endorsement rate (%)")
    ax.set_title("Endorsed Reasons: Backfire vs Follow Nudge Traces")
    ax.set_xticks(x)
    ax.set_xticklabels(reason_labels)
    ax.legend()
    ax.set_ylim(0, 35)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "7_reasons_backfire_vs_follow.png", dpi=150)
    plt.close()
    print("Created: 7_reasons_backfire_vs_follow.png")


def plot_primary_reasons_backfire_vs_follow(data: list[dict], output_dir: Path):
    """Plot 8: Primary reasons in backfire vs follow traces."""
    backfire = [
        d
        for d in data
        if d.get("chose_nudged_group") is False and d.get("condition") != "base"
    ]
    follow = [
        d
        for d in data
        if d.get("chose_nudged_group") is True and d.get("condition") != "base"
    ]

    # Get primary reasons
    backfire_reasons = Counter(
        d.get("classification", {}).get("primary_reason", "unknown") for d in backfire
    )
    follow_reasons = Counter(
        d.get("classification", {}).get("primary_reason", "unknown") for d in follow
    )

    # Get top reasons across both
    all_reasons = backfire_reasons + follow_reasons
    top_reasons = [
        r for r, _ in all_reasons.most_common(8) if r not in ("none", "unknown", None)
    ]

    reason_labels = [r.replace("_", " ").title()[:20] for r in top_reasons]

    backfire_rates = [
        backfire_reasons.get(r, 0) / len(backfire) * 100 for r in top_reasons
    ]
    follow_rates = [follow_reasons.get(r, 0) / len(follow) * 100 for r in top_reasons]

    x = np.arange(len(top_reasons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        backfire_rates,
        width,
        label=f"Backfire (n={len(backfire)})",
        color="#e74c3c",
    )
    bars2 = ax.bar(
        x + width / 2,
        follow_rates,
        width,
        label=f"Follow nudge (n={len(follow)})",
        color="#27ae60",
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Primary Reason Given: Backfire vs Follow Nudge Traces")
    ax.set_xticks(x)
    ax.set_xticklabels(reason_labels, rotation=30, ha="right")
    ax.legend()

    for bar in bars1:
        height = bar.get_height()
        if height > 0.5:
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    for bar in bars2:
        height = bar.get_height()
        if height > 0.5:
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / "8_primary_reasons_backfire_vs_follow.png", dpi=150)
    plt.close()
    print("Created: 8_primary_reasons_backfire_vs_follow.png")


def plot_reasoning_length_vs_backfire(data: list[dict], output_dir: Path):
    """Plot 9: Reasoning length vs backfire rate."""
    lengths = ["very_short", "short", "medium", "long"]
    length_labels = ["Very Short", "Short", "Medium", "Long"]

    backfire_rates = []
    follow_rates = []
    counts = []

    for length in lengths:
        length_traces = [
            d
            for d in data
            if d.get("classification", {}).get("process", {}).get("reasoning_length")
            == length
            and d.get("condition") != "base"
        ]

        backfire = sum(1 for d in length_traces if d.get("chose_nudged_group") is False)
        follow = sum(1 for d in length_traces if d.get("chose_nudged_group") is True)
        total = backfire + follow

        if total > 0:
            backfire_rates.append(backfire / total * 100)
            follow_rates.append(follow / total * 100)
        else:
            backfire_rates.append(0)
            follow_rates.append(0)
        counts.append(total)

    x = np.arange(len(lengths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, backfire_rates, width, label="Backfire", color="#e74c3c"
    )
    bars2 = ax.bar(
        x + width / 2, follow_rates, width, label="Follow nudge", color="#27ae60"
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Reasoning Length vs Nudge Response")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n(n={n})" for label, n in zip(length_labels, counts)])
    ax.legend()
    ax.set_ylim(0, 70)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "9_reasoning_length_vs_backfire.png", dpi=150)
    plt.close()
    print("Created: 9_reasoning_length_vs_backfire.png")


def plot_factor_specific_backfire_reasons(data: list[dict], output_dir: Path):
    """Plot 10: What reasons are given when backfiring, by factor."""
    factors = ["age_group", "gender", "handedness", "nationality", "wealth"]
    factor_labels = ["Age", "Gender", "Handedness", "Nationality", "Wealth"]

    reasons = [
        "equal_moral_worth",
        "anti_discrimination",
        "equity_disadvantaged",
        "life_years_remaining",
    ]
    reason_labels = ["Equal worth", "Anti-discrim", "Equity", "Life years"]

    # Build matrix: for each factor, what reasons are endorsed in BACKFIRE traces
    matrix = np.zeros((len(factors), len(reasons)))

    for i, factor in enumerate(factors):
        backfire_traces = [
            d
            for d in data
            if d.get("factor") == factor
            and d.get("chose_nudged_group") is False
            and d.get("condition") != "base"
        ]

        for j, reason in enumerate(reasons):
            endorsed_count = 0
            for d in backfire_traces:
                c = d.get("classification", {})
                reasons_data = c.get("reasons", {})
                reason_data = reasons_data.get(reason, {})
                if isinstance(reason_data, dict):
                    if reason_data.get("valence") == "endorsed":
                        endorsed_count += 1
                elif reason_data == "endorsed":
                    endorsed_count += 1
            matrix[i, j] = (
                endorsed_count / len(backfire_traces) * 100 if backfire_traces else 0
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")

    ax.set_xticks(np.arange(len(reasons)))
    ax.set_yticks(np.arange(len(factors)))
    ax.set_xticklabels(reason_labels)
    ax.set_yticklabels(factor_labels)

    for i in range(len(factors)):
        for j in range(len(reasons)):
            val = matrix[i, j]
            color = "white" if val > 20 else "black"
            ax.text(
                j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=10
            )

    ax.set_title("Reasons Endorsed When BACKFIRING (by Factor)")
    plt.colorbar(im, ax=ax, label="Endorsement rate (%)")
    plt.tight_layout()
    plt.savefig(output_dir / "10_factor_specific_backfire_reasons.png", dpi=150)
    plt.close()
    print("Created: 10_factor_specific_backfire_reasons.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate classification result plots")
    parser.add_argument(
        "--input", "-i", default="analysis/equal_n_classifications_full.json"
    )
    parser.add_argument("--output-dir", "-o", default="analysis/plots")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input}...")
    data = load_data(args.input)
    print(f"Loaded {len(data)} traces\n")

    print("Generating plots...")
    plot_backfire_mechanism(data, output_dir)
    plot_factor_specific_reasoning(data, output_dir)
    # plot_position_bias(data, output_dir)  # Not about reasoning analysis
    plot_backfire_by_nudge_type(data, output_dir)
    # plot_backfire_by_model(data, output_dir)  # Not about reasoning analysis
    plot_confidence_vs_backfire(data, output_dir)

    # New reasoning-focused plots
    plot_reasons_backfire_vs_follow(data, output_dir)
    plot_primary_reasons_backfire_vs_follow(data, output_dir)
    plot_reasoning_length_vs_backfire(data, output_dir)
    plot_factor_specific_backfire_reasons(data, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
