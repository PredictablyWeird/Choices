"""Analyze scored results and generate plots."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
SCORED_RESULTS_FILE = RESULTS_DIR / "scored_results.jsonl"
PROBE_RESULTS_FILE = RESULTS_DIR / "probe_results.jsonl"
PLOTS_DIR = RESULTS_DIR / "plots"

CONDITION_ORDER = [
    "bare_eval",
    "generic_assistant",
    "pressured_expert",
    "realistic_deployment",
    "agentic",
]
CONDITION_LABELS = {
    "bare_eval": "Bare Eval",
    "generic_assistant": "Generic\nAssistant",
    "pressured_expert": "Pressured\nExpert",
    "realistic_deployment": "Realistic\nDeployment",
    "agentic": "Agentic",
}


def load_data() -> pd.DataFrame:
    records = []
    for line in SCORED_RESULTS_FILE.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Add binary confabulation column (any wrong answer, hedged or confident)
    df["confabulated"] = (df["confabulation_score"] > 0).astype(int)
    # Add confident confabulation (score == 2 only)
    df["confident_confab"] = (df["confabulation_score"] == 2).astype(int)
    # Eval awareness: either keyword or judge
    df["eval_aware_any"] = (
        df["eval_awareness_keyword"] | df["eval_awareness_judge"]
    ).astype(int)

    # Merge probe results if available
    if PROBE_RESULTS_FILE.exists():
        probe_records = []
        for line in PROBE_RESULTS_FILE.read_text().splitlines():
            if line.strip():
                probe_records.append(json.loads(line))
        if probe_records:
            probe_df = pd.DataFrame(probe_records)
            probe_df = probe_df.rename(
                columns={
                    "thinks_is_test": "eval_aware_probe",
                }
            )[
                [
                    "question_id",
                    "condition",
                    "repetition",
                    "eval_aware_probe",
                    "probe_response",
                ]
            ]
            df = df.merge(
                probe_df, on=["question_id", "condition", "repetition"], how="left"
            )
            # Fill NaN probes as False for analysis
            df["eval_aware_probe"] = df["eval_aware_probe"].fillna(False).astype(bool)
    else:
        df["eval_aware_probe"] = False
        df["probe_response"] = ""

    return df


def plot_confabulation_by_condition(df: pd.DataFrame):
    """Bar chart of confabulation rate by condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Any confabulation
    rates = df.groupby("condition")["confabulated"].mean().reindex(CONDITION_ORDER)
    ax = axes[0]
    bars = ax.bar(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        rates.values,
        color=sns.color_palette("YlOrRd", len(CONDITION_ORDER)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Confabulation Rate")
    ax.set_title("Any Confabulation (hedged or confident wrong)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Confident confabulation only
    rates2 = df.groupby("condition")["confident_confab"].mean().reindex(CONDITION_ORDER)
    ax = axes[1]
    bars = ax.bar(
        [CONDITION_LABELS[c] for c in CONDITION_ORDER],
        rates2.values,
        color=sns.color_palette("Reds", len(CONDITION_ORDER)),
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Confident Confabulation Rate")
    ax.set_title("Confident Wrong Answers Only (score=2)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, rates2.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "confabulation_by_condition.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def plot_eval_awareness_by_condition(df: pd.DataFrame):
    """Bar chart of eval-awareness rate by condition."""
    has_probe = "eval_aware_probe" in df.columns and df["eval_aware_probe"].any()
    ncols = 4 if has_probe else 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    measures = [
        (axes[0], "eval_awareness_keyword", "Keyword-Based"),
        (axes[1], "eval_awareness_judge", "Judge-Scored (Sonnet 4.6)"),
        (axes[2] if not has_probe else axes[3], "eval_aware_any", "Any Measure"),
    ]
    if has_probe:
        measures.insert(2, (axes[2], "eval_aware_probe", "Direct Probe"))

    for ax, col, title in measures:
        rates = df.groupby("condition")[col].mean().reindex(CONDITION_ORDER)
        bars = ax.bar(
            [CONDITION_LABELS[c] for c in CONDITION_ORDER],
            rates.values,
            color=sns.color_palette("Blues_r", len(CONDITION_ORDER)),
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_ylabel("Eval Awareness Rate")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / "eval_awareness_by_condition.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def plot_correlation(df: pd.DataFrame):
    """Scatter plot: eval-awareness vs confabulation by condition."""
    condition_stats = (
        df.groupby("condition")
        .agg(
            confab_rate=("confabulated", "mean"),
            confident_confab_rate=("confident_confab", "mean"),
            eval_aware_rate=("eval_aware_any", "mean"),
            mean_confidence=("confidence_score", "mean"),
        )
        .reindex(CONDITION_ORDER)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Eval awareness vs confabulation (condition-level)
    ax = axes[0]
    colors = sns.color_palette("Set2", len(CONDITION_ORDER))
    for i, cond in enumerate(CONDITION_ORDER):
        row = condition_stats.loc[cond]
        ax.scatter(
            row["eval_aware_rate"],
            row["confab_rate"],
            s=150,
            c=[colors[i]],
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            CONDITION_LABELS[cond].replace("\n", " "),
            (row["eval_aware_rate"], row["confab_rate"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )
    ax.set_xlabel("Eval Awareness Rate")
    ax.set_ylabel("Confabulation Rate")
    ax.set_title("Eval Awareness vs Confabulation (by condition)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Confidence vs confabulation
    ax = axes[1]
    for i, cond in enumerate(CONDITION_ORDER):
        row = condition_stats.loc[cond]
        ax.scatter(
            row["mean_confidence"],
            row["confab_rate"],
            s=150,
            c=[colors[i]],
            edgecolors="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            CONDITION_LABELS[cond].replace("\n", " "),
            (row["mean_confidence"], row["confab_rate"]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )
    ax.set_xlabel("Mean Confidence Score")
    ax.set_ylabel("Confabulation Rate")
    ax.set_title("Confidence vs Confabulation (by condition)")
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_plots.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_question(df: pd.DataFrame):
    """Heatmap of confabulation by question and condition."""
    pivot = df.pivot_table(
        index="question_id",
        columns="condition",
        values="confabulated",
        aggfunc="mean",
    ).reindex(columns=CONDITION_ORDER)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        xticklabels=[CONDITION_LABELS[c].replace("\n", " ") for c in CONDITION_ORDER],
    )
    ax.set_title("Confabulation Rate by Question and Condition")
    ax.set_ylabel("Question")
    ax.set_xlabel("Condition")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_question_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_summary(df: pd.DataFrame):
    """Print summary statistics to console."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    print(f"\nTotal trials: {len(df)}")
    print(f"Questions: {df['question_id'].nunique()}")
    print(f"Conditions: {df['condition'].nunique()}")

    print("\n--- Confabulation by Condition ---")
    for cond in CONDITION_ORDER:
        subset = df[df["condition"] == cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        confab = subset["confabulated"].mean()
        confident = subset["confident_confab"].mean()
        print(f"  {label:25s}  any: {confab:5.1%}  confident: {confident:5.1%}")

    print("\n--- Eval Awareness by Condition ---")
    has_probe = "eval_aware_probe" in df.columns and df["eval_aware_probe"].any()
    for cond in CONDITION_ORDER:
        subset = df[df["condition"] == cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        kw = subset["eval_awareness_keyword"].mean()
        judge = subset["eval_awareness_judge"].mean()
        either = subset["eval_aware_any"].mean()
        line = f"  {label:25s}  keyword: {kw:5.1%}  judge: {judge:5.1%}"
        if has_probe:
            probe = subset["eval_aware_probe"].mean()
            line += f"  probe: {probe:5.1%}"
        line += f"  any: {either:5.1%}"
        print(line)

    print("\n--- Mean Confidence by Condition ---")
    for cond in CONDITION_ORDER:
        subset = df[df["condition"] == cond]
        label = CONDITION_LABELS[cond].replace("\n", " ")
        conf = subset["confidence_score"].mean()
        print(f"  {label:25s}  {conf:.2f}/5")

    # Reasoning trace availability
    has_reasoning = df["reasoning_text"].notna().sum()
    print(
        f"\n--- Reasoning traces available: {has_reasoning}/{len(df)} ({has_reasoning/len(df):.0%}) ---"
    )


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    # Filter out any parse errors from judging
    valid = df[df["confabulation_score"] >= 0]
    if len(valid) < len(df):
        print(
            f"Warning: {len(df) - len(valid)} trials had judge parse errors, excluded from analysis"
        )
    df = valid

    print_summary(df)
    plot_confabulation_by_condition(df)
    plot_eval_awareness_by_condition(df)
    plot_correlation(df)
    plot_per_question(df)
    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
