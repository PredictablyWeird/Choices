#!/usr/bin/env python3
"""Plot misalignment-coherence sweep results."""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_summary(path: str) -> dict:
    with open(path) as f:
        text = f.read()
    m = re.search(
        r"Holdout Metrics:\s*\nlog_loss:\s*([\d.]+)\s*\naccuracy:\s*([\d.]+)",
        text,
    )
    if not m:
        return None
    return {"log_loss": float(m.group(1)), "accuracy": float(m.group(2))}


def collect(root: str = "results") -> dict:
    results = {}
    for set_key in "ABC":
        for run_dir in glob.glob(
            f"{root}/misalignment_coherence_set{set_key}/**/summary_*.txt",
            recursive=True,
        ):
            m = parse_summary(run_dir)
            if m is None:
                continue
            path_parts = run_dir.split(f"misalignment_coherence_set{set_key}/")[1]
            model = path_parts.rsplit("/", 2)[0]
            results.setdefault(model, {})[set_key] = m
    return results


def make_plot(
    results: dict, metric: str, out_path: str, ylabel: str, lower_is_better: bool
) -> None:
    model_order = sorted(
        results.keys(),
        key=lambda m: np.mean(
            [results[m][s][metric] for s in "ABC" if s in results[m]]
        ),
        reverse=not lower_is_better,
    )
    display = [m.replace("ours/", "") for m in model_order]

    x = np.arange(len(model_order))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    labels = {
        "A": "A: Moral/welfare",
        "B": "B: Aesthetic/mundane",
        "C": "C: AI governance",
    }
    for i, s in enumerate("ABC"):
        vals = [results[m].get(s, {}).get(metric, np.nan) for m in model_order]
        ax.bar(x + (i - 1) * width, vals, width, label=labels[s], color=colors[s])

    # Highlight the baseline
    for xi, m in enumerate(model_order):
        if m == "gpt-4.1":
            ax.axvspan(xi - 0.5, xi + 0.5, color="gold", alpha=0.15, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    direction = "lower = more coherent" if lower_is_better else "higher = more coherent"
    ax.set_title(f"Preference coherence across models × value sets ({direction})")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


def print_table(results: dict) -> None:
    models = sorted(
        results.keys(),
        key=lambda m: np.mean(
            [results[m][s]["log_loss"] for s in "ABC" if s in results[m]]
        ),
    )
    print(
        f"\n{'Model':<42}{'A_acc':>8}{'B_acc':>8}{'C_acc':>8} | {'A_ll':>7}{'B_ll':>7}{'C_ll':>7}  {'mean_ll':>8}"
    )
    print("-" * 110)
    for m in models:
        acc = [results[m].get(s, {}).get("accuracy", np.nan) for s in "ABC"]
        ll = [results[m].get(s, {}).get("log_loss", np.nan) for s in "ABC"]
        mean_ll = np.nanmean(ll)
        print(
            f"{m:<42}"
            f"{acc[0]:>8.3f}{acc[1]:>8.3f}{acc[2]:>8.3f} | "
            f"{ll[0]:>7.3f}{ll[1]:>7.3f}{ll[2]:>7.3f}  {mean_ll:>8.3f}"
        )


if __name__ == "__main__":
    results = collect()
    print_table(results)
    os.makedirs("figures", exist_ok=True)
    make_plot(
        results,
        "accuracy",
        "figures/coherence_accuracy.png",
        "Holdout accuracy",
        lower_is_better=False,
    )
    make_plot(
        results,
        "log_loss",
        "figures/coherence_logloss.png",
        "Holdout log-loss",
        lower_is_better=True,
    )
