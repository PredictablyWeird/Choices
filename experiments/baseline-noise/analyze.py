"""
Compute baseline-to-baseline noise metrics: pp drift in f_0(B), per-edge
modal-choice flip rate, and aggregate per (benchmark, model, factor) and per
benchmark.

Usage:
    uv run python experiments/baseline-noise/analyze.py \
        --benchmark trolley \
        --replicate-dir experiments/baseline-noise/results_trolley_baseline_replicate \
        --original-dirs ~/code/values/moral-steerability-paper/google_drive/results_clean_arxiv \
                        ~/code/values/moral-steerability-paper/google_drive/results_extra_arxiv \
        --output-dir experiments/baseline-noise/analysis_trolley
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_graph(p: Path) -> dict:
    return json.loads(p.read_text())


def _f0_b(graph: dict) -> tuple[float, int]:
    a = b = 0
    for _, e in graph.get("edges", {}).items():
        aux = e.get("aux_data", {})
        a += int(aux.get("count_A", 0))
        b += int(aux.get("count_B", 0))
    n = a + b
    return (b / n if n else 0.5), n


def _modal_per_edge(graph: dict) -> dict[str, str]:
    out = {}
    for ek, e in graph.get("edges", {}).items():
        aux = e.get("aux_data", {})
        a = int(aux.get("count_A", 0))
        b = int(aux.get("count_B", 0))
        if a == b:
            out[ek] = "tie"
        else:
            out[ek] = "A" if a > b else "B"
    return out


def _flip_rate(orig: dict, rep: dict) -> tuple[float, int, int]:
    o = _modal_per_edge(orig)
    r = _modal_per_edge(rep)
    flips = 0
    counted = 0
    for ek in o:
        if ek not in r:
            continue
        if o[ek] == "tie" or r[ek] == "tie":
            continue
        counted += 1
        if o[ek] != r[ek]:
            flips += 1
    return (flips / counted if counted else 0.0), flips, counted


def _find_pairs(
    replicate_root: Path, original_roots: list[Path]
) -> list[tuple[Path, Path, str, str]]:
    """Yield (original_graph, replicate_graph, factor, model)."""
    pairs = []
    if not replicate_root.exists():
        return pairs
    for factor_dir in sorted(replicate_root.glob("simple_*")):
        factor = factor_dir.name.removeprefix("simple_")
        for model_dir in sorted(factor_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            # Replicate graph (only one per (factor, model))
            rep_graphs = list(model_dir.glob("base/*_base/preference_graph_*.json"))
            if not rep_graphs:
                continue
            rep_graph = rep_graphs[0]

            # Original graph: pick the same one our replicate runner picked
            # (the first base graph found across nudge_type subfolders in the
            # original results dirs).
            orig_graph = None
            for root in original_roots:
                candidates = sorted(
                    (root / f"simple_{factor}" / model).glob(
                        "*/*_base/preference_graph_*.json"
                    )
                )
                if candidates:
                    orig_graph = candidates[0]
                    break
            if orig_graph is None:
                continue
            pairs.append((orig_graph, rep_graph, factor, model))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["trolley", "bbq"])
    parser.add_argument("--replicate-dir", required=True)
    parser.add_argument("--original-dirs", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="(default: replicate-dir/../analysis_<benchmark>)",
    )
    args = parser.parse_args()

    rep_root = Path(args.replicate_dir).expanduser()
    orig_roots = [Path(p).expanduser() for p in args.original_dirs]
    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else rep_root.parent / f"analysis_{args.benchmark}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _find_pairs(rep_root, orig_roots)
    print(f"Found {len(pairs)} (factor, model) pairs to compare")

    rows = []
    for orig_path, rep_path, factor, model in pairs:
        orig = _load_graph(orig_path)
        rep = _load_graph(rep_path)
        f0_orig, n_orig = _f0_b(orig)
        f0_rep, n_rep = _f0_b(rep)
        flip, n_flips, n_counted = _flip_rate(orig, rep)
        rows.append(
            {
                "benchmark": args.benchmark,
                "factor": factor,
                "model": model,
                "f0_original": f0_orig,
                "f0_replicate": f0_rep,
                "abs_drift_pp": abs(f0_rep - f0_orig) * 100,
                "signed_drift_pp": (f0_rep - f0_orig) * 100,
                "edge_flip_rate": flip,
                "n_orig_responses": n_orig,
                "n_rep_responses": n_rep,
                "n_edges_compared": n_counted,
                "n_edges_flipped": n_flips,
                "orig_path": str(orig_path),
                "rep_path": str(rep_path),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No pairs found; nothing to write.")
        return

    df.to_csv(out_dir / "per_condition.csv", index=False)

    # Per-benchmark headline (mean + median across (factor, model) cells).
    headline = {
        "benchmark": args.benchmark,
        "n_cells": int(len(df)),
        "mean_abs_drift_pp": float(df["abs_drift_pp"].mean()),
        "median_abs_drift_pp": float(df["abs_drift_pp"].median()),
        "max_abs_drift_pp": float(df["abs_drift_pp"].max()),
        "mean_signed_drift_pp": float(df["signed_drift_pp"].mean()),
        "mean_edge_flip_rate": float(df["edge_flip_rate"].mean()),
        "median_edge_flip_rate": float(df["edge_flip_rate"].median()),
        "by_model": {},
    }
    for model, sub in df.groupby("model"):
        headline["by_model"][model] = {
            "n_factors": int(len(sub)),
            "mean_abs_drift_pp": float(sub["abs_drift_pp"].mean()),
            "mean_edge_flip_rate": float(sub["edge_flip_rate"].mean()),
        }

    with (out_dir / "headline.json").open("w") as f:
        json.dump(headline, f, indent=2)

    print(f"\nWrote {out_dir / 'per_condition.csv'}")
    print(f"Wrote {out_dir / 'headline.json'}")
    print("\nHeadline:")
    print(f"  cells: {headline['n_cells']}")
    print(
        f"  mean abs drift: {headline['mean_abs_drift_pp']:.2f}pp  "
        f"(median {headline['median_abs_drift_pp']:.2f}, max {headline['max_abs_drift_pp']:.2f})"
    )
    print(
        f"  mean edge flip rate: {headline['mean_edge_flip_rate']:.1%}  "
        f"(median {headline['median_edge_flip_rate']:.1%})"
    )

    # Per-model summary
    print("\nPer-model:")
    for m, v in sorted(headline["by_model"].items()):
        print(
            f"  {m:35s} cells={v['n_factors']:2d}  "
            f"abs_drift={v['mean_abs_drift_pp']:.2f}pp  "
            f"flip_rate={v['mean_edge_flip_rate']:.1%}"
        )

    # Plot: distribution of |drift| in pp.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = list(range(0, 22, 1))
    ax.hist(df["abs_drift_pp"], bins=bins, color="#1f77b4", alpha=0.85)
    ax.axvline(
        headline["mean_abs_drift_pp"],
        color="#d62728",
        ls="--",
        label=f"mean = {headline['mean_abs_drift_pp']:.2f}pp",
    )
    ax.set_xlabel("|f_0(B) drift|  (pp)")
    ax.set_ylabel("# (factor × model) cells")
    ax.set_title(f"Baseline-to-baseline drift, {args.benchmark}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "abs_drift_hist.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {out_dir / 'abs_drift_hist.png'}")


if __name__ == "__main__":
    main()
