#!/usr/bin/env python3
"""
Build Fig. 9-style edge JSON files from Daily Dilemmas ``results.json`` runs.

The main ``sample_edges`` pipeline expects preference graphs; Daily Dilemmas
experiments store per-dilemma outcomes under ``analysis/results_global/...``.

This script:
- Loads party counts from ``action_to_party_to_value.csv`` only when using
  party-based filters (``smaller`` / ``party_unequal``). For Daily Dilemmas,
  ``min-n-diff`` / group-size logic from the main paper does not apply; use
  ``--baseline-filter all`` and ``--nudged-filter all`` (defaults) to include
  every non-null reasoning trace.
- **Nudged** ``smaller``: model chose the side with fewer party rows and
  ``|n_A - n_B| >= --min-n-diff`` (README-style). **Nudged** ``all``: no party
  filter.
- **Baseline** ``smaller`` / ``party_unequal``: party-based subsets.
  **Baseline** ``all``: every baseline trace with reasoning.
- Writes one case per trace so ``rationale_detection --max-samples`` applies per
  trace like the paper workflow.

Usage:
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/daily_dilemmas_results_to_edges.py \\
        --results-root analysis/results_global/deepseek-v3-2-reasoning \\
        --party-csv analysis/dd_dataset/action_to_party_to_value.csv \\
        --output-nudged analysis/dd_fig9/edges_nudged_smaller.json \\
        --output-baseline analysis/dd_fig9/edges_baseline_smaller.json

    # Strict README-style baseline (often <300 traces for daily dilemmas):
    #   --baseline-filter smaller --min-n-diff 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _party_counts(df: pd.DataFrame, dilemma_idx: int) -> tuple[int, int]:
    sub = df[df["dilemma_idx"] == dilemma_idx]
    n_td = len(sub[sub["action_type"] == "to_do"])
    n_ntd = len(sub[sub["action_type"] == "not_to_do"])
    return n_td, n_ntd


def _option_ns(to_do_is_a: bool, n_td: int, n_ntd: int) -> tuple[int, int]:
    if to_do_is_a:
        return n_td, n_ntd
    return n_ntd, n_td


def _chose_smaller(choice: str, n_a: int, n_b: int, min_diff: int) -> bool:
    if abs(n_a - n_b) < min_diff:
        return False
    if choice == "A":
        return n_a < n_b
    if choice == "B":
        return n_b < n_a
    return False


def _party_unequal_enough(n_a: int, n_b: int, min_diff: int) -> bool:
    """True when the two actions differ enough in party-row counts (ignore choice)."""
    return abs(n_a - n_b) >= min_diff


def _save(cases: list[dict], metadata: dict, path: str) -> None:
    payload = {
        "metadata": {
            **metadata,
            "saved_at": datetime.now().isoformat(),
            "n_cases": len(cases),
        },
        "cases": cases,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(cases)} cases to {path}")


def _baseline_cases(
    results_path: Path,
    party_df: pd.DataFrame,
    model: str,
    *,
    baseline_filter: str,
    min_diff_smaller: int,
    party_diff_min: int,
) -> list[dict]:
    data = json.loads(results_path.read_text())
    reasoning = data.get("reasoning") or {}
    cases: list[dict] = []
    for row in data.get("results", []):
        did = row["dilemma_id"]
        key = str(did)
        rs = reasoning.get(key)
        if not rs:
            continue
        n_td, n_ntd = _party_counts(party_df, did)
        n_a, n_b = _option_ns(row["to_do_is_a"], n_td, n_ntd)
        for i, ch in enumerate(row.get("responses", [])):
            if i >= len(rs) or rs[i] is None:
                continue
            if baseline_filter == "all":
                pass
            elif baseline_filter == "smaller":
                if not _chose_smaller(ch, n_a, n_b, min_diff_smaller):
                    continue
            elif baseline_filter == "party_unequal":
                if not _party_unequal_enough(n_a, n_b, party_diff_min):
                    continue
            else:
                raise ValueError(f"Unknown baseline_filter {baseline_filter!r}")
            ek = f"baseline/{did}/t{i}"
            cases.append(
                {
                    "edge_key": ek,
                    "model": model,
                    "factor": "daily_dilemmas",
                    "nudge_type": "baseline",
                    "level_A": str(n_a),
                    "level_B": str(n_b),
                    "option_a_label": "A",
                    "option_b_label": "B",
                    "option_a_n": n_a,
                    "option_b_n": n_b,
                    "f_0_A": 0.0,
                    "f_A_A": 0.0,
                    "f_B_A": 0.0,
                    "nudged_option": "A",
                    "baseline_freq": 0.0,
                    "nudged_freq": 0.0,
                    "nudged_n": n_a,
                    "other_n": n_b,
                    "condition_a_name": "baseline",
                    "condition_b_name": "nudged_towards_A",
                    "condition_a_traces": [
                        {
                            "choice": ch,
                            "reasoning": rs[i],
                            "is_flipped": False,
                        }
                    ],
                    "condition_b_traces": [],
                }
            )
    return cases


def _nudged_cases_for_folder(
    results_path: Path,
    party_df: pd.DataFrame,
    model: str,
    nudge_type: str,
    min_diff: int,
    nudged_filter: str,
) -> list[dict]:
    data = json.loads(results_path.read_text())
    reasoning = data.get("reasoning") or {}
    cases: list[dict] = []
    for row in data.get("results", []):
        direc = row.get("direction")
        if not direc:
            continue
        did = row["dilemma_id"]
        key = f"{did}:{direc}"
        rs = reasoning.get(key)
        if not rs:
            continue
        n_td, n_ntd = _party_counts(party_df, did)
        n_a, n_b = _option_ns(row["to_do_is_a"], n_td, n_ntd)
        for i, ch in enumerate(row.get("responses", [])):
            if i >= len(rs) or rs[i] is None:
                continue
            if nudged_filter == "smaller":
                if not _chose_smaller(ch, n_a, n_b, min_diff):
                    continue
            elif nudged_filter == "all":
                pass
            else:
                raise ValueError(f"Unknown nudged_filter {nudged_filter!r}")
            ek = f"{nudge_type}/{did}/{direc}/t{i}"
            cases.append(
                {
                    "edge_key": ek,
                    "model": model,
                    "factor": "daily_dilemmas",
                    "nudge_type": nudge_type,
                    "level_A": str(n_a),
                    "level_B": str(n_b),
                    "option_a_label": "A",
                    "option_b_label": "B",
                    "option_a_n": n_a,
                    "option_b_n": n_b,
                    "f_0_A": 0.0,
                    "f_A_A": 0.0,
                    "f_B_A": 0.0,
                    "nudged_option": "A",
                    "baseline_freq": 0.0,
                    "nudged_freq": 0.0,
                    "nudged_n": n_a,
                    "other_n": n_b,
                    "condition_a_name": "baseline",
                    "condition_b_name": f"nudged_{direc}",
                    "condition_a_traces": [],
                    "condition_b_traces": [
                        {
                            "choice": ch,
                            "reasoning": rs[i],
                            "is_flipped": False,
                        }
                    ],
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Fig. 9 rationale inputs from Daily Dilemmas results.json"
    )
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Experiment root containing per-nudge subdirs with results.json",
    )
    parser.add_argument(
        "--party-csv",
        type=str,
        default="analysis/dd_dataset/action_to_party_to_value.csv",
        help="CSV with party rows per dilemma action",
    )
    parser.add_argument(
        "--output-baseline",
        type=str,
        default="analysis/dd_fig9/edges_baseline_smaller.json",
    )
    parser.add_argument(
        "--output-nudged",
        type=str,
        default="analysis/dd_fig9/edges_nudged_smaller.json",
    )
    parser.add_argument(
        "--min-n-diff",
        type=int,
        default=2,
        help=(
            "Minimum |n_A - n_B| for nudged traces with chose-smaller filter, "
            "and for baseline when --baseline-filter smaller (default: 2)"
        ),
    )
    parser.add_argument(
        "--baseline-filter",
        choices=["all", "smaller", "party_unequal"],
        default="all",
        help=(
            "Baseline: 'all' = every non-null reasoning trace (default, no party "
            "filter); 'smaller' / 'party_unequal' = party-row filters"
        ),
    )
    parser.add_argument(
        "--nudged-filter",
        choices=["all", "smaller"],
        default="all",
        help=(
            "Nudged: 'all' = every non-null trace (default); "
            "'smaller' = chose fewer parties with --min-n-diff"
        ),
    )
    parser.add_argument(
        "--baseline-party-diff-min",
        type=int,
        default=1,
        help="Minimum |n_A - n_B| when --baseline-filter party_unequal (default: 1)",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    party_df = pd.read_csv(args.party_csv)

    baseline_path = root / "baseline" / "results.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    model = json.loads(baseline_path.read_text()).get("model", "unknown")

    bl_cases = _baseline_cases(
        baseline_path,
        party_df,
        model,
        baseline_filter=args.baseline_filter,
        min_diff_smaller=args.min_n_diff,
        party_diff_min=args.baseline_party_diff_min,
    )
    bl_kind = {
        "all": "baseline_all_traces",
        "smaller": "baseline_smaller_group",
        "party_unequal": "baseline_party_unequal",
    }[args.baseline_filter]
    _save(
        bl_cases,
        {
            "source": str(baseline_path),
            "baseline_filter": args.baseline_filter,
            "min_n_diff": args.min_n_diff,
            "baseline_party_diff_min": args.baseline_party_diff_min,
            "kind": bl_kind,
        },
        args.output_baseline,
    )
    if len(bl_cases) < 300 and args.baseline_filter == "party_unequal":
        print(
            f"Warning: only {len(bl_cases)} baseline traces — "
            "try lowering --baseline-party-diff-min or check data coverage."
        )

    nudge_dirs = sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and d.name != "baseline" and (d / "results.json").is_file()
    )
    all_nudged: list[dict] = []
    for nd in nudge_dirs:
        all_nudged.extend(
            _nudged_cases_for_folder(
                nd / "results.json",
                party_df,
                model,
                nd.name,
                args.min_n_diff,
                args.nudged_filter,
            )
        )
    nudged_kind = (
        "nudged_smaller_group"
        if args.nudged_filter == "smaller"
        else "nudged_all_traces"
    )
    _save(
        all_nudged,
        {
            "results_root": str(root),
            "nudge_folders": [d.name for d in nudge_dirs],
            "min_n_diff": args.min_n_diff,
            "nudged_filter": args.nudged_filter,
            "kind": nudged_kind,
        },
        args.output_nudged,
    )


if __name__ == "__main__":
    main()
