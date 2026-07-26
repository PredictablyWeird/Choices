#!/usr/bin/env python3
"""Build a blind human-labelling sheet for the compliance judge.

Reads ``2000_sampled_compliance.json`` (Gemini-3-Flash compliance labels on the
reasoning traces), reconstructs the context each trace was judged in (option
labels + choice as the model saw them, and the influence description for the
nudge), then draws a sample stratified by compliance category.

Two outputs:
- ``compliance_sheet.csv`` -- annotator-facing, NO judge label (blind).
- ``compliance_key.csv``   -- hidden key: id -> compliance_category + metadata.

Usage:
    uv run python experiments/judge-validation/sample_compliance.py --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
for p in (str(_SCRIPT_DIR), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from _shared import stratified_sample, write_key, write_sheet  # noqa: E402

from choices.experiments.nudging.templates import (  # noqa: E402
    get_influence_description,
)

_DEFAULT_INPUT = _REPO_ROOT / "2000_sampled_compliance.json"

SHEET_COLUMNS = [
    "id",
    "model",
    "factor",
    "nudge_type",
    "influence_description",
    "nudged_group",
    "option_a",
    "option_b",
    "model_choice",
    "reasoning",
    "human_label",
    "human_notes",
]

KEY_COLUMNS = [
    "id",
    "compliance_category",
    "mentions_influence",
    "model",
    "factor",
    "nudge_type",
]


def reconstruct_context(case: dict, trace: dict) -> dict:
    """Recreate what the compliance judge saw for a single nudged trace.

    Mirrors ``build_compliance_prompt`` / ``build_annotated_traces`` in
    ``compliance_classification.py``: resolve the nudged group, render the
    influence description, and swap option labels + choice back when the trial
    was presented flipped.
    """
    condition_name = case.get("condition_b_name", "nudged")
    level_a = case.get("level_A", "A")
    level_b = case.get("level_B", "B")
    if "towards_A" in condition_name:
        nudged_group, other_group = level_a, level_b
    else:
        nudged_group, other_group = level_b, level_a

    influence = get_influence_description(
        case.get("nudge_type", ""),
        group_label=nudged_group,
        other_group_label=other_group,
    )

    a_label = case.get("option_a_label", "")
    b_label = case.get("option_b_label", "")
    choice = trace.get("choice")
    if trace.get("is_flipped"):
        a_label, b_label = b_label, a_label
        choice = "B" if choice == "A" else "A"

    return {
        "influence_description": influence or "Unknown influence.",
        "nudged_group": nudged_group,
        "option_a": a_label,
        "option_b": b_label,
        "model_choice": choice,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(_DEFAULT_INPUT))
    ap.add_argument("--n", type=int, default=200, help="Target sample size")
    ap.add_argument("--min-per-stratum", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-reasoning-chars", type=int, default=2000)
    ap.add_argument("--out-dir", default=str(_SCRIPT_DIR / "sheets"))
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)
    cases = data.get("cases", [])
    print(f"Loaded {len(cases)} compliance cases from {args.input}")

    candidates: list[dict] = []
    for case in cases:
        for trace in case.get("condition_b_traces", []):
            comp = trace.get("compliance")
            reasoning = (trace.get("reasoning") or "").strip()
            if not comp or not comp.get("compliance_category") or not reasoning:
                continue
            # option/influence context is per-case, but choice + flip are
            # per-trace, so recompute the trace-dependent bits each time.
            ctx = reconstruct_context(case, trace)
            candidates.append(
                {
                    "model": case.get("model"),
                    "factor": case.get("factor"),
                    "nudge_type": case.get("nudge_type"),
                    "influence_description": ctx["influence_description"],
                    "nudged_group": ctx["nudged_group"],
                    "option_a": ctx["option_a"],
                    "option_b": ctx["option_b"],
                    "model_choice": ctx["model_choice"],
                    "reasoning": reasoning[: args.max_reasoning_chars],
                    "compliance_category": comp.get("compliance_category"),
                    "mentions_influence": comp.get("mentions_influence"),
                }
            )
    print(f"{len(candidates)} labelled traces available as candidates")

    sample = stratified_sample(
        candidates,
        key_fn=lambda r: r["compliance_category"],
        n=args.n,
        seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )

    for i, r in enumerate(sample):
        r["id"] = f"cmp-{i:04d}"
        r["human_label"] = ""
        r["human_notes"] = ""

    out_dir = Path(args.out_dir)
    sheet_path = out_dir / "compliance_sheet.csv"
    key_path = out_dir / "compliance_key.csv"
    write_sheet(sheet_path, sample, SHEET_COLUMNS)
    write_key(key_path, sample, KEY_COLUMNS)

    print(f"\nWrote {len(sample)} rows:")
    print(f"  sheet (blind): {sheet_path}")
    print(f"  key   (hidden): {key_path}")

    print("\nSampled compliance-category distribution:")
    for lbl, c in Counter(r["compliance_category"] for r in sample).most_common():
        print(f"  {lbl:26s} {c}")


if __name__ == "__main__":
    main()
