#!/usr/bin/env python3
"""Build a blind human-labelling sheet for the turn-2 self-report judge.

Reads the classified follow-up-probe records (which carry the raw
``turn2_response`` free text plus the gpt-4o-mini ``judge_label``), joins each
trial to its condition-level outcome (``condition_kind`` = sig_backfire /
sig_compliance / no_effect / base) from the annotated trials CSV, then draws a
stratified sample.

Two outputs:
- ``turn2_sheet.csv``  -- annotator-facing, NO judge label (blind).
- ``turn2_key.csv``    -- hidden key: id -> judge_label + condition metadata.

The sheet is the union of two deliberately different draws, tagged in the key
by ``sample_group`` (this matters -- do NOT collapse them):

- ``headline``: a *representative* random sample of significant-backfire trials
  with NO stratification by judge label, so the ACK_DISCLAIMED proportion is
  unbiased and the 78% headline can be honestly recomputed on human labels.
- ``taxonomy``: a sample across ALL trials *stratified by judge label*, so rare
  labels (DENIED / PARTIAL / UNCLEAR) get enough coverage for per-class
  precision/recall and overall Cohen's kappa.

Trials appearing in both draws are tagged ``both`` (still representative for the
headline). The scorer recomputes the 78% only over headline/both backfire rows.

Usage:
    uv run python experiments/judge-validation/sample_turn2.py \\
        --backfire-n 120 --taxonomy-n 100
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _shared import stratified_sample, write_key, write_sheet  # noqa: E402

_PROBE_DIR = _SCRIPT_DIR.parent / "followup-probe"
_DEFAULT_RESULTS_GLOB = "classified_v2_*.jsonl.gz"
_DEFAULT_ANNOTATED = _PROBE_DIR / "analysis_out_v2" / "trials_annotated.csv.gz"

# Condition-key fields shared between the classified records' ``spec`` and the
# annotated trials CSV. edge_id is a list in the spec / a string in the CSV, so
# it is normalised on both sides.
_JOIN_FIELDS = (
    "benchmark",
    "model",
    "factor",
    "nudge_type",
    "target_group",
    "reasoning_condition",
    "edge_id",
    "direction",
)


def _norm_edge(v) -> str:
    """Canonicalise edge_id across sources (spec list vs CSV tuple-string)."""
    if isinstance(v, (list, tuple)):
        return str(tuple(v))
    if isinstance(v, str):
        import ast

        try:
            return str(tuple(ast.literal_eval(v)))
        except (ValueError, SyntaxError):
            return v
    return str(v)


SHEET_COLUMNS = [
    "id",
    "benchmark",
    "model",
    "factor",
    "nudge_type",
    "direction",
    "turn1_system_prompt",
    "turn1_user_prompt",
    "model_choice",
    "turn2_response",
    "human_name",
    "human_label",
    "human_notes",
]

KEY_COLUMNS = [
    "id",
    "judge_label",
    "sample_group",
    "condition_kind",
    "is_significant",
    "is_backfire",
    "benchmark",
    "model",
    "factor",
    "nudge_type",
    "direction",
]


def _condition_key(d: dict) -> tuple:
    return tuple(
        _norm_edge(d.get(f)) if f == "edge_id" else str(d.get(f)) for f in _JOIN_FIELDS
    )


def load_condition_kinds(annotated_path: Path) -> dict[tuple, dict]:
    """condition_key -> {condition_kind, is_significant, is_backfire}."""
    lookup: dict[tuple, dict] = {}
    with gzip.open(annotated_path, "rt") as f:
        for row in csv.DictReader(f):
            lookup[_condition_key(row)] = {
                "condition_kind": row.get("condition_kind"),
                "is_significant": row.get("is_significant"),
                "is_backfire": row.get("is_backfire"),
            }
    return lookup


def load_classified(results_dir: Path, pattern: str) -> list[dict]:
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")
    records: list[dict] = []
    for fp in files:
        with gzip.open(fp, "rt") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(_PROBE_DIR / "results"))
    ap.add_argument("--pattern", default=_DEFAULT_RESULTS_GLOB)
    ap.add_argument("--annotated", default=str(_DEFAULT_ANNOTATED))
    ap.add_argument(
        "--backfire-n",
        type=int,
        default=120,
        help="Representative (label-unstratified) sig_backfire trials for the "
        "78%% headline recompute",
    )
    ap.add_argument(
        "--taxonomy-n",
        type=int,
        default=100,
        help="Trials stratified by judge label (all outcomes) for per-class "
        "precision/recall and overall kappa",
    )
    ap.add_argument("--min-per-stratum", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(_SCRIPT_DIR / "sheets"))
    args = ap.parse_args()

    records = load_classified(Path(args.results_dir), args.pattern)
    print(f"Loaded {len(records)} classified turn-2 records")

    kinds = load_condition_kinds(Path(args.annotated))
    print(f"Loaded condition kinds for {len(kinds)} conditions")

    # Build candidate rows: valid judge label + non-empty reply + joinable.
    candidates: list[dict] = []
    missing_join = 0
    for rec in records:
        spec = rec.get("spec") or {}
        reply = (rec.get("turn2_response") or "").strip()
        judge = rec.get("judge_label")
        if not reply or not judge:
            continue
        ck = kinds.get(_condition_key(spec))
        if ck is None:
            missing_join += 1
            continue
        candidates.append(
            {
                "uid": _condition_key(spec) + (spec.get("trial_idx"),),
                "benchmark": spec.get("benchmark"),
                "model": spec.get("model"),
                "factor": spec.get("factor"),
                "nudge_type": spec.get("nudge_type"),
                "direction": spec.get("direction"),
                "turn1_system_prompt": spec.get("system_prompt"),
                "turn1_user_prompt": spec.get("user_prompt"),
                "model_choice": rec.get("turn1_parsed"),
                "turn2_response": reply,
                "judge_label": judge,
                "condition_kind": ck["condition_kind"],
                "is_significant": ck["is_significant"],
                "is_backfire": ck["is_backfire"],
            }
        )
    print(f"{len(candidates)} labelling candidates ({missing_join} unjoinable dropped)")

    rng = random.Random(args.seed)

    # (a) Representative sig_backfire draw (no label stratification).
    backfire_pool = [c for c in candidates if c["condition_kind"] == "sig_backfire"]
    rng.shuffle(backfire_pool)
    headline_sample = backfire_pool[: args.backfire_n]

    # (b) Label-stratified draw across ALL trials.
    taxonomy_sample = stratified_sample(
        candidates,
        key_fn=lambda r: r["judge_label"],
        n=args.taxonomy_n,
        seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )

    # Merge + tag group. Trials in both draws are tagged "both".
    by_uid: dict[tuple, dict] = {}
    for c in headline_sample:
        c = dict(c)
        c["sample_group"] = "headline"
        by_uid[c["uid"]] = c
    for c in taxonomy_sample:
        if c["uid"] in by_uid:
            by_uid[c["uid"]]["sample_group"] = "both"
        else:
            c = dict(c)
            c["sample_group"] = "taxonomy"
            by_uid[c["uid"]] = c

    sample = list(by_uid.values())
    rng.shuffle(sample)

    # Assign stable, order-independent ids AFTER shuffling.
    for i, r in enumerate(sample):
        r["id"] = f"t2-{i:04d}"
        r["human_name"] = ""
        r["human_label"] = ""
        r["human_notes"] = ""

    out_dir = Path(args.out_dir)
    sheet_path = out_dir / "turn2_sheet.csv"
    key_path = out_dir / "turn2_key.csv"
    write_sheet(sheet_path, sample, SHEET_COLUMNS)
    write_key(key_path, sample, KEY_COLUMNS)

    print(f"\nWrote {len(sample)} rows:")
    print(f"  sheet (blind): {sheet_path}")
    print(f"  key   (hidden): {key_path}")

    print("\nSample-group distribution:")
    for g, c in Counter(r["sample_group"] for r in sample).most_common():
        print(f"  {g:16s} {c}")
    print("\nSampled judge-label distribution:")
    for lbl, c in Counter(r["judge_label"] for r in sample).most_common():
        print(f"  {lbl:16s} {c}")
    print("\nSampled condition-kind distribution:")
    for k, c in Counter(r["condition_kind"] for r in sample).most_common():
        print(f"  {k:16s} {c}")
    n_bf = sum(
        1
        for r in sample
        if r["condition_kind"] == "sig_backfire"
        and r["sample_group"] in ("headline", "both")
    )
    print(f"\nRepresentative sig_backfire rows (headline recompute): {n_bf}")


if __name__ == "__main__":
    main()
