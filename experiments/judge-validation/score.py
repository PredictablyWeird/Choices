#!/usr/bin/env python3
"""Score completed human-annotation sheets against the hidden judge key.

Given one or more filled-in annotator sheets (each with an ``id`` and a
``human_label`` column) plus the hidden key produced by the sampler, this
reports:

- per-annotator vs judge: raw agreement + Cohen's kappa;
- inter-annotator: pairwise Cohen's kappa (if >= 2 annotators);
- human ground truth vs judge: confusion matrix, per-class precision/recall/F1
  for the judge (human = ground truth);
- a task-specific headline recompute:
    * turn2      -> ACK_DISCLAIMED share among representative sig_backfire
                    trials (the paper's 78%), under judge vs human labels;
    * compliance -> full category distribution under judge vs human.

Usage:
    uv run python experiments/judge-validation/score.py \\
        --task turn2 \\
        --key experiments/judge-validation/sheets/turn2_key.csv \\
        --sheets annotatorA.csv annotatorB.csv [--adjudicated gold.csv]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _shared import (  # noqa: E402
    TASK_LABELS,
    cohen_kappa,
    confusion_matrix,
    majority_vote,
    normalize_label,
    precision_recall_f1,
    raw_agreement,
    read_csv_rows,
)

TASK_KEY_LABEL = {"turn2": "judge_label", "compliance": "compliance_category"}


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"


def load_annotator(path: str) -> dict[str, str]:
    rows = read_csv_rows(path)
    out: dict[str, str] = {}
    for r in rows:
        rid = r.get("id")
        lbl = normalize_label(r.get("human_label"))
        if rid and lbl:
            out[rid] = lbl
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["turn2", "compliance"])
    ap.add_argument("--key", required=True, help="Hidden key CSV from the sampler")
    ap.add_argument(
        "--sheets",
        required=True,
        nargs="+",
        help="One filled sheet per annotator (id + human_label)",
    )
    ap.add_argument(
        "--adjudicated",
        default=None,
        help="Optional gold sheet (id + human_label) with disagreements "
        "resolved. If given, it is used as the human ground truth for "
        "sections [3] and [4] (recommended for 2 annotators).",
    )
    ap.add_argument(
        "--strict-labels",
        action="store_true",
        help="Error out on any human_label not in the task taxonomy",
    )
    args = ap.parse_args()

    categories = list(TASK_LABELS[args.task].keys())
    key_label_col = TASK_KEY_LABEL[args.task]

    key_rows = read_csv_rows(args.key)
    key = {r["id"]: r for r in key_rows}
    judge = {r["id"]: normalize_label(r.get(key_label_col)) for r in key_rows}

    annotators = {Path(p).stem: load_annotator(p) for p in args.sheets}

    # Validate labels
    unknown = defaultdict(set)
    for name, labs in annotators.items():
        for rid, lbl in labs.items():
            if lbl not in categories:
                unknown[name].add(lbl)
    if unknown:
        print("WARNING: human labels outside the taxonomy were found:")
        for name, labs in unknown.items():
            print(f"  {name}: {sorted(labs)}")
        if args.strict_labels:
            sys.exit("Aborting due to --strict-labels.")
        print()

    common_ids = [rid for rid in key if any(rid in a for a in annotators.values())]
    print("=" * 68)
    print(
        f"TASK: {args.task}   |   key rows: {len(key)}   |   "
        f"annotators: {len(annotators)}"
    )
    print(f"Labelled ids covered by >=1 annotator: {len(common_ids)}")
    print("=" * 68)

    # ---- Per-annotator vs judge ----
    print("\n[1] Per-annotator agreement with the LLM judge")
    print(f"  {'annotator':22s} {'n':>5s} {'raw':>8s} {'kappa':>8s}")
    for name, labs in annotators.items():
        ids = [rid for rid in labs if rid in judge]
        h = [labs[i] for i in ids]
        j = [judge[i] for i in ids]
        print(
            f"  {name:22s} {len(ids):5d} {_fmt(raw_agreement(h, j)):>8s} "
            f"{_fmt(cohen_kappa(h, j)):>8s}"
        )

    # ---- Inter-annotator ----
    if len(annotators) >= 2:
        print("\n[2] Inter-annotator agreement (pairwise Cohen's kappa)")
        names = list(annotators)
        for a, b in combinations(names, 2):
            ids = [i for i in annotators[a] if i in annotators[b]]
            la = [annotators[a][i] for i in ids]
            lb = [annotators[b][i] for i in ids]
            print(
                f"  {a} vs {b}: n={len(ids)}  raw={_fmt(raw_agreement(la, lb))}  "
                f"kappa={_fmt(cohen_kappa(la, lb))}"
            )
    else:
        print(
            "\n[2] Inter-annotator agreement: only one annotator sheet given " "(skip)."
        )

    # ---- Establish the human "ground truth" for [3]/[4] ----
    # Priority: adjudicated gold > majority vote (>=3 annotators) > pooled.
    pairs: list[tuple[str, str]] = []
    pair_ids: list[str] = []
    mode = ""

    if args.adjudicated:
        gold = load_annotator(args.adjudicated)
        mode = f"adjudicated gold ({Path(args.adjudicated).stem})"
        for rid, hl in gold.items():
            if judge.get(rid):
                pairs.append((hl, judge[rid]))
                pair_ids.append(rid)
    elif len(annotators) >= 3:
        mode = "majority vote (>=3 annotators; ties excluded)"
        ties = 0
        for rid in key:
            if not judge.get(rid):
                continue
            votes = [a[rid] for a in annotators.values() if rid in a]
            if not votes:
                continue
            lbl, is_tie = majority_vote(votes)
            if is_tie:
                ties += 1
                continue
            pairs.append((lbl, judge[rid]))
            pair_ids.append(rid)
        mode += f" [{ties} ties dropped]"
    else:
        mode = (
            "pooled annotator labels (each annotation is one instance; "
            "no consensus formed). Provide --adjudicated for a single gold "
            "truth."
        )
        for name, labs in annotators.items():
            for rid, hl in labs.items():
                if judge.get(rid):
                    pairs.append((hl, judge[rid]))
                    pair_ids.append(rid)

    h = [p[0] for p in pairs]
    j = [p[1] for p in pairs]
    print("\n[3] Human ground truth vs judge")
    print(f"  truth mode: {mode}")
    print(f"  instances: {len(pairs)}")
    print(
        f"  raw agreement: {_fmt(raw_agreement(h, j))}   "
        f"Cohen's kappa: {_fmt(cohen_kappa(h, j))}"
    )

    print("\n  Per-class judge precision/recall/F1 (human = truth):")
    print(
        f"    {'label':26s} {'prec':>6s} {'rec':>6s} {'f1':>6s} "
        f"{'tp':>4s} {'fp':>4s} {'fn':>4s}"
    )
    for cat in categories:
        m = precision_recall_f1(h, j, cat)
        print(
            f"    {cat:26s} {_fmt(m['precision']):>6s} {_fmt(m['recall']):>6s} "
            f"{_fmt(m['f1']):>6s} {m['tp']:>4d} {m['fp']:>4d} {m['fn']:>4d}"
        )

    print("\n  Confusion matrix (rows = human, cols = judge):")
    cm = confusion_matrix(h, j, categories)
    short = [c[:10] for c in categories]
    print("    " + "".join(f"{s:>12s}" for s in [""] + short))
    for t in categories:
        print(f"    {t[:12]:12s}" + "".join(f"{cm[t][p]:>12d}" for p in categories))

    # ---- Task-specific headline recompute ----
    print("\n[4] Headline recompute")
    if args.task == "turn2":
        # Only the representative backfire draw is unbiased for the rate; the
        # taxonomy draw is label-stratified and would distort it.
        bf = [
            (hl, jl, rid)
            for (hl, jl), rid in zip(pairs, pair_ids)
            if key[rid].get("condition_kind") == "sig_backfire"
            and key[rid].get("sample_group", "headline") in ("headline", "both")
        ]
        if not bf:
            print("  No representative sig_backfire instances with human labels.")
        else:
            j_rate = sum(1 for _, jl, _ in bf if jl == "ACK_DISCLAIMED") / len(bf)
            h_rate = sum(1 for hl, _, _ in bf if hl == "ACK_DISCLAIMED") / len(bf)
            n_bf_ids = len({rid for _, _, rid in bf})
            print(
                f"  ACK_DISCLAIMED share among sig_backfire "
                f"(instances={len(bf)}, unique trials={n_bf_ids}):"
            )
            print(f"    judge labels : {j_rate:.1%}")
            print(f"    human labels : {h_rate:.1%}")
            print("    paper headline: 78%")
    else:
        print("  Compliance category distribution:")
        jc = Counter(j)
        hc = Counter(h)
        n = len(pairs) or 1
        print(f"    {'category':26s} {'judge':>10s} {'human':>10s}")
        for cat in categories:
            print(f"    {cat:26s} {jc.get(cat, 0)/n:>9.1%} {hc.get(cat, 0)/n:>9.1%}")


if __name__ == "__main__":
    main()
