#!/usr/bin/env python3
"""Analyze classification results from reasoning traces."""

import json
from collections import Counter


def analyze_classifications(filepath: str):
    """Analyze classification results."""
    with open(filepath) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} classified traces\n")

    # Extract classifications
    classifications = [d["classification"] for d in data if d.get("classification")]
    print(f"Valid classifications: {len(classifications)}\n")

    # 1. Reason distributions
    print("=" * 60)
    print("REASON DISTRIBUTIONS")
    print("=" * 60)

    reason_counts = Counter()
    reason_valences = {
        "endorsed": Counter(),
        "rejected": Counter(),
        "mentioned": Counter(),
    }

    for c in classifications:
        reasons = c.get("reasons", {})
        for reason_name, value in reasons.items():
            if isinstance(value, dict):
                valence = value.get("valence", "absent")
            else:
                valence = value
            if valence != "absent":
                reason_counts[reason_name] += 1
                # Normalize valence to expected values
                if valence in ("endorsed", "rejected", "mentioned"):
                    reason_valences[valence][reason_name] += 1
                else:
                    # Unexpected valence, treat as "mentioned"
                    reason_valences["mentioned"][reason_name] += 1

    print("\nReasons present (any valence):")
    for reason, count in reason_counts.most_common():
        pct = count / len(classifications) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    print("\nReasons by valence:")
    for valence in ["endorsed", "rejected", "mentioned"]:
        print(f"\n  {valence.upper()}:")
        for reason, count in reason_valences[valence].most_common(5):
            pct = count / len(classifications) * 100
            print(f"    {reason}: {count} ({pct:.1f}%)")

    # 2. Rhetorical moves
    print("\n" + "=" * 60)
    print("RHETORICAL MOVES")
    print("=" * 60)

    rhetorical_counts = Counter()
    for c in classifications:
        moves = c.get("rhetorical_moves", {})
        for move_name, present in moves.items():
            if present:
                rhetorical_counts[move_name] += 1

    print("\nMoves present:")
    for move, count in rhetorical_counts.most_common():
        pct = count / len(classifications) * 100
        print(f"  {move}: {count} ({pct:.1f}%)")

    # 3. Process markers
    print("\n" + "=" * 60)
    print("PROCESS MARKERS")
    print("=" * 60)

    reasoning_lengths = Counter()
    confidence_levels = Counter()
    defaults_to_a = 0
    shows_deliberation = 0
    changes_mind = 0

    for c in classifications:
        process = c.get("process", {})
        reasoning_lengths[process.get("reasoning_length", "unknown")] += 1
        confidence_levels[process.get("confidence_level", "unknown")] += 1
        if process.get("defaults_to_A"):
            defaults_to_a += 1
        if process.get("shows_deliberation"):
            shows_deliberation += 1
        if process.get("changes_mind"):
            changes_mind += 1

    print("\nReasoning length:")
    for length, count in sorted(reasoning_lengths.items()):
        pct = count / len(classifications) * 100
        print(f"  {length}: {count} ({pct:.1f}%)")

    print("\nConfidence level:")
    for level, count in sorted(confidence_levels.items()):
        pct = count / len(classifications) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")

    print(
        f"\nDefaults to A: {defaults_to_a} ({defaults_to_a/len(classifications)*100:.1f}%)"
    )
    print(
        f"Shows deliberation: {shows_deliberation} ({shows_deliberation/len(classifications)*100:.1f}%)"
    )
    print(
        f"Changes mind: {changes_mind} ({changes_mind/len(classifications)*100:.1f}%)"
    )

    # 4. Primary reasons
    print("\n" + "=" * 60)
    print("PRIMARY REASONS")
    print("=" * 60)

    primary_reasons = Counter()
    for c in classifications:
        primary = c.get("primary_reason", "unknown")
        primary_reasons[primary] += 1

    print("\nMost common primary reasons:")
    for reason, count in primary_reasons.most_common(10):
        pct = count / len(classifications) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # 5. Cross-analysis: reasons by factor
    print("\n" + "=" * 60)
    print("REASONS BY DEMOGRAPHIC FACTOR")
    print("=" * 60)

    factor_reasons = {}
    for d in data:
        factor = d.get("factor", "unknown")
        c = d.get("classification", {})
        if factor not in factor_reasons:
            factor_reasons[factor] = Counter()

        reasons = c.get("reasons", {})
        for reason_name, value in reasons.items():
            if isinstance(value, dict):
                valence = value.get("valence", "absent")
            else:
                valence = value
            if valence == "endorsed":
                factor_reasons[factor][reason_name] += 1

    for factor in sorted(factor_reasons.keys()):
        print(f"\n{factor}:")
        for reason, count in factor_reasons[factor].most_common(3):
            print(f"  {reason}: {count}")

    # 6. Backfire analysis
    print("\n" + "=" * 60)
    print("BACKFIRE ANALYSIS")
    print("=" * 60)

    backfire_traces = [
        d
        for d in data
        if d.get("chose_nudged_group") is False and d.get("condition") != "base"
    ]
    follow_traces = [
        d
        for d in data
        if d.get("chose_nudged_group") is True and d.get("condition") != "base"
    ]
    base_traces = [d for d in data if d.get("condition") == "base"]

    print(f"\nTraces following nudge: {len(follow_traces)}")
    print(f"Traces backfiring against nudge: {len(backfire_traces)}")
    print(f"Baseline traces (no nudge): {len(base_traces)}")

    # Compare rhetorical moves in backfire vs follow
    print("\nRhetorical moves in BACKFIRE traces:")
    backfire_moves = Counter()
    for d in backfire_traces:
        c = d.get("classification", {})
        moves = c.get("rhetorical_moves", {})
        for move_name, present in moves.items():
            if present:
                backfire_moves[move_name] += 1

    for move, count in backfire_moves.most_common(5):
        pct = count / len(backfire_traces) * 100 if backfire_traces else 0
        print(f"  {move}: {count} ({pct:.1f}%)")

    print("\nRhetorical moves in FOLLOW traces:")
    follow_moves = Counter()
    for d in follow_traces:
        c = d.get("classification", {})
        moves = c.get("rhetorical_moves", {})
        for move_name, present in moves.items():
            if present:
                follow_moves[move_name] += 1

    for move, count in follow_moves.most_common(5):
        pct = count / len(follow_traces) * 100 if follow_traces else 0
        print(f"  {move}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import sys

    filepath = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "analysis/equal_n_classifications_sample.json"
    )
    analyze_classifications(filepath)
