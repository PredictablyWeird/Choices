#!/usr/bin/env python3
"""
Cross-model investigation of backfiring cases.

Produces comprehensive statistics on backfire patterns across all models,
factors, nudge types, and N-differences.

Usage:
    uv run python -m choices.analysis.reasoning_traces.investigate_backfires \
        --results-dirs results/main_results/results_main0
"""

import argparse
import json
from collections import Counter, defaultdict

from choices.analysis.reasoning_traces.case_study_backfire import (
    find_backfire_cases,
)
from choices.analysis.reasoning_traces.edge_filtering import (
    EdgeFilteringPipeline,
)


def investigate_backfires(
    results_dirs: list[str],
    min_effect: float = 0.0,
    output: str | None = None,
) -> dict:
    """Run cross-model backfire investigation."""

    # Extract all edges (no filter)
    pipeline = EdgeFilteringPipeline(results_dirs=results_dirs)
    all_edges = pipeline.get_edges()
    print(f"Total edges across all models: {len(all_edges)}")

    # Group by model
    by_model: dict[str, list] = defaultdict(list)
    for edge in all_edges:
        by_model[edge.model].append(edge)

    results: dict = {"models": {}, "summary": {}}

    for model, edges in sorted(by_model.items()):
        cases = find_backfire_cases(edges, min_effect=min_effect)
        n_edges = len(edges)
        n_cases = len(cases)

        # Break down by factor
        by_factor = Counter(c.edge.factor for c in cases)
        # Break down by nudge type
        by_nudge = Counter(c.edge.nudge_type for c in cases)
        # Nudged has smaller vs larger N
        nudged_smaller = sum(1 for c in cases if c.nudged_has_smaller_n)
        nudged_larger = sum(1 for c in cases if c.nudged_has_larger_n)
        nudged_equal = n_cases - nudged_smaller - nudged_larger

        # Strong backfires (effect > 0.2)
        strong = [c for c in cases if c.effect_magnitude > 0.2]
        # Very strong (effect > 0.4)
        very_strong = [c for c in cases if c.effect_magnitude > 0.4]

        # Effect magnitude by factor
        effect_by_factor: dict[str, list[float]] = defaultdict(list)
        for c in cases:
            effect_by_factor[c.edge.factor].append(c.effect_magnitude)

        model_results = {
            "total_edges": n_edges,
            "backfire_cases": n_cases,
            "backfire_rate": n_cases / n_edges if n_edges > 0 else 0,
            "strong_backfires": len(strong),
            "very_strong_backfires": len(very_strong),
            "by_factor": dict(by_factor.most_common()),
            "by_nudge_type": dict(by_nudge.most_common()),
            "nudged_smaller_n": nudged_smaller,
            "nudged_larger_n": nudged_larger,
            "nudged_equal_n": nudged_equal,
            "effect_by_factor": {
                f: {
                    "count": len(effects),
                    "mean": sum(effects) / len(effects),
                    "max": max(effects),
                }
                for f, effects in sorted(effect_by_factor.items())
            },
        }
        results["models"][model] = model_results

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        print(
            f"  Edges: {n_edges}, Backfires: {n_cases} ({model_results['backfire_rate']:.1%})"
        )
        print(f"  Strong (>0.2): {len(strong)}, Very strong (>0.4): {len(very_strong)}")
        print(f"  By factor: {dict(by_factor.most_common())}")
        print(f"  By nudge: {dict(by_nudge.most_common())}")
        print(
            f"  Nudged has smaller N: {nudged_smaller}, larger N: {nudged_larger}, equal: {nudged_equal}"
        )

        if very_strong:
            print("\n  Top very strong backfires:")
            for c in sorted(
                very_strong, key=lambda x: x.effect_magnitude, reverse=True
            )[:5]:
                e = c.edge
                print(
                    f"    {e.factor}/{e.nudge_type}: "
                    f"{e.option_a_n} {e.level_A} vs {e.option_b_n} {e.level_B} | "
                    f"nudged={c.nudged_option}, effect={c.effect_magnitude:.3f}, "
                    f"baseline={c.baseline_freq:.3f}→nudged={c.nudged_freq:.3f}"
                )

    # Cross-model summary
    all_cases = find_backfire_cases(all_edges, min_effect=min_effect)
    total_by_factor = Counter(c.edge.factor for c in all_cases)
    total_by_nudge = Counter(c.edge.nudge_type for c in all_cases)

    results["summary"] = {
        "total_edges": len(all_edges),
        "total_backfires": len(all_cases),
        "overall_backfire_rate": len(all_cases) / len(all_edges) if all_edges else 0,
        "by_factor": dict(total_by_factor.most_common()),
        "by_nudge_type": dict(total_by_nudge.most_common()),
    }

    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*60}")
    print(f"Total edges: {len(all_edges)}")
    print(
        f"Total backfires: {len(all_cases)} ({results['summary']['overall_backfire_rate']:.1%})"
    )
    print(f"By factor: {dict(total_by_factor.most_common())}")
    print(f"By nudge type: {dict(total_by_nudge.most_common())}")

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {output}")

    return results


def investigate_extreme_baselines(
    results_dirs: list[str],
    min_baseline_pref: float = 0.6,
) -> None:
    """Find edges where baseline makes surprising choices (prefers fewer people)."""

    pipeline = EdgeFilteringPipeline(results_dirs=results_dirs)
    all_edges = pipeline.get_edges()

    surprising = []
    for edge in all_edges:
        # A has fewer people but baseline prefers A
        if edge.option_a_n < edge.option_b_n and edge.f_0_A > min_baseline_pref:
            surprising.append(("A", edge))
        # B has fewer people but baseline prefers B
        elif edge.option_b_n < edge.option_a_n and edge.f_0_B > min_baseline_pref:
            surprising.append(("B", edge))

    print(f"\n{'='*60}")
    print(
        f"EXTREME BASELINE: Prefers option with fewer people (threshold={min_baseline_pref})"
    )
    print(f"{'='*60}")
    print(f"Found {len(surprising)} cases")

    # Group by model
    by_model = defaultdict(list)
    for preferred, edge in surprising:
        by_model[edge.model].append((preferred, edge))

    for model, cases in sorted(by_model.items()):
        print(f"\n  {model}: {len(cases)} cases")
        # Group by factor
        by_factor = Counter(e.factor for _, e in cases)
        print(f"    By factor: {dict(by_factor.most_common())}")

        # Show top examples
        sorted_cases = sorted(cases, key=lambda x: x[1].n_difference, reverse=True)
        for preferred, e in sorted_cases[:3]:
            pref_n = e.option_a_n if preferred == "A" else e.option_b_n
            other_n = e.option_b_n if preferred == "A" else e.option_a_n
            f_pref = e.f_0_A if preferred == "A" else e.f_0_B
            print(
                f"    {e.factor}/{e.nudge_type}: prefers {pref_n} over {other_n} "
                f"(f={f_pref:.3f}, n_diff={e.n_difference})"
            )


def investigate_few_shot_at_unequal_n(
    results_dirs: list[str],
    min_n_diff: int = 3,
) -> None:
    """Compare few-shot effects vs other nudge types at very unequal N."""

    pipeline = EdgeFilteringPipeline(results_dirs=results_dirs)
    all_edges = pipeline.get_edges()

    # Filter to unequal N
    unequal = [e for e in all_edges if e.n_difference >= min_n_diff]
    print(f"\n{'='*60}")
    print(f"FEW-SHOT EFFECTS AT UNEQUAL N (n_diff >= {min_n_diff})")
    print(f"{'='*60}")
    print(f"Edges with n_diff >= {min_n_diff}: {len(unequal)}")

    # Group by nudge_type
    by_nudge: dict[str, list[float]] = defaultdict(list)
    by_nudge_backfire: dict[str, int] = Counter()

    for edge in unequal:
        # Use max of effect_A and effect_B magnitude
        max_effect = max(edge.effect_A_magnitude, edge.effect_B_magnitude)
        by_nudge[edge.nudge_type].append(max_effect)
        if edge.backfire_A or edge.backfire_B:
            by_nudge_backfire[edge.nudge_type] += 1

    print("\n  Effect by nudge type:")
    for nudge_type in sorted(by_nudge.keys()):
        effects = by_nudge[nudge_type]
        n = len(effects)
        mean_eff = sum(effects) / n
        max_eff = max(effects)
        backfire_count = by_nudge_backfire.get(nudge_type, 0)
        print(
            f"    {nudge_type}: n={n}, mean_effect={mean_eff:.3f}, "
            f"max={max_eff:.3f}, backfires={backfire_count} ({backfire_count/n:.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description="Cross-model backfire investigation")
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results/main_results/results_main0"],
    )
    parser.add_argument("--min-effect", type=float, default=0.0)
    parser.add_argument("--output", "-o", type=str, default=None)

    args = parser.parse_args()

    investigate_backfires(
        results_dirs=args.results_dirs,
        min_effect=args.min_effect,
        output=args.output,
    )

    investigate_extreme_baselines(
        results_dirs=args.results_dirs,
        min_baseline_pref=0.6,
    )

    investigate_few_shot_at_unequal_n(
        results_dirs=args.results_dirs,
        min_n_diff=3,
    )


if __name__ == "__main__":
    main()
