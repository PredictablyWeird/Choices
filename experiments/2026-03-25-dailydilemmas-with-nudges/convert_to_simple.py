#!/usr/bin/env python3
"""
Convert results_global/ (run_global.py format) into the simple_* directory
layout used by create_summary.py.

Each primary value becomes a binary factor whose two levels are the value
names on the to_do and not_to_do sides.  Every dilemma where that value
appears on *exactly one* side becomes an edge in the preference graph.
Because a single dilemma can involve two different primary values it will
appear in both factor directories (intentional duplication).

Target layout (per value, per model, per nudge_type):

    <output_dir>/simple_<value>/<model>/<nudge_type>/
        base/
            preference_graph_<model>.json
        <value>/                          # nudge toward value
            preference_graph_<model>.json
        <non_value>/                      # nudge toward non-value (one dir per opponent)
            preference_graph_<model>.json

The preference_graph JSON follows the schema consumed by create_summary.py:
  options, edges (with aux_data), variables, analysis_config,
  simple_experiment_config, and nudge_config (for non-base conditions).

Usage:
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/convert_to_simple.py

    # Custom input/output
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/convert_to_simple.py \
        --input results_global --output results_simple

    # Only convert specific values or models
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/convert_to_simple.py \
        --values honesty safety --models llama-33-70b
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
TIMESTAMP = "20260325_000000"


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_condition(input_dir: Path, model: str, condition: str) -> dict | None:
    path = input_dir / model / condition / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _discover_models(input_dir: Path) -> list[str]:
    return sorted(d.name for d in input_dir.iterdir() if d.is_dir())


def _discover_values(input_dir: Path, model: str) -> set[str]:
    baseline = _load_condition(input_dir, model, "baseline")
    if baseline is None:
        return set()
    values: set[str] = set()
    for r in baseline["results"]:
        values.add(r["primary_value_to_do"])
        values.add(r["primary_value_not_to_do"])
    return values


# ---------------------------------------------------------------------------
# Building preference graphs
# ---------------------------------------------------------------------------


def _make_option(opt_id: int, label: str, value_factor: str, value_level: str) -> dict:
    return {
        "id": opt_id,
        "label": label,
        value_factor: value_level,
    }


def _non_value_label(value: str) -> str:
    return f"non_{value}"


def _build_graph_for_value(
    value: str,
    results: list[dict],
    config: dict,
    model: str,
    nudge_type: str | None = None,
    target_group: str | None = None,
) -> dict:
    """Build a preference_graph dict for dilemmas involving *value*.

    Each dilemma where *value* appears on exactly one side produces one edge.
    Option A = the value-side option, Option B = the non-value-side option.
    Responses are mapped so "A" = chose the value-side.

    The factor is binary: *value* vs non_<value>.  create_summary.py
    requires exactly two factor levels.
    """
    factor_name = "primary_value"
    nval_label = _non_value_label(value)

    options: list[dict] = []
    edges: dict[str, dict] = {}
    next_id = 0

    by_dilemma: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_dilemma[r["dilemma_id"]].append(r)

    for dilemma_id, dilemma_results in sorted(by_dilemma.items()):
        first = dilemma_results[0]
        pv_to = first["primary_value_to_do"]
        pv_not = first["primary_value_not_to_do"]

        if pv_to != value and pv_not != value:
            continue

        value_is_to_do = pv_to == value

        id_val = next_id
        next_id += 1
        id_nval = next_id
        next_id += 1

        opt_val = _make_option(id_val, f"d{dilemma_id}_{value}", factor_name, value)
        opt_nval = _make_option(
            id_nval, f"d{dilemma_id}_{nval_label}", factor_name, nval_label
        )
        options.extend([opt_val, opt_nval])

        r = dilemma_results[0]
        to_do_is_a = r["to_do_is_a"]

        # Map raw A/B responses to value-side perspective.
        # In original_parsed, "A" means chose id_val (value-side).
        value_is_a = (value_is_to_do and to_do_is_a) or (
            not value_is_to_do and not to_do_is_a
        )

        original_parsed: list[str] = []
        for resp in r["responses"]:
            if resp not in ("A", "B"):
                continue
            if value_is_a:
                original_parsed.append(resp)
            else:
                original_parsed.append("B" if resp == "A" else "A")

        edge_key = f"({id_val}, {id_nval})"
        edges[edge_key] = {
            "option_A": opt_val["label"],
            "option_B": opt_nval["label"],
            "probability_A": (
                sum(1 for x in original_parsed if x == "A") / len(original_parsed)
                if original_parsed
                else 0.5
            ),
            "aux_data": {
                "original_parsed": original_parsed,
                "flipped_parsed": [],
            },
        }

    variables = [
        {
            "name": factor_name,
            "values": [value, nval_label],
            "description": "Primary moral value",
        }
    ]

    graph: dict = {
        "options": options,
        "edges": edges,
        "training_edges": [
            [options[i]["id"], options[i + 1]["id"]] for i in range(0, len(options), 2)
        ],
        "holdout_edges": None,
        "variables": variables,
        "analysis_config": {
            "fields": {factor_name: "categorical"},
        },
        "simple_experiment_config": {
            "max_requests": config.get("repetitions", config.get("k_per_dilemma", 3)),
            "requests_per_edge": config.get(
                "repetitions", config.get("k_per_dilemma", 3)
            ),
            "seed": config.get("seed", 42),
            "reasoning_mode": config.get("reasoning_mode", "none"),
        },
    }

    if nudge_type is not None and target_group is not None:
        graph["nudge_config"] = {
            "nudge_type": nudge_type,
            "target_group": target_group,
            "nudge_text": "",
            "nudge_position": "after_system",
            "nudge_brackets": True,
        }

    return graph


def _filter_results_for_value(results: list[dict], value: str) -> list[dict]:
    """Keep only results where *value* is one of the primary values."""
    return [
        r
        for r in results
        if r["primary_value_to_do"] == value or r["primary_value_not_to_do"] == value
    ]


def _split_nudge_for_value(
    results: list[dict], value: str
) -> tuple[list[dict], list[dict]]:
    """Split nudge results into toward-value and toward-non-value lists.

    dir_a always favours the to_do-side's primary value.
    Returns (toward_value_results, toward_non_value_results) where each
    entry is a single result dict per dilemma.
    """
    toward_val: dict[int, dict] = {}
    toward_nval: dict[int, dict] = {}

    for r in results:
        pv_to = r["primary_value_to_do"]
        pv_not = r["primary_value_not_to_do"]
        if pv_to != value and pv_not != value:
            continue

        value_is_to_do = pv_to == value
        direction = r.get("direction")

        if value_is_to_do:
            if direction == "dir_a":
                toward_val[r["dilemma_id"]] = r
            elif direction == "dir_b":
                toward_nval[r["dilemma_id"]] = r
        else:
            if direction == "dir_b":
                toward_val[r["dilemma_id"]] = r
            elif direction == "dir_a":
                toward_nval[r["dilemma_id"]] = r

    return list(toward_val.values()), list(toward_nval.values())


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert_model_value(
    input_dir: Path,
    output_dir: Path,
    model: str,
    value: str,
) -> int:
    """Convert all conditions for one model+value. Returns number of graphs written."""
    baseline_data = _load_condition(input_dir, model, "baseline")
    if baseline_data is None:
        return 0

    config = baseline_data.get("config", {})
    baseline_results = _filter_results_for_value(baseline_data["results"], value)
    if not baseline_results:
        return 0

    n_written = 0

    # Discover nudge conditions
    model_dir = input_dir / model
    nudge_types = [
        d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "baseline"
    ]

    for nudge_type in sorted(nudge_types):
        nudge_data = _load_condition(input_dir, model, nudge_type)
        if nudge_data is None:
            continue

        nudge_config = nudge_data.get("config", {})

        toward_val, toward_nval = _split_nudge_for_value(nudge_data["results"], value)
        if not toward_val and not toward_nval:
            continue

        base_dir = (
            output_dir / f"simple_{value}" / model / nudge_type / f"{TIMESTAMP}_base"
        )
        base_dir.mkdir(parents=True, exist_ok=True)
        base_graph = _build_graph_for_value(value, baseline_results, config, model)
        _write_graph(base_graph, base_dir, model)
        n_written += 1

        # Nudge toward value
        if toward_val:
            val_dir = (
                output_dir
                / f"simple_{value}"
                / model
                / nudge_type
                / f"{TIMESTAMP}_{value}"
            )
            val_dir.mkdir(parents=True, exist_ok=True)
            val_graph = _build_graph_for_value(
                value,
                toward_val,
                nudge_config,
                model,
                nudge_type=nudge_type,
                target_group=value,
            )
            _write_graph(val_graph, val_dir, model)
            n_written += 1

        # Combine all toward-non-value results into a single condition
        if toward_nval:
            nval_label = f"non_{value}"
            nval_dir = (
                output_dir
                / f"simple_{value}"
                / model
                / nudge_type
                / f"{TIMESTAMP}_{nval_label}"
            )
            nval_dir.mkdir(parents=True, exist_ok=True)
            nval_graph = _build_graph_for_value(
                value,
                toward_nval,
                nudge_config,
                model,
                nudge_type=nudge_type,
                target_group=nval_label,
            )
            _write_graph(nval_graph, nval_dir, model)
            n_written += 1

    return n_written


def _write_graph(graph: dict, directory: Path, model: str) -> None:
    with open(directory / f"preference_graph_{model}.json", "w") as f:
        json.dump(graph, f, indent=2)

    reasoning_mode = graph.get("simple_experiment_config", {}).get(
        "reasoning_mode", "none"
    )
    utility = {
        "utility_model_arguments": {
            "reasoning_mode": reasoning_mode,
        }
    }
    with open(directory / f"utility_model_{model}.json", "w") as f:
        json.dump(utility, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert results_global/ to simple_* format for create_summary.py",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(EXPERIMENT_DIR / "results_global"),
        help="Input results directory (default: results_global/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(EXPERIMENT_DIR / "results_simple"),
        help="Output directory (default: results_simple/)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to convert (default: all)",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help="Values to convert (default: all discovered values)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)

    models = args.models or _discover_models(input_dir)

    # Discover all values across models
    all_values: set[str] = set()
    for model in models:
        all_values |= _discover_values(input_dir, model)

    if args.values:
        value_list = sorted(
            v.strip().lower() for v in args.values if v.strip().lower() in all_values
        )
    else:
        value_list = sorted(all_values)

    if not value_list:
        print("No values found.")
        sys.exit(1)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Models: {models}")
    print(
        f"Values: {len(value_list)} ({', '.join(value_list[:10])}{'...' if len(value_list) > 10 else ''})"
    )
    print()

    total = 0
    for model in models:
        for value in value_list:
            n = convert_model_value(input_dir, output_dir, model, value)
            if n > 0:
                print(f"  {model} / {value}: {n} graphs")
                total += n

    print(f"\nWrote {total} preference graphs to {output_dir}")


if __name__ == "__main__":
    main()
