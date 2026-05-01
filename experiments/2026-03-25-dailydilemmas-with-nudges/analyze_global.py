#!/usr/bin/env python3
"""
Analyze global DailyDilemmas nudge experiment results.

Reads the results produced by run_global.py and computes per-value, per-model,
and per-nudge-type metrics.  Supports filtering and aggregation by reasoning
condition (none, before, native) so that instructed-reasoning and
native-reasoning models can be compared side-by-side.

Flip rates measure how often the majority choice (across k runs) changes:
  - flip_toward: flip rate when the nudge opposes the baseline majority
                 (e.g. influence toward value when baseline chose against it,
                 and the choice flips to the value)
  - flip_away:   flip rate when the nudge reinforces the baseline majority
                 (e.g. influence toward the already-chosen option, but the
                 choice paradoxically flips away from it)
  - flip_all:    combined across both nudge directions

Baseline reliability mode (--baseline-reliability) compares two independent
baseline runs (baseline vs baseline_2) to measure test-retest flip rates,
i.e. how often the model's majority choice changes with no nudge at all.

Usage:
    # Single-value analysis for all models
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --value honesty

    # Cross-value overview (default)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py

    # Filter by model, nudge type, value
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --models gpt-5-2-non-reasoning llama-33-70b \
        --values honesty safety \
        --nudge-types user_preference emotional

    # Filter by reasoning condition
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --reasoning none before

    # Output CSV
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --output overview.csv

    # Sort by column
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --sort abs_steerability --reverse

    # Baseline reliability (test-retest)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/analyze_global.py \
        --baseline-reliability
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from choices.analysis.metrics import compute_steerability_asym_from_counts
from choices.analysis.utils import (
    binomial_test_vs_half,
    get_base_model_name,
    get_reasoning_condition,
    two_proportion_z_test,
)

EXPERIMENT_DIR = Path(__file__).parent
MIN_RESPONSES = 1
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Results directory helpers
# ---------------------------------------------------------------------------


def results_dir() -> Path:
    return EXPERIMENT_DIR / "results_global"


def _load_condition(model: str, condition: str) -> dict | None:
    path = results_dir() / model / condition / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _discover_models() -> list[str]:
    rdir = results_dir()
    if not rdir.exists():
        return []
    return sorted(d.name for d in rdir.iterdir() if d.is_dir())


def _discover_values(model: str) -> set[str]:
    baseline = _load_condition(model, "baseline")
    if baseline is None:
        return set()
    values: set[str] = set()
    for r in baseline["results"]:
        values.add(r["primary_value_to_do"])
        values.add(r["primary_value_not_to_do"])
    return values


def _get_reasoning_condition(model_dir_name: str) -> str:
    """Determine the reasoning condition for a model results directory.

    Reads the stored model key and reasoning_mode from the baseline results
    and delegates to ``get_reasoning_condition`` which knows about reasoning
    models (low/medium/high/off) and instructed reasoning (before/after/none).
    """
    baseline = _load_condition(model_dir_name, "baseline")
    if baseline is None:
        return "unknown"

    stored_model = baseline.get("model", model_dir_name)
    result_dir = results_dir() / model_dir_name / "baseline"
    cond = get_reasoning_condition(stored_model, result_dir)

    # For chat models where no result_dir reasoning_mode was found, fall back
    # to the config block stored in the results JSON.
    if cond == "unknown":
        rm = baseline.get("config", {}).get("reasoning_mode")
        if rm is not None:
            return rm
    return cond


# ---------------------------------------------------------------------------
# Per-value analysis (extracted from run_global.py)
# ---------------------------------------------------------------------------


def _result_counts_for_value(result: dict, value: str) -> tuple[int, int, bool] | None:
    n_to_do = result["n_to_do"]
    n_not_to_do = result["n_not_to_do"]
    total = n_to_do + n_not_to_do
    if total < MIN_RESPONSES:
        return None
    pv_to_do = result["primary_value_to_do"]
    pv_not_to_do = result["primary_value_not_to_do"]
    if pv_to_do == value:
        return n_to_do, n_not_to_do, True
    if pv_not_to_do == value:
        return n_not_to_do, n_to_do, False
    return None


def _filter_results_by_value(
    results: list[dict],
    value: str,
    allowed_ids: set[int] | None = None,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in results:
        if allowed_ids is not None and r["dilemma_id"] not in allowed_ids:
            continue
        pv_to_do = r["primary_value_to_do"]
        pv_not_to_do = r["primary_value_not_to_do"]
        if value in (pv_to_do, pv_not_to_do):
            out[r["dilemma_id"]] = r
    return out


def _split_results_by_direction(
    results: list[dict],
    value: str,
    allowed_ids: set[int] | None = None,
) -> dict[str, dict[int, dict]]:
    by_dir: dict[str, dict[int, dict]] = {"dir_a": {}, "dir_b": {}}
    for r in results:
        if allowed_ids is not None and r["dilemma_id"] not in allowed_ids:
            continue
        direction = r.get("direction")
        if direction not in ("dir_a", "dir_b"):
            continue
        pv_to_do = r["primary_value_to_do"]
        pv_not_to_do = r["primary_value_not_to_do"]
        if value in (pv_to_do, pv_not_to_do):
            by_dir[direction][r["dilemma_id"]] = r
    return by_dir


def analyze_model(
    model: str,
    value: str,
    allowed_ids: set[int] | None = None,
) -> dict | None:
    """Compute all metrics for one model by slicing global results for *value*."""
    baseline_data = _load_condition(model, "baseline")
    if baseline_data is None:
        return None

    baseline_by_id = _filter_results_by_value(
        baseline_data["results"], value, allowed_ids
    )
    if not baseline_by_id:
        return None

    model_dir = results_dir() / model
    if not model_dir.exists():
        return None

    condition_dirs = [
        d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "baseline"
    ]

    influence_conditions: dict[str, dict[str, dict[int, dict]]] = {}
    for cond_name in condition_dirs:
        cond_data = _load_condition(model, cond_name)
        if cond_data is None:
            continue
        inf_type = cond_data["config"].get("influence_type")
        if not inf_type:
            continue
        by_dir = _split_results_by_direction(cond_data["results"], value, allowed_ids)
        if by_dir["dir_a"] or by_dir["dir_b"]:
            influence_conditions[inf_type] = by_dir

    if not influence_conditions:
        return None

    total_value = total_non_value = 0
    for r in baseline_by_id.values():
        counts = _result_counts_for_value(r, value)
        if counts is None:
            continue
        total_value += counts[0]
        total_non_value += counts[1]

    baseline_p_value = (
        total_value / (total_value + total_non_value)
        if (total_value + total_non_value) > 0
        else 0.5
    )
    baseline_test = binomial_test_vs_half(
        total_value,
        total_value + total_non_value,
        ALPHA,
    )

    metrics_by_type: dict[str, dict] = {}

    agg_c_0_A = agg_c_0_B = 0
    agg_c_A_A = agg_c_A_B = 0
    agg_c_B_A = agg_c_B_B = 0

    for inf_type, directions in influence_conditions.items():
        dir_a_by_id = directions.get("dir_a", {})
        dir_b_by_id = directions.get("dir_b", {})

        c_0_A = c_0_B = 0
        c_A_A = c_A_B = 0
        c_B_A = c_B_B = 0

        flip_toward_n = flip_toward_total = 0
        flip_away_n = flip_away_total = 0

        all_ids = set(dir_a_by_id) | set(dir_b_by_id)

        for did in all_ids:
            base_r = baseline_by_id.get(did)
            if base_r is None:
                continue
            base_counts = _result_counts_for_value(base_r, value)
            if base_counts is None:
                continue
            n_val_base, n_nval_base, value_is_to_do = base_counts

            c_0_A += n_val_base
            c_0_B += n_nval_base

            if n_val_base > n_nval_base:
                base_majority = "value"
            elif n_nval_base > n_val_base:
                base_majority = "non_value"
            else:
                base_majority = None

            if value_is_to_do:
                toward_value_id = "dir_a"
                toward_non_value_id = "dir_b"
            else:
                toward_value_id = "dir_b"
                toward_non_value_id = "dir_a"

            tv_by_id = directions.get(toward_value_id, {})
            tnv_by_id = directions.get(toward_non_value_id, {})

            if did in tv_by_id:
                tv_counts = _result_counts_for_value(tv_by_id[did], value)
                if tv_counts is not None:
                    c_A_A += tv_counts[0]
                    c_A_B += tv_counts[1]
                    if base_majority is not None:
                        nudge_maj = (
                            "value"
                            if tv_counts[0] > tv_counts[1]
                            else "non_value"
                            if tv_counts[1] > tv_counts[0]
                            else None
                        )
                        if nudge_maj is not None:
                            is_flip = base_majority != nudge_maj
                            if base_majority == "value":
                                flip_away_total += 1
                                flip_away_n += int(is_flip)
                            else:
                                flip_toward_total += 1
                                flip_toward_n += int(is_flip)

            if did in tnv_by_id:
                tnv_counts = _result_counts_for_value(tnv_by_id[did], value)
                if tnv_counts is not None:
                    c_B_A += tnv_counts[0]
                    c_B_B += tnv_counts[1]
                    if base_majority is not None:
                        nudge_maj = (
                            "value"
                            if tnv_counts[0] > tnv_counts[1]
                            else "non_value"
                            if tnv_counts[1] > tnv_counts[0]
                            else None
                        )
                        if nudge_maj is not None:
                            is_flip = base_majority != nudge_maj
                            if base_majority == "non_value":
                                flip_away_total += 1
                                flip_away_n += int(is_flip)
                            else:
                                flip_toward_total += 1
                                flip_toward_n += int(is_flip)

        s_A, s_B, asym, norm_asym = compute_steerability_asym_from_counts(
            c_0_A,
            c_0_B,
            c_A_A,
            c_A_B,
            c_B_A,
            c_B_B,
        )

        n_0 = c_0_A + c_0_B
        n_A = c_A_A + c_A_B
        n_B = c_B_A + c_B_B
        f_0_val = c_0_A / n_0 if n_0 > 0 else None
        f_0_nval = c_0_B / n_0 if n_0 > 0 else None
        p_val_toward_val = c_A_A / n_A if n_A > 0 else None
        p_val_toward_non = c_B_A / n_B if n_B > 0 else None
        f_nval_nval = c_B_B / n_B if n_B > 0 else None

        # Significance on pooled counts (matching create_summary.py):
        # sig_A: did nudging toward value change P(value)?
        # sig_B: did nudging toward non-value change P(non-value)?
        sig_A = False
        sig_B = False
        if f_0_val is not None and p_val_toward_val is not None and n_0 > 0 and n_A > 0:
            sig_A = two_proportion_z_test(f_0_val, n_0, p_val_toward_val, n_A, ALPHA)[
                "is_significant"
            ]
        if f_0_nval is not None and f_nval_nval is not None and n_0 > 0 and n_B > 0:
            sig_B = two_proportion_z_test(f_0_nval, n_0, f_nval_nval, n_B, ALPHA)[
                "is_significant"
            ]

        n_nudge_dirs = 2
        sig_count = int(sig_A) + int(sig_B)
        sig_rate = sig_count / n_nudge_dirs

        flip_all_total = flip_toward_total + flip_away_total
        flip_all_n = flip_toward_n + flip_away_n
        flip_rate_toward = (
            flip_toward_n / flip_toward_total if flip_toward_total > 0 else 0.0
        )
        flip_rate_away = flip_away_n / flip_away_total if flip_away_total > 0 else 0.0
        flip_rate_all = flip_all_n / flip_all_total if flip_all_total > 0 else 0.0

        abs_effect = None
        if all(
            v is not None for v in [f_0_val, p_val_toward_val, f_0_nval, f_nval_nval]
        ):
            abs_effect = (
                abs(p_val_toward_val - f_0_val) + abs(f_nval_nval - f_0_nval)
            ) / 2

        avg_steer = (s_A + s_B) / 2 if s_A is not None and s_B is not None else None
        abs_steer = (
            (abs(s_A) + abs(s_B)) / 2 if s_A is not None and s_B is not None else None
        )
        base_bias = max(f_0_val, 1 - f_0_val) if f_0_val is not None else None

        metrics_by_type[inf_type] = {
            "f_0_val": f_0_val,
            "p_val_toward_value": p_val_toward_val,
            "p_val_toward_non_value": p_val_toward_non,
            "abs_effect": abs_effect,
            "steerability_value": s_A,
            "steerability_non_value": s_B,
            "avg_steerability": avg_steer,
            "abs_steerability": abs_steer,
            "asymmetry": asym,
            "normalized_asymmetry": norm_asym,
            "base_bias": base_bias,
            "sig_rate": sig_rate,
            "sig_A": sig_A,
            "sig_B": sig_B,
            "flip_rate_toward": flip_rate_toward,
            "flip_rate_away": flip_rate_away,
            "flip_rate_all": flip_rate_all,
            "n_flip_toward": flip_toward_n,
            "n_flip_toward_total": flip_toward_total,
            "n_flip_away": flip_away_n,
            "n_flip_away_total": flip_away_total,
            "counts": {
                "c_0_A": c_0_A,
                "c_0_B": c_0_B,
                "c_A_A": c_A_A,
                "c_A_B": c_A_B,
                "c_B_A": c_B_A,
                "c_B_B": c_B_B,
            },
        }

        agg_c_0_A += c_0_A
        agg_c_0_B += c_0_B
        agg_c_A_A += c_A_A
        agg_c_A_B += c_A_B
        agg_c_B_A += c_B_A
        agg_c_B_B += c_B_B

    s_A, s_B, asym, norm_asym = compute_steerability_asym_from_counts(
        agg_c_0_A,
        agg_c_0_B,
        agg_c_A_A,
        agg_c_A_B,
        agg_c_B_A,
        agg_c_B_B,
    )

    agg_n_0 = agg_c_0_A + agg_c_0_B
    agg_n_A = agg_c_A_A + agg_c_A_B
    agg_n_B = agg_c_B_A + agg_c_B_B
    agg_p_val_toward_val = agg_c_A_A / agg_n_A if agg_n_A > 0 else None
    agg_p_val_toward_non = agg_c_B_A / agg_n_B if agg_n_B > 0 else None
    agg_f_0_nval = agg_c_0_B / agg_n_0 if agg_n_0 > 0 else None
    agg_f_nval_nval = agg_c_B_B / agg_n_B if agg_n_B > 0 else None

    # Aggregate sig/flip across influence types
    agg_sig_count = sum(
        int(m["sig_A"]) + int(m["sig_B"]) for m in metrics_by_type.values()
    )
    agg_n_nudge_dirs = 2 * len(metrics_by_type)
    agg_sig_rate = agg_sig_count / agg_n_nudge_dirs if agg_n_nudge_dirs > 0 else 0.0

    agg_flip_toward_n = sum(m["n_flip_toward"] for m in metrics_by_type.values())
    agg_flip_toward_total = sum(
        m["n_flip_toward_total"] for m in metrics_by_type.values()
    )
    agg_flip_away_n = sum(m["n_flip_away"] for m in metrics_by_type.values())
    agg_flip_away_total = sum(m["n_flip_away_total"] for m in metrics_by_type.values())
    agg_flip_all_n = agg_flip_toward_n + agg_flip_away_n
    agg_flip_all_total = agg_flip_toward_total + agg_flip_away_total

    agg_abs_effect = None
    if all(
        v is not None
        for v in [baseline_p_value, agg_p_val_toward_val, agg_f_0_nval, agg_f_nval_nval]
    ):
        agg_abs_effect = (
            abs(agg_p_val_toward_val - baseline_p_value)
            + abs(agg_f_nval_nval - agg_f_0_nval)
        ) / 2

    agg_avg_steer = (s_A + s_B) / 2 if s_A is not None and s_B is not None else None
    agg_abs_steer = (
        (abs(s_A) + abs(s_B)) / 2 if s_A is not None and s_B is not None else None
    )
    agg_base_bias = (
        max(baseline_p_value, 1 - baseline_p_value)
        if baseline_p_value is not None
        else None
    )

    return {
        "model": model,
        "selected_value": value,
        "n_dilemmas": len(baseline_by_id),
        "overall": {
            "baseline_p_value_side": baseline_p_value,
            "baseline_sig": bool(baseline_test["is_significant"]),
            "f_0_val": baseline_p_value,
            "p_val_toward_value": agg_p_val_toward_val,
            "p_val_toward_non_value": agg_p_val_toward_non,
            "abs_effect": agg_abs_effect,
            "steerability_value": s_A,
            "steerability_non_value": s_B,
            "avg_steerability": agg_avg_steer,
            "abs_steerability": agg_abs_steer,
            "asymmetry": asym,
            "normalized_asymmetry": norm_asym,
            "base_bias": agg_base_bias,
            "sig_rate": agg_sig_rate,
            "n_sig_nudges": agg_sig_count,
            "n_nudge_dirs": agg_n_nudge_dirs,
            "flip_rate_toward": agg_flip_toward_n / agg_flip_toward_total
            if agg_flip_toward_total > 0
            else 0.0,
            "flip_rate_away": agg_flip_away_n / agg_flip_away_total
            if agg_flip_away_total > 0
            else 0.0,
            "flip_rate_all": agg_flip_all_n / agg_flip_all_total
            if agg_flip_all_total > 0
            else 0.0,
            "n_flip_toward": agg_flip_toward_n,
            "n_flip_toward_total": agg_flip_toward_total,
            "n_flip_away": agg_flip_away_n,
            "n_flip_away_total": agg_flip_away_total,
        },
        "by_influence_type": metrics_by_type,
    }


# ---------------------------------------------------------------------------
# Overview row construction
# ---------------------------------------------------------------------------


def _build_overview_rows(
    all_metrics: list[dict],
    nudge_types: list[str] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for m in all_metrics:
        reasoning = _get_reasoning_condition(m["model"])
        base_model = get_base_model_name(m["model"])
        for inf_type, it in sorted(m["by_influence_type"].items()):
            if nudge_types and inf_type not in nudge_types:
                continue
            rows.append(
                {
                    "value": m["selected_value"],
                    "model": m["model"],
                    "base_model": base_model,
                    "reasoning": reasoning,
                    "influence_type": inf_type,
                    "n_dilemmas": m["n_dilemmas"],
                    "baseline_p_value": m["overall"]["baseline_p_value_side"],
                    "base_bias": it["base_bias"],
                    "f_toward_val": it["p_val_toward_value"],
                    "f_toward_nval": it["p_val_toward_non_value"],
                    "abs_effect": it["abs_effect"],
                    "steerability_value": it["steerability_value"],
                    "steerability_non_value": it["steerability_non_value"],
                    "avg_steerability": it["avg_steerability"],
                    "abs_steerability": it["abs_steerability"],
                    "asymmetry": it["asymmetry"],
                    "normalized_asymmetry": it["normalized_asymmetry"],
                    "sig_rate": it["sig_rate"],
                    "sig_A": it["sig_A"],
                    "sig_B": it["sig_B"],
                    "flip_rate_toward": it["flip_rate_toward"],
                    "flip_rate_away": it["flip_rate_away"],
                    "flip_rate_all": it["flip_rate_all"],
                    "n_flip_toward": it["n_flip_toward"],
                    "n_flip_toward_total": it["n_flip_toward_total"],
                    "n_flip_away": it["n_flip_away"],
                    "n_flip_away_total": it["n_flip_away_total"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt(v, fmt=".3f"):
    return f"{v:{fmt}}" if v is not None else "  n/a"


def _fmt_ci(val, ci, fmt=".3f"):
    if val is None:
        return "n/a"
    s = f"{val:{fmt}}"
    if ci is not None and ci[0] is not None:
        s += f" ({ci[0]:{fmt}}, {ci[1]:{fmt}})"
    return s


def _safe_mean(dicts: list[dict], key: str):
    vals = [d[key] for d in dicts if d.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _ci(values: list[float]):
    n = len(values)
    if n == 0:
        return None, None, None
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    se = math.sqrt(variance) / math.sqrt(n)
    return mean, mean - 1.96 * se, mean + 1.96 * se


def _compute_aggregate(rows: list[dict]) -> dict | None:
    if not rows:
        return None

    total_nudge_dirs = 2 * len(rows)
    total_sig = sum(int(r["sig_A"]) + int(r["sig_B"]) for r in rows)

    tot_flip_toward_n = sum(r["n_flip_toward"] for r in rows)
    tot_flip_toward_total = sum(r["n_flip_toward_total"] for r in rows)
    tot_flip_away_n = sum(r["n_flip_away"] for r in rows)
    tot_flip_away_total = sum(r["n_flip_away_total"] for r in rows)
    tot_flip_all_n = tot_flip_toward_n + tot_flip_away_n
    tot_flip_all_total = tot_flip_toward_total + tot_flip_away_total

    abs_eff_m, abs_eff_lo, abs_eff_hi = _ci(
        [r["abs_effect"] for r in rows if r["abs_effect"] is not None]
    )
    avg_s_m, avg_s_lo, avg_s_hi = _ci(
        [r["avg_steerability"] for r in rows if r["avg_steerability"] is not None]
    )
    abs_s_m, abs_s_lo, abs_s_hi = _ci(
        [r["abs_steerability"] for r in rows if r["abs_steerability"] is not None]
    )
    abs_a_m, abs_a_lo, abs_a_hi = _ci(
        [abs(r["asymmetry"]) for r in rows if r["asymmetry"] is not None]
    )
    abs_na_m, abs_na_lo, abs_na_hi = _ci(
        [
            abs(r["normalized_asymmetry"])
            for r in rows
            if r["normalized_asymmetry"] is not None
        ]
    )
    bias_m, bias_lo, bias_hi = _ci(
        [r["base_bias"] for r in rows if r["base_bias"] is not None]
    )

    def _pack_ci(m, lo, hi):
        return (lo, hi) if m is not None else None

    return {
        "n": len(rows),
        "baseline_p_value": _safe_mean(rows, "baseline_p_value"),
        "base_bias": bias_m,
        "base_bias_ci": _pack_ci(bias_m, bias_lo, bias_hi),
        "abs_effect": abs_eff_m,
        "abs_effect_ci": _pack_ci(abs_eff_m, abs_eff_lo, abs_eff_hi),
        "avg_steerability": avg_s_m,
        "avg_steerability_ci": _pack_ci(avg_s_m, avg_s_lo, avg_s_hi),
        "abs_steerability": abs_s_m,
        "abs_steerability_ci": _pack_ci(abs_s_m, abs_s_lo, abs_s_hi),
        "abs_asymmetry": abs_a_m,
        "abs_asymmetry_ci": _pack_ci(abs_a_m, abs_a_lo, abs_a_hi),
        "abs_norm_asymmetry": abs_na_m,
        "abs_norm_asymmetry_ci": _pack_ci(abs_na_m, abs_na_lo, abs_na_hi),
        "sig_rate": total_sig / total_nudge_dirs if total_nudge_dirs > 0 else 0.0,
        "flip_rate_toward": tot_flip_toward_n / tot_flip_toward_total
        if tot_flip_toward_total > 0
        else 0.0,
        "flip_rate_away": tot_flip_away_n / tot_flip_away_total
        if tot_flip_away_total > 0
        else 0.0,
        "flip_rate_all": tot_flip_all_n / tot_flip_all_total
        if tot_flip_all_total > 0
        else 0.0,
        "n_nudge_dirs": total_nudge_dirs,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _print_aggregate_line(label: str, agg: dict) -> None:
    parts = [f"n={agg['n']}"]
    if agg.get("baseline_p_value") is not None:
        parts.append(f"P(val)={agg['baseline_p_value']:.3f}")
    parts.append(f"base_bias={_fmt_ci(agg['base_bias'], agg.get('base_bias_ci'))}")
    parts.append(f"|effect|={_fmt_ci(agg['abs_effect'], agg.get('abs_effect_ci'))}")
    parts.append(
        f"|steer|={_fmt_ci(agg['abs_steerability'], agg.get('abs_steerability_ci'))}"
    )
    parts.append(
        f"avg_steer={_fmt_ci(agg['avg_steerability'], agg.get('avg_steerability_ci'))}"
    )
    parts.append(f"|asym|={_fmt_ci(agg['abs_asymmetry'], agg.get('abs_asymmetry_ci'))}")
    parts.append(
        f"|n-asym|={_fmt_ci(agg['abs_norm_asymmetry'], agg.get('abs_norm_asymmetry_ci'))}"
    )
    parts.append(f"sig={agg['sig_rate']:.1%}")
    parts.append(f"flip_toward={agg['flip_rate_toward']:.1%}")
    parts.append(f"flip_away={agg['flip_rate_away']:.1%}")
    parts.append(f"flip_all={agg['flip_rate_all']:.1%}")
    print(f"  {label}: {', '.join(parts)}")


def _print_overview_table(
    rows: list[dict], sort_key: str | None, reverse: bool
) -> None:
    sorted_rows = _sort_rows(rows, sort_key, reverse)

    header = (
        f"{'Value':<14} {'Model':<34} {'Reason':<8} {'Nudge':<16} "
        f"{'n':>4} {'P(val)':>6} {'Bias':>5} "
        f"{'|Eff|':>6} {'s(v)':>7} {'s(~v)':>7} {'|s|':>7} "
        f"{'Asym':>7} {'N-Asy':>7} "
        f"{'Sig%':>5} {'→Fl':>5} {'←Fl':>5} {'Fl':>5}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for r in sorted_rows:
        print(
            f"{r['value']:<14} "
            f"{r['model']:<34} "
            f"{r['reasoning']:<8} "
            f"{r['influence_type']:<16} "
            f"{r['n_dilemmas']:>4d} "
            f"{r['baseline_p_value']:>6.3f} "
            f"{_fmt(r['base_bias'], '.3f'):>5} "
            f"{_fmt(r['abs_effect'], '.3f'):>6} "
            f"{_fmt(r['steerability_value'], '.3f'):>7} "
            f"{_fmt(r['steerability_non_value'], '.3f'):>7} "
            f"{_fmt(r['abs_steerability'], '.3f'):>7} "
            f"{_fmt(r['asymmetry'], '.3f'):>7} "
            f"{_fmt(r['normalized_asymmetry'], '.3f'):>7} "
            f"{r['sig_rate']:>5.1%} "
            f"{r['flip_rate_toward']:>5.1%} "
            f"{r['flip_rate_away']:>5.1%} "
            f"{r['flip_rate_all']:>5.1%}"
        )


def _print_single_value_table(all_metrics: list[dict]) -> None:
    """Print the per-model table for a single --value analysis."""
    value = all_metrics[0]["selected_value"] if all_metrics else "?"

    print(f"\n{'='*96}")
    print(f'GLOBAL DAILYDILEMMAS RESULTS  —  value = "{value}"')
    print(f"{'='*96}")

    header = (
        f"{'Model':<32} {'Rsn':<6} {'P(val)':<8} {'P→val':<8} {'P→~val':<8} "
        f"{'s(val)':>8} {'s(~val)':>8} {'Asym':>8} {'N-Asym':>8} "
        f"{'Sig%':>6} {'→Fl%':>6} {'←Fl%':>6} {'Fl%':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for m in all_metrics:
        o = m["overall"]
        reasoning = _get_reasoning_condition(m["model"])
        print(
            f"{m['model']:<32} "
            f"{reasoning:<6} "
            f"{o['baseline_p_value_side']:<8.3f} "
            f"{_fmt(o['p_val_toward_value']):<8} "
            f"{_fmt(o['p_val_toward_non_value']):<8} "
            f"{_fmt(o['steerability_value']):>8} "
            f"{_fmt(o['steerability_non_value']):>8} "
            f"{_fmt(o['asymmetry']):>8} "
            f"{_fmt(o['normalized_asymmetry']):>8} "
            f"{o['sig_rate']:>5.1%} "
            f"{o['flip_rate_toward']:>5.1%} "
            f"{o['flip_rate_away']:>5.1%} "
            f"{o['flip_rate_all']:>5.1%}"
        )

    if len(all_metrics) > 1:
        overalls = [m["overall"] for m in all_metrics]
        print("-" * len(header))
        print(
            f"{'  across models':<32} "
            f"{'':6} "
            f"{_safe_mean(overalls, 'baseline_p_value_side') or 0:<8.3f} "
            f"{_fmt(_safe_mean(overalls, 'p_val_toward_value')):<8} "
            f"{_fmt(_safe_mean(overalls, 'p_val_toward_non_value')):<8} "
            f"{_fmt(_safe_mean(overalls, 'steerability_value')):>8} "
            f"{_fmt(_safe_mean(overalls, 'steerability_non_value')):>8} "
            f"{_fmt(_safe_mean(overalls, 'asymmetry')):>8} "
            f"{_fmt(_safe_mean(overalls, 'normalized_asymmetry')):>8} "
            f"{_safe_mean(overalls, 'sig_rate') or 0:>5.1%} "
            f"{_safe_mean(overalls, 'flip_rate_toward') or 0:>5.1%} "
            f"{_safe_mean(overalls, 'flip_rate_away') or 0:>5.1%} "
            f"{_safe_mean(overalls, 'flip_rate_all') or 0:>5.1%}"
        )

    inf_types: set[str] = set()
    for m in all_metrics:
        inf_types.update(m["by_influence_type"])

    for inf_type in sorted(inf_types):
        print(f"\n--- {inf_type} ---")
        sub_header = (
            f"{'Model':<32} {'Rsn':<6} {'P→val':<8} {'P→~val':<8} "
            f"{'s(val)':>8} {'s(~val)':>8} {'Asym':>8} "
            f"{'Sig%':>6} {'→Fl%':>6} {'←Fl%':>6} {'Fl%':>6}"
        )
        print(sub_header)
        print("-" * len(sub_header))
        type_rows: list[dict] = []
        for m in all_metrics:
            it = m["by_influence_type"].get(inf_type)
            if it is None:
                continue
            reasoning = _get_reasoning_condition(m["model"])
            print(
                f"{m['model']:<32} "
                f"{reasoning:<6} "
                f"{_fmt(it['p_val_toward_value']):<8} "
                f"{_fmt(it['p_val_toward_non_value']):<8} "
                f"{_fmt(it['steerability_value']):>8} "
                f"{_fmt(it['steerability_non_value']):>8} "
                f"{_fmt(it['asymmetry']):>8} "
                f"{it['sig_rate']:>5.1%} "
                f"{it['flip_rate_toward']:>5.1%} "
                f"{it['flip_rate_away']:>5.1%} "
                f"{it['flip_rate_all']:>5.1%}"
            )
            type_rows.append(it)
        if len(type_rows) > 1:
            print("-" * len(sub_header))
            print(
                f"{'  across models':<32} "
                f"{'':6} "
                f"{_fmt(_safe_mean(type_rows, 'p_val_toward_value')):<8} "
                f"{_fmt(_safe_mean(type_rows, 'p_val_toward_non_value')):<8} "
                f"{_fmt(_safe_mean(type_rows, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean(type_rows, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean(type_rows, 'asymmetry')):>8} "
                f"{_safe_mean(type_rows, 'sig_rate') or 0:>5.1%} "
                f"{_safe_mean(type_rows, 'flip_rate_toward') or 0:>5.1%} "
                f"{_safe_mean(type_rows, 'flip_rate_away') or 0:>5.1%} "
                f"{_safe_mean(type_rows, 'flip_rate_all') or 0:>5.1%}"
            )


def _print_aggregate_stats(rows: list[dict]) -> None:
    """Print aggregate statistics grouped by various dimensions."""

    # --- By value ---
    values = sorted({r["value"] for r in rows})
    print(f"\nBy Value ({len(values)}):")
    for value in values:
        agg = _compute_aggregate([r for r in rows if r["value"] == value])
        if agg:
            _print_aggregate_line(value, agg)

    # --- By influence type ---
    inf_types = sorted({r["influence_type"] for r in rows})
    print(f"\nBy Influence Type ({len(inf_types)}):")
    for inf_type in inf_types:
        agg = _compute_aggregate([r for r in rows if r["influence_type"] == inf_type])
        if agg:
            _print_aggregate_line(inf_type, agg)

    # --- By model ---
    models = sorted({r["model"] for r in rows})
    print(f"\nBy Model ({len(models)}):")
    for model in models:
        agg = _compute_aggregate([r for r in rows if r["model"] == model])
        if agg:
            _print_aggregate_line(model, agg)

    # --- By reasoning condition ---
    reasoning_conditions = sorted({r["reasoning"] for r in rows})
    print(f"\nBy Reasoning Condition ({len(reasoning_conditions)}):")
    for rc in reasoning_conditions:
        agg = _compute_aggregate([r for r in rows if r["reasoning"] == rc])
        if agg:
            _print_aggregate_line(rc, agg)

    # --- By model x reasoning condition ---
    model_reasoning_pairs = sorted({(r["base_model"], r["reasoning"]) for r in rows})
    if len(model_reasoning_pairs) > 1:
        print(f"\nBy Base Model x Reasoning ({len(model_reasoning_pairs)}):")
        for base_model, rc in model_reasoning_pairs:
            subset = [
                r
                for r in rows
                if r["base_model"] == base_model and r["reasoning"] == rc
            ]
            agg = _compute_aggregate(subset)
            if agg:
                _print_aggregate_line(f"{base_model} ({rc})", agg)

    # --- Overall ---
    agg = _compute_aggregate(rows)
    if agg:
        print("\nOverall:")
        _print_aggregate_line("all", agg)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

SORTABLE_COLUMNS = {
    "value",
    "model",
    "reasoning",
    "influence_type",
    "n_dilemmas",
    "baseline_p_value",
    "base_bias",
    "f_toward_val",
    "f_toward_nval",
    "abs_effect",
    "steerability_value",
    "steerability_non_value",
    "avg_steerability",
    "abs_steerability",
    "asymmetry",
    "normalized_asymmetry",
    "sig_rate",
    "flip_rate_toward",
    "flip_rate_away",
    "flip_rate_all",
}


def _sort_rows(rows: list[dict], sort_key: str | None, reverse: bool) -> list[dict]:
    if sort_key is None:
        return sorted(rows, key=lambda x: (x["value"], x["model"], x["influence_type"]))

    use_abs = sort_key.startswith("abs-")
    col = sort_key[4:] if use_abs else sort_key

    if col not in SORTABLE_COLUMNS:
        print(f"Warning: unknown sort column '{col}', using default order")
        return sorted(rows, key=lambda x: (x["value"], x["model"], x["influence_type"]))

    def _key(r):
        v = r.get(col)
        if v is None:
            return (1, 0)
        if use_abs and isinstance(v, (int, float)):
            return (0, abs(v))
        return (0, v)

    return sorted(rows, key=_key, reverse=reverse)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "value",
    "model",
    "base_model",
    "reasoning",
    "influence_type",
    "n_dilemmas",
    "baseline_p_value",
    "base_bias",
    "f_toward_val",
    "f_toward_nval",
    "abs_effect",
    "steerability_value",
    "steerability_non_value",
    "avg_steerability",
    "abs_steerability",
    "asymmetry",
    "normalized_asymmetry",
    "sig_rate",
    "sig_A",
    "sig_B",
    "flip_rate_toward",
    "flip_rate_away",
    "flip_rate_all",
    "n_flip_toward",
    "n_flip_toward_total",
    "n_flip_away",
    "n_flip_away_total",
]


def _write_csv(rows: list[dict], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(
            rows, key=lambda x: (x["value"], x["model"], x["influence_type"])
        ):
            out = {}
            for k in _CSV_FIELDS:
                v = r.get(k)
                out[k] = "" if v is None else v
            writer.writerow(out)
    print(f"Wrote {len(rows)} rows to {output_path}")


# ---------------------------------------------------------------------------
# Baseline reliability (test-retest flip rates)
# ---------------------------------------------------------------------------


def _majority_choice(n_val: int, n_nval: int) -> str | None:
    """Return 'value', 'non_value', or None if tied."""
    if n_val > n_nval:
        return "value"
    if n_nval > n_val:
        return "non_value"
    return None


def _compute_baseline_reliability(
    model: str,
    baseline_name: str = "baseline",
    baseline_2_name: str = "baseline_2",
) -> dict | None:
    """Compare two baseline runs and compute per-dilemma flip rates.

    Each dilemma is counted once for the totals. For the per-value breakdown,
    each dilemma appears under both its primary values (to_do and not_to_do).

    Flip rates are split into three categories:
      - flip_toward: dilemmas where baseline_1 majority was "value"
      - flip_away:   dilemmas where baseline_1 majority was "non_value"
      - flip_all:    all dilemmas combined
    """
    b1 = _load_condition(model, baseline_name)
    b2 = _load_condition(model, baseline_2_name)
    if b1 is None or b2 is None:
        return None

    b1_by_id: dict[int, dict] = {r["dilemma_id"]: r for r in b1["results"]}
    b2_by_id: dict[int, dict] = {r["dilemma_id"]: r for r in b2["results"]}

    common_ids = sorted(set(b1_by_id) & set(b2_by_id))
    if not common_ids:
        return None

    @dataclass
    class _FlipBucket:
        n: int = 0
        n_flips: int = 0

    per_value_toward: dict[str, _FlipBucket] = {}
    per_value_away: dict[str, _FlipBucket] = {}
    per_value_all: dict[str, _FlipBucket] = {}

    total_toward = _FlipBucket()
    total_away = _FlipBucket()
    total_all = _FlipBucket()
    total_skipped = 0

    for did in common_ids:
        r1 = b1_by_id[did]
        r2 = b2_by_id[did]
        pv_to_do = r1["primary_value_to_do"]
        pv_not_to_do = r1["primary_value_not_to_do"]

        counts1 = _result_counts_for_value(r1, pv_to_do)
        counts2 = _result_counts_for_value(r2, pv_to_do)
        if counts1 is None or counts2 is None:
            total_skipped += 1
            continue

        n_val_1, n_nval_1, _ = counts1
        n_val_2, n_nval_2, _ = counts2

        maj1 = _majority_choice(n_val_1, n_nval_1)
        maj2 = _majority_choice(n_val_2, n_nval_2)
        if maj1 is None or maj2 is None:
            total_skipped += 1
            continue

        is_flip = maj1 != maj2
        flip_int = int(is_flip)

        total_all.n += 1
        total_all.n_flips += flip_int

        for v in (pv_to_do, pv_not_to_do):
            pv_all = per_value_all.setdefault(v, _FlipBucket())
            pv_all.n += 1
            pv_all.n_flips += flip_int

        if maj1 == "value":
            total_toward.n += 1
            total_toward.n_flips += flip_int
            for v in (pv_to_do, pv_not_to_do):
                pv_tw = per_value_toward.setdefault(v, _FlipBucket())
                pv_tw.n += 1
                pv_tw.n_flips += flip_int
        else:
            total_away.n += 1
            total_away.n_flips += flip_int
            for v in (pv_to_do, pv_not_to_do):
                pv_aw = per_value_away.setdefault(v, _FlipBucket())
                pv_aw.n += 1
                pv_aw.n_flips += flip_int

    def _rate(bucket: _FlipBucket) -> float:
        return bucket.n_flips / bucket.n if bucket.n > 0 else 0.0

    all_values = sorted(set(per_value_all))

    return {
        "model": model,
        "n_dilemmas": len(common_ids),
        "n_compared": total_all.n,
        "n_skipped": total_skipped,
        "n_flips": total_all.n_flips,
        "flip_rate": _rate(total_all),
        "flip_toward_n": total_toward.n_flips,
        "flip_toward_total": total_toward.n,
        "flip_toward_rate": _rate(total_toward),
        "flip_away_n": total_away.n_flips,
        "flip_away_total": total_away.n,
        "flip_away_rate": _rate(total_away),
        "per_value": {
            v: {
                "n": per_value_all[v].n,
                "n_flips": per_value_all[v].n_flips,
                "flip_rate": _rate(per_value_all[v]),
                "flip_toward_n": per_value_toward.get(v, _FlipBucket()).n_flips,
                "flip_toward_total": per_value_toward.get(v, _FlipBucket()).n,
                "flip_toward_rate": _rate(per_value_toward.get(v, _FlipBucket())),
                "flip_away_n": per_value_away.get(v, _FlipBucket()).n_flips,
                "flip_away_total": per_value_away.get(v, _FlipBucket()).n,
                "flip_away_rate": _rate(per_value_away.get(v, _FlipBucket())),
            }
            for v in all_values
        },
    }


def _fmt_flip3(toward_n, toward_tot, away_n, away_tot, all_n, all_tot) -> str:
    """Format three flip-rate columns: →Fl  ←Fl  Fl"""
    tw = f"{toward_n / toward_tot:.1%}" if toward_tot else "  n/a"
    aw = f"{away_n / away_tot:.1%}" if away_tot else "  n/a"
    al = f"{all_n / all_tot:.1%}" if all_tot else "  n/a"
    return f"{tw:>6} {aw:>6} {al:>6}"


def _print_baseline_reliability(results: list[dict]) -> None:
    """Print baseline reliability (test-retest) results."""
    print(f"\n{'=' * 80}")
    print("BASELINE RELIABILITY (test-retest flip rates)")
    print(f"{'=' * 80}")

    header = f"{'Model':<40} {'Rsn':<6} {'n':>5}  " f"{'→Fl':>6} {'←Fl':>6} {'Fl':>6}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        reasoning = _get_reasoning_condition(r["model"])
        flip_cols = _fmt_flip3(
            r["flip_toward_n"],
            r["flip_toward_total"],
            r["flip_away_n"],
            r["flip_away_total"],
            r["n_flips"],
            r["n_compared"],
        )
        print(
            f"{r['model']:<40} "
            f"{reasoning:<6} "
            f"{r['n_compared']:>5}  "
            f"{flip_cols}"
        )

    if len(results) > 1:
        tot_compared = sum(r["n_compared"] for r in results)
        tot_tw_n = sum(r["flip_toward_n"] for r in results)
        tot_tw_t = sum(r["flip_toward_total"] for r in results)
        tot_aw_n = sum(r["flip_away_n"] for r in results)
        tot_aw_t = sum(r["flip_away_total"] for r in results)
        tot_flips = sum(r["n_flips"] for r in results)
        flip_cols = _fmt_flip3(
            tot_tw_n, tot_tw_t, tot_aw_n, tot_aw_t, tot_flips, tot_compared
        )
        print("-" * len(header))
        print(f"{'  all models':<40} {'':6} " f"{tot_compared:>5}  " f"{flip_cols}")

    all_values: set[str] = set()
    for r in results:
        all_values.update(r["per_value"])

    print("\nPer Value:")
    val_header = f"  {'Value':<20} {'n':>5}  " f"{'→Fl':>6} {'←Fl':>6} {'Fl':>6}"
    print(val_header)
    print("  " + "-" * (len(val_header) - 2))

    for v in sorted(all_values):
        n = tw_n = tw_t = aw_n = aw_t = flips = 0
        for r in results:
            vd = r["per_value"].get(v)
            if vd:
                n += vd["n"]
                flips += vd["n_flips"]
                tw_n += vd["flip_toward_n"]
                tw_t += vd["flip_toward_total"]
                aw_n += vd["flip_away_n"]
                aw_t += vd["flip_away_total"]
        flip_cols = _fmt_flip3(tw_n, tw_t, aw_n, aw_t, flips, n)
        print(f"  {v:<20} {n:>5}  {flip_cols}")

    if len(results) > 1:
        print("\nPer Model x Value:")
        for r in results:
            reasoning = _get_reasoning_condition(r["model"])
            base_model = get_base_model_name(r["model"])
            label = f"{base_model} ({reasoning})"
            print(f"\n  {label}:")
            for v in sorted(r["per_value"]):
                vd = r["per_value"][v]
                flip_cols = _fmt_flip3(
                    vd["flip_toward_n"],
                    vd["flip_toward_total"],
                    vd["flip_away_n"],
                    vd["flip_away_total"],
                    vd["n_flips"],
                    vd["n"],
                )
                print(f"    {v:<20} {vd['n']:>5}  {flip_cols}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _collect_all_metrics(
    models: list[str],
    value_list: list[str],
) -> list[dict]:
    all_metrics: list[dict] = []
    for value in value_list:
        for model in models:
            metrics = analyze_model(model, value)
            if metrics:
                all_metrics.append(metrics)
    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze global DailyDilemmas nudge experiment results",
    )

    # --- Filters ---
    parser.add_argument(
        "--value",
        type=str,
        help="Single value for detailed per-model table (e.g. 'honesty')",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        default=None,
        help="Values to include (default: all discovered values)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model directory names to include (default: all)",
    )
    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="Influence/nudge types to include (default: all)",
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=None,
        help="Reasoning conditions to include: 'none', 'before', 'native' "
        "(default: all)",
    )

    # --- Modes ---
    parser.add_argument(
        "--baseline-reliability",
        action="store_true",
        help="Compare baseline vs baseline_2 to compute test-retest flip rates",
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Write overview to CSV file instead of printing",
    )
    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default=None,
        help="Column to sort the overview table by. Prefix with 'abs-' to sort "
        "by absolute value. Valid columns: " + ", ".join(sorted(SORTABLE_COLUMNS)),
    )
    parser.add_argument(
        "--reverse",
        "-r",
        action="store_true",
        help="Sort in descending order (default: ascending)",
    )

    args = parser.parse_args()

    rdir = results_dir()
    if not rdir.exists():
        print(f"No results directory at {rdir}")
        sys.exit(1)

    # Discover models
    all_model_dirs = _discover_models()
    if args.models:
        models = [m for m in all_model_dirs if m in args.models]
    else:
        models = all_model_dirs

    if not models:
        print("No model directories found matching filters.")
        sys.exit(1)

    # Print filter summary
    print("=" * 80)
    print("DailyDilemmas Global Analysis")
    print("=" * 80)
    print(f"Results directory: {rdir}")
    print(f"Models: {models}")
    if args.values:
        print(f"Value filter: {args.values}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning:
        print(f"Reasoning condition filter: {args.reasoning}")
    if args.sort:
        sort_desc = f"Sort by: {args.sort}"
        if args.reverse:
            sort_desc += " (descending)"
        print(sort_desc)
    print("=" * 80)

    # --- Baseline reliability mode ---
    if args.baseline_reliability:
        reliability_results = []
        for model in models:
            r = _compute_baseline_reliability(model)
            if r is not None:
                reasoning = _get_reasoning_condition(model)
                if args.reasoning and reasoning not in args.reasoning:
                    continue
                reliability_results.append(r)
        if reliability_results:
            _print_baseline_reliability(reliability_results)
        else:
            print(
                "No models found with both baseline and baseline_2 results. "
                "Run with --baseline-2 first."
            )
        return

    # --- Single-value mode ---
    if args.value:
        selected_value = args.value.strip().lower()
        all_metrics = []
        for model in models:
            metrics = analyze_model(model, selected_value)
            if metrics:
                reasoning = _get_reasoning_condition(model)
                if args.reasoning and reasoning not in args.reasoning:
                    continue
                all_metrics.append(metrics)

        if all_metrics:
            _print_single_value_table(all_metrics)
            out_path = rdir / f"metrics_{selected_value.replace(' ', '_')}.json"
            with open(out_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"\nFull metrics saved to {out_path}")
        else:
            print(f"No results found for value '{selected_value}'.")
        return

    # --- Overview mode (default) ---
    available_values: set[str] = set()
    for model in models:
        available_values |= _discover_values(model)

    if args.values:
        value_list = sorted(
            v.strip().lower()
            for v in args.values
            if v.strip().lower() in available_values
        )
    else:
        value_list = sorted(available_values)

    if not value_list:
        print("No matching values found in results.")
        return

    all_metrics = _collect_all_metrics(models, value_list)
    if not all_metrics:
        print("No results found.")
        return

    rows = _build_overview_rows(all_metrics, args.nudge_types)

    # Apply reasoning condition filter
    if args.reasoning:
        rows = [r for r in rows if r["reasoning"] in args.reasoning]

    if not rows:
        print("No rows after filtering.")
        return

    print(f"\nFound {len(rows)} result rows\n")

    if args.output:
        _write_csv(rows, args.output)
    else:
        print(f"{'=' * 110}")
        print("CROSS-VALUE OVERVIEW (global)")
        print(f"{'=' * 110}")
        _print_overview_table(rows, args.sort, args.reverse)
        _print_aggregate_stats(rows)


if __name__ == "__main__":
    main()
