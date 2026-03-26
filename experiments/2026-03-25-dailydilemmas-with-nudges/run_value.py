#!/usr/bin/env python3
"""
Value-based DailyDilemmas runner.

Filters dilemmas by a selected moral value (appearing exclusively on one side),
runs baseline + nudge conditions, and computes main-experiment-style metrics
where the value acts as the binary factor (analogous to gender/age in the main
Choices experiments).

Level A = "value side"   (the option whose values_aggregated contains the value)
Level B = "non-value side" (the other option)

Metrics:
  - Baseline preference: P(choosing value-side option)
  - Steerability: log-odds shift when nudged toward each side
  - Asymmetry: s(B) - s(A) and normalized version
  - Backfiring rate: fraction of nudges that move probability against intended direction

Usage:
    # Run a single model + all conditions for a value
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_value.py \
        --value honesty --model llama-33-70b --all-conditions

    # Run everything for a value
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_value.py \
        --value honesty --all

    # Analyze only (no API calls)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_value.py \
        --value honesty --analyze-only

    # Dry run: print sample prompts
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_value.py \
        --value honesty --model llama-33-70b --all-conditions --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from choices.utils import (
    create_agent,
    generate_responses,
    parse_responses_forced_choice,
)
from choices import ReasoningMode
from choices.analysis.metrics import (
    compute_steerability_asym_from_counts,
)
from choices.analysis.utils import (
    binomial_test_vs_half,
    two_proportion_z_test,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_dilemmas, Dilemma
from influence_templates import (
    INFLUENCE_TYPES,
    generate_few_shot_action_examples,
    generate_few_shot_value_examples,
    render_influence_text,
)
from prompts import build_prompt, DilemmaPromptInfo


EXPERIMENT_DIR = Path(__file__).parent
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ValueDilemma:
    """A dilemma annotated with which side holds the selected value."""

    dilemma: Dilemma
    value_is_to_do: bool  # True when the selected value is on the to_do side


@dataclass
class DilemmaResult:
    dilemma_id: int
    to_do_is_a: bool
    value_is_to_do: bool
    responses: list[str]
    n_to_do: int
    n_not_to_do: int
    n_unparseable: int

    @property
    def n_value(self) -> int:
        return self.n_to_do if self.value_is_to_do else self.n_not_to_do

    @property
    def n_non_value(self) -> int:
        return self.n_not_to_do if self.value_is_to_do else self.n_to_do


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_dilemmas_by_value(
    dilemmas: list[Dilemma],
    value: str,
) -> list[ValueDilemma]:
    """Keep dilemmas where *value* appears in exactly one side (XOR)."""
    value_lower = value.strip().lower()
    result = []
    for d in dilemmas:
        in_to_do = value_lower in {v.strip().lower() for v in d.values_to_do}
        in_not_to_do = value_lower in {v.strip().lower() for v in d.values_not_to_do}
        if in_to_do == in_not_to_do:
            continue
        result.append(ValueDilemma(dilemma=d, value_is_to_do=in_to_do))
    return result


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def get_conditions(config: dict) -> list[dict]:
    """
    Build the condition list.

    Each condition dict has:
        name: str
        influence_type: str | None
        nudge_toward: str | None   — "value" or "non_value"
    """
    conditions: list[dict] = [
        {"name": "baseline", "influence_type": None, "nudge_toward": None},
    ]
    for inf_type in config["influence_types"]:
        for toward in ("value", "non_value"):
            conditions.append(
                {
                    "name": f"{inf_type}_toward_{toward}",
                    "influence_type": inf_type,
                    "nudge_toward": toward,
                }
            )
    return conditions


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _resolve_target_action(
    nudge_toward: str,
    value_is_to_do: bool,
) -> str:
    """Map (nudge_toward, value_is_to_do) → 'to_do' | 'not_to_do'."""
    toward_value_side = nudge_toward == "value"
    if toward_value_side == value_is_to_do:
        return "to_do"
    return "not_to_do"


def build_prompts_for_condition(
    vdilemmas: list[ValueDilemma],
    condition: dict,
    config: dict,
    selected_value: str,
    all_dilemmas: list[Dilemma],
) -> list[DilemmaPromptInfo]:
    """Build prompts for every filtered dilemma under one condition."""
    seed = config["seed"]
    influence_type = condition["influence_type"]
    nudge_toward = condition["nudge_toward"]

    prompt_infos: list[DilemmaPromptInfo] = []
    for vd in vdilemmas:
        dilemma = vd.dilemma
        nudge_text = None
        nudge_position = "start"
        nudge_brackets = "none"

        if influence_type is not None:
            influence = INFLUENCE_TYPES[influence_type]
            nudge_position = influence.position
            nudge_brackets = influence.brackets

            target_action = _resolve_target_action(nudge_toward, vd.value_is_to_do)

            if influence_type == "few_shot_action":
                nudge_text = generate_few_shot_action_examples(
                    target_action=target_action,
                    current_dilemma_id=dilemma.id,
                    dilemmas=all_dilemmas,
                    n_examples=config.get("n_few_shot_examples", 3),
                    seed=seed,
                )
            elif influence_type == "few_shot_value":
                nudge_text = generate_few_shot_value_examples(
                    target_value=selected_value,
                    favor_value_side=(nudge_toward == "value"),
                    current_dilemma_id=dilemma.id,
                    dilemmas=all_dilemmas,
                    n_examples=config.get("n_few_shot_examples", 3),
                    seed=seed,
                )
            else:
                # Determine value1 (nudged toward) and value2 (nudged away)
                other_side_values = (
                    dilemma.values_not_to_do
                    if vd.value_is_to_do
                    else dilemma.values_to_do
                )
                other_value = (
                    other_side_values[0] if other_side_values else "the other option"
                )

                if nudge_toward == "value":
                    value1, value2 = selected_value, other_value
                    option_text = (
                        dilemma.action_to_do
                        if vd.value_is_to_do
                        else dilemma.action_not_to_do
                    )
                else:
                    value1, value2 = other_value, selected_value
                    option_text = (
                        dilemma.action_not_to_do
                        if vd.value_is_to_do
                        else dilemma.action_to_do
                    )

                nudge_text = render_influence_text(
                    influence_type,
                    value1,
                    value2,
                    option_text=option_text,
                )

        info = build_prompt(
            dilemma=dilemma,
            nudge_text=nudge_text,
            nudge_position=nudge_position,
            nudge_brackets=nudge_brackets,
            seed=seed,
        )
        prompt_infos.append(info)

    return prompt_infos


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def results_dir_for(value: str) -> Path:
    return EXPERIMENT_DIR / "results_value" / value.replace(" ", "_")


def load_experiment_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


async def run_condition(
    model: str,
    vdilemmas: list[ValueDilemma],
    condition: dict,
    config: dict,
    selected_value: str,
    all_dilemmas: list[Dilemma],
    dry_run: bool = False,
) -> list[DilemmaResult] | None:
    """Run a single condition for one model across all filtered dilemmas."""
    condition_name = condition["name"]
    k = config["k_per_dilemma"]
    max_concurrent = config.get("max_concurrent", 100)

    save_dir = results_dir_for(selected_value) / model / condition_name
    results_path = save_dir / "results.json"

    existing_results: dict[int, dict] = {}
    if results_path.exists() and not dry_run:
        with open(results_path) as f:
            existing_data = json.load(f)
        for r in existing_data["results"]:
            existing_results[r["dilemma_id"]] = r

    print(f"\n  Condition: {condition_name}")
    all_prompt_infos = build_prompts_for_condition(
        vdilemmas,
        condition,
        config,
        selected_value,
        all_dilemmas,
    )

    # Build a lookup so we can attach value_is_to_do to results
    vd_by_id = {vd.dilemma.id: vd for vd in vdilemmas}

    prompt_infos = [
        info for info in all_prompt_infos if info.dilemma_id not in existing_results
    ]

    if not prompt_infos and not dry_run:
        print(f"  All {len(all_prompt_infos)} dilemmas already done, skipping")
        return None

    print(
        f"  {len(prompt_infos)} new dilemmas to run "
        f"({len(existing_results)} already done)"
    )

    if dry_run:
        for info in all_prompt_infos[:2]:
            print(
                f"\n  --- Dilemma {info.dilemma_id} "
                f"(to_do_is_A={info.to_do_is_a}) ---"
            )
            print(f"  System: {info.system_prompt}")
            print(f"  Prompt:\n{info.prompt_text}")
        return None

    agent = create_agent(
        model_key=model,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 16),
        concurrency_limit=max_concurrent,
    )

    system_message = prompt_infos[0].system_prompt
    prompts = [info.prompt_text for info in prompt_infos]

    print(f"  Sending {len(prompts)} prompts x K={k} = {len(prompts) * k} API calls...")

    responses_by_prompt = await generate_responses(
        agent=agent,
        prompts=prompts,
        system_message=system_message,
        K=k,
        verbose=True,
        reasoning_mode=ReasoningMode.NONE,
        valid_choices=["A", "B"],
        max_retries=2,
    )

    parsed_responses, _, _ = parse_responses_forced_choice(
        responses_by_prompt,
        choices=["A", "B"],
        verbose=True,
    )

    results: list[DilemmaResult] = []
    for prompt_idx, info in enumerate(prompt_infos):
        parsed = parsed_responses.get(prompt_idx, [])

        n_to_do = n_not_to_do = n_unparseable = 0
        for choice in parsed:
            if choice == "unparseable":
                n_unparseable += 1
            elif (choice == "A" and info.to_do_is_a) or (
                choice == "B" and not info.to_do_is_a
            ):
                n_to_do += 1
            else:
                n_not_to_do += 1

        results.append(
            DilemmaResult(
                dilemma_id=info.dilemma_id,
                to_do_is_a=info.to_do_is_a,
                value_is_to_do=vd_by_id[info.dilemma_id].value_is_to_do,
                responses=parsed,
                n_to_do=n_to_do,
                n_not_to_do=n_not_to_do,
                n_unparseable=n_unparseable,
            )
        )

    all_results_by_id = dict(existing_results)
    for r in results:
        all_results_by_id[r.dilemma_id] = asdict(r)

    save_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model,
        "condition": condition_name,
        "selected_value": selected_value,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_per_dilemma": k,
            "seed": config["seed"],
            "influence_type": condition.get("influence_type"),
            "nudge_toward": condition.get("nudge_toward"),
        },
        "results": list(all_results_by_id.values()),
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    n_new = len(results)
    n_total = len(all_results_by_id)
    print(f"  Saved {n_total} results ({n_new} new) to {results_path}")

    if prompt_infos:
        example_path = save_dir / "example_prompt.txt"
        info = prompt_infos[0]
        with open(example_path, "w") as f:
            f.write(f"System Message:\n{info.system_prompt}\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(info.prompt_text)
            f.write(f"\n\n{'=' * 60}\n")
            f.write(f"Dilemma ID: {info.dilemma_id}\n")
            f.write(f"to_do is Option A: {info.to_do_is_a}\n")
            f.write(f"Condition: {condition_name}\n")
            f.write(f"Value: {selected_value}\n")

    return results


async def run_model(
    model: str,
    vdilemmas: list[ValueDilemma],
    config: dict,
    selected_value: str,
    all_dilemmas: list[Dilemma],
    conditions: list[dict] | None = None,
    dry_run: bool = False,
) -> None:
    if conditions is None:
        conditions = get_conditions(config)

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Value: {selected_value}")
    print(f"Conditions: {len(conditions)}")
    print(f"Dilemmas (filtered): {len(vdilemmas)}")
    print(f"{'='*60}")

    for condition in conditions:
        await run_condition(
            model,
            vdilemmas,
            condition,
            config,
            selected_value,
            all_dilemmas,
            dry_run,
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

MIN_RESPONSES = 5
ALPHA = 0.05


def _load_condition(value: str, model: str, condition: str) -> dict | None:
    path = results_dir_for(value) / model / condition / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _result_counts(result: dict) -> tuple[int, int] | None:
    """Return (n_value, n_non_value) or None if insufficient data."""
    n_to_do = result["n_to_do"]
    n_not_to_do = result["n_not_to_do"]
    total = n_to_do + n_not_to_do
    if total < MIN_RESPONSES:
        return None
    if result["value_is_to_do"]:
        return n_to_do, n_not_to_do
    return n_not_to_do, n_to_do


def analyze_model(value: str, model: str) -> dict | None:
    """Compute all metrics for one model."""
    baseline_data = _load_condition(value, model, "baseline")
    if baseline_data is None:
        print(f"  No baseline for {model}")
        return None

    baseline_by_id = {r["dilemma_id"]: r for r in baseline_data["results"]}

    model_dir = results_dir_for(value) / model
    if not model_dir.exists():
        return None

    condition_dirs = [
        d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "baseline"
    ]

    # Group by influence type: inf_type → {"value": {id: result}, "non_value": ...}
    influence_conditions: dict[str, dict[str, dict[int, dict]]] = {}
    for cond_name in condition_dirs:
        cond_data = _load_condition(value, model, cond_name)
        if cond_data is None:
            continue
        inf_type = cond_data["config"].get("influence_type")
        nudge_toward = cond_data["config"].get("nudge_toward")
        if inf_type and nudge_toward:
            influence_conditions.setdefault(inf_type, {})[nudge_toward] = {
                r["dilemma_id"]: r for r in cond_data["results"]
            }

    if not influence_conditions:
        print(f"  No nudge conditions for {model}")
        return None

    # --- Baseline preference (aggregated across all dilemmas) ---
    total_value = total_non_value = 0
    for r in baseline_by_id.values():
        counts = _result_counts(r)
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

    # --- Per-influence-type metrics ---
    metrics_by_type: dict[str, dict] = {}

    # Accumulators for overall aggregate counts
    agg_c_0_A = agg_c_0_B = 0
    agg_c_A_A = agg_c_A_B = 0
    agg_c_B_A = agg_c_B_B = 0
    all_backfire_count = all_backfire_total = 0
    all_backfire_sig_count = all_backfire_sig_total = 0

    for inf_type, directions in influence_conditions.items():
        toward_value_by_id = directions.get("value", {})
        toward_non_value_by_id = directions.get("non_value", {})

        # Aggregate counts across dilemmas for this influence type
        # A = value side, B = non-value side
        c_0_A = c_0_B = 0
        c_A_A = c_A_B = 0
        c_B_A = c_B_B = 0
        backfire_count = backfire_total = 0
        backfire_sig_count = backfire_sig_total = 0

        # All dilemma IDs that appear in any direction
        all_ids = set(toward_value_by_id) | set(toward_non_value_by_id)

        for did in all_ids:
            base_r = baseline_by_id.get(did)
            if base_r is None:
                continue
            base_counts = _result_counts(base_r)
            if base_counts is None:
                continue

            # Baseline
            c_0_A += base_counts[0]
            c_0_B += base_counts[1]

            # Nudge toward value (A)
            if did in toward_value_by_id:
                tv = _result_counts(toward_value_by_id[did])
                if tv is not None:
                    c_A_A += tv[0]
                    c_A_B += tv[1]

                    n_base = base_counts[0] + base_counts[1]
                    n_nudge = tv[0] + tv[1]
                    f_base = base_counts[0] / n_base
                    f_nudge = tv[0] / n_nudge
                    backfired = f_nudge < f_base
                    sig = two_proportion_z_test(f_base, n_base, f_nudge, n_nudge, ALPHA)

                    backfire_total += 1
                    if backfired:
                        backfire_count += 1
                    if sig["is_significant"]:
                        backfire_sig_total += 1
                        if backfired:
                            backfire_sig_count += 1

            # Nudge toward non-value (B)
            if did in toward_non_value_by_id:
                tnv = _result_counts(toward_non_value_by_id[did])
                if tnv is not None:
                    c_B_A += tnv[0]
                    c_B_B += tnv[1]

                    n_base = base_counts[0] + base_counts[1]
                    n_nudge = tnv[0] + tnv[1]
                    f_base = base_counts[1] / n_base
                    f_nudge = tnv[1] / n_nudge
                    backfired = f_nudge < f_base
                    sig = two_proportion_z_test(f_base, n_base, f_nudge, n_nudge, ALPHA)

                    backfire_total += 1
                    if backfired:
                        backfire_count += 1
                    if sig["is_significant"]:
                        backfire_sig_total += 1
                        if backfired:
                            backfire_sig_count += 1

        s_A, s_B, asym, norm_asym = compute_steerability_asym_from_counts(
            c_0_A,
            c_0_B,
            c_A_A,
            c_A_B,
            c_B_A,
            c_B_B,
        )

        p_val_toward_val = c_A_A / (c_A_A + c_A_B) if (c_A_A + c_A_B) > 0 else None
        p_val_toward_non = c_B_A / (c_B_A + c_B_B) if (c_B_A + c_B_B) > 0 else None
        sig_rate = backfire_sig_total / backfire_total if backfire_total else 0.0

        metrics_by_type[inf_type] = {
            "p_val_toward_value": p_val_toward_val,
            "p_val_toward_non_value": p_val_toward_non,
            "steerability_value": s_A,
            "steerability_non_value": s_B,
            "asymmetry": asym,
            "normalized_asymmetry": norm_asym,
            "sig_rate": sig_rate,
            "backfire_rate": backfire_count / backfire_total if backfire_total else 0.0,
            "backfire_sig_rate": (
                backfire_sig_count / backfire_sig_total if backfire_sig_total else 0.0
            ),
            "n_backfires": backfire_count,
            "n_backfires_sig": backfire_sig_count,
            "n_observations": backfire_total,
            "n_sig_nudges": backfire_sig_total,
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
        all_backfire_count += backfire_count
        all_backfire_total += backfire_total
        all_backfire_sig_count += backfire_sig_count
        all_backfire_sig_total += backfire_sig_total

    s_A, s_B, asym, norm_asym = compute_steerability_asym_from_counts(
        agg_c_0_A,
        agg_c_0_B,
        agg_c_A_A,
        agg_c_A_B,
        agg_c_B_A,
        agg_c_B_B,
    )

    agg_p_val_toward_val = (
        agg_c_A_A / (agg_c_A_A + agg_c_A_B) if (agg_c_A_A + agg_c_A_B) > 0 else None
    )
    agg_p_val_toward_non = (
        agg_c_B_A / (agg_c_B_A + agg_c_B_B) if (agg_c_B_A + agg_c_B_B) > 0 else None
    )
    agg_sig_rate = (
        all_backfire_sig_total / all_backfire_total if all_backfire_total else 0.0
    )

    return {
        "model": model,
        "selected_value": value,
        "n_dilemmas": len(baseline_by_id),
        "overall": {
            "baseline_p_value_side": baseline_p_value,
            "baseline_sig": bool(baseline_test["is_significant"]),
            "p_val_toward_value": agg_p_val_toward_val,
            "p_val_toward_non_value": agg_p_val_toward_non,
            "steerability_value": s_A,
            "steerability_non_value": s_B,
            "asymmetry": asym,
            "normalized_asymmetry": norm_asym,
            "sig_rate": agg_sig_rate,
            "backfire_rate": (
                all_backfire_count / all_backfire_total if all_backfire_total else 0.0
            ),
            "backfire_sig_rate": (
                all_backfire_sig_count / all_backfire_sig_total
                if all_backfire_sig_total
                else 0.0
            ),
            "n_observations": all_backfire_total,
            "n_sig_nudges": all_backfire_sig_total,
        },
        "by_influence_type": metrics_by_type,
    }


def _fmt(v, fmt=".3f"):
    return f"{v:{fmt}}" if v is not None else "  n/a"


def print_analysis(all_metrics: list[dict]) -> None:
    value = all_metrics[0]["selected_value"] if all_metrics else "?"

    print(f"\n{'='*96}")
    print(f'VALUE-BASED DAILYDILEMMAS RESULTS  —  value = "{value}"')
    print(f"{'='*96}")

    # Overall table
    header = (
        f"{'Model':<32} {'P(val)':<8} {'P→val':<8} {'P→~val':<8} "
        f"{'s(val)':>8} {'s(~val)':>8} {'Asym':>8} {'N-Asym':>8} "
        f"{'Sig%':>6} {'BF%':>6} {'BF-s%':>6} {'N':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for m in all_metrics:
        o = m["overall"]
        print(
            f"{m['model']:<32} "
            f"{o['baseline_p_value_side']:<8.3f} "
            f"{_fmt(o['p_val_toward_value']):<8} "
            f"{_fmt(o['p_val_toward_non_value']):<8} "
            f"{_fmt(o['steerability_value']):>8} "
            f"{_fmt(o['steerability_non_value']):>8} "
            f"{_fmt(o['asymmetry']):>8} "
            f"{_fmt(o['normalized_asymmetry']):>8} "
            f"{o['sig_rate']:>5.1%} "
            f"{o['backfire_rate']:>5.1%} "
            f"{o['backfire_sig_rate']:>5.1%} "
            f"{o['n_observations']:>6d}"
        )

    # Per influence type
    inf_types: set[str] = set()
    for m in all_metrics:
        inf_types.update(m["by_influence_type"])

    for inf_type in sorted(inf_types):
        print(f"\n--- {inf_type} ---")
        sub_header = (
            f"{'Model':<32} {'P→val':<8} {'P→~val':<8} "
            f"{'s(val)':>8} {'s(~val)':>8} {'Asym':>8} "
            f"{'Sig%':>6} {'BF%':>6} {'BF-s%':>6} {'N':>6}"
        )
        print(sub_header)
        print("-" * len(sub_header))
        for m in all_metrics:
            it = m["by_influence_type"].get(inf_type)
            if it is None:
                continue
            print(
                f"{m['model']:<32} "
                f"{_fmt(it['p_val_toward_value']):<8} "
                f"{_fmt(it['p_val_toward_non_value']):<8} "
                f"{_fmt(it['steerability_value']):>8} "
                f"{_fmt(it['steerability_non_value']):>8} "
                f"{_fmt(it['asymmetry']):>8} "
                f"{it['sig_rate']:>5.1%} "
                f"{it['backfire_rate']:>5.1%} "
                f"{it['backfire_sig_rate']:>5.1%} "
                f"{it['n_observations']:>6d}"
            )


def run_analysis(value: str, models: list[str] | None = None) -> None:
    rdir = results_dir_for(value)
    if not rdir.exists():
        print(f"No results directory at {rdir}")
        sys.exit(1)

    if models is None:
        models = [d.name for d in sorted(rdir.iterdir()) if d.is_dir()]

    all_metrics = []
    for model in models:
        print(f"  Analyzing {model}...")
        metrics = analyze_model(value, model)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print_analysis(all_metrics)
        out_path = rdir / "metrics_summary.json"
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nFull metrics saved to {out_path}")
    else:
        print("No results found to analyze.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Value-based DailyDilemmas nudge experiment",
    )
    parser.add_argument(
        "--value",
        type=str,
        required=True,
        help="Moral value to filter on (e.g. 'honesty')",
    )
    parser.add_argument("--model", type=str, help="Model key from config")
    parser.add_argument(
        "--condition",
        type=str,
        help="Single condition name (e.g. baseline, survey_toward_value)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all conditions for the specified model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all models and conditions",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only compute and print metrics (no API calls)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample prompts without API calls",
    )
    parser.add_argument(
        "--max-dilemmas",
        type=int,
        help="Limit number of dilemmas (for testing)",
    )
    args = parser.parse_args()

    config = load_experiment_config()
    selected_value = args.value.strip().lower()

    if args.analyze_only:
        models = [args.model] if args.model else None
        run_analysis(selected_value, models)
        return

    all_dilemmas = load_dilemmas()
    vdilemmas = filter_dilemmas_by_value(all_dilemmas, selected_value)

    if not vdilemmas:
        print(f"No dilemmas found with value '{selected_value}' on exactly one side.")
        sys.exit(1)

    n_on_to_do = sum(1 for vd in vdilemmas if vd.value_is_to_do)
    print(f"Value: '{selected_value}'")
    print(
        f"Filtered dilemmas: {len(vdilemmas)}  "
        f"(value on to_do side: {n_on_to_do}, on not_to_do side: {len(vdilemmas) - n_on_to_do})"
    )

    if args.max_dilemmas:
        vdilemmas = vdilemmas[: args.max_dilemmas]
        print(f"Limited to {len(vdilemmas)} dilemmas")

    all_conditions = get_conditions(config)

    if args.all:
        for model in config["models"]:
            await run_model(
                model,
                vdilemmas,
                config,
                selected_value,
                all_dilemmas,
                dry_run=args.dry_run,
            )
        run_analysis(selected_value)

    elif args.model:
        if args.condition:
            matching = [c for c in all_conditions if c["name"] == args.condition]
            if not matching:
                print(f"Unknown condition: {args.condition}")
                print(f"Available: {[c['name'] for c in all_conditions]}")
                sys.exit(1)
            await run_model(
                args.model,
                vdilemmas,
                config,
                selected_value,
                all_dilemmas,
                conditions=matching,
                dry_run=args.dry_run,
            )
        elif args.all_conditions:
            await run_model(
                args.model,
                vdilemmas,
                config,
                selected_value,
                all_dilemmas,
                dry_run=args.dry_run,
            )
        else:
            parser.error("Specify --condition, --all-conditions, or --all")

        if not args.dry_run:
            run_analysis(selected_value, [args.model])
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    asyncio.run(main())
