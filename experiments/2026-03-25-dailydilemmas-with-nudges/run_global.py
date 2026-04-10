#!/usr/bin/env python3
"""
Global DailyDilemmas runner (primary-value pairs).

Each dilemma is characterised by its *primary* value on each side (the first
entry in values_aggregated).  Dilemmas where both sides share the same primary
value are discarded.  Results are stored globally per model — one baseline file
and one file per influence type — so that per-value analysis is a post-hoc
slice with no re-running.

Nudge directions are always value-level.  For each dilemma the influence type
produces two prompts (dir_a and dir_b), both stored in the same file:
  dir_a  =  "promote the to_do-side's primary value"
  dir_b  =  "promote the not_to_do-side's primary value"

At analysis time for a queried value X:
  - Filter to dilemmas where X is a primary value on exactly one side
  - If X is the to_do-side value: dir_a → toward_value, dir_b → toward_non_value
  - Otherwise:                     dir_a → toward_non_value, dir_b → toward_value

File layout:
  results_global/<model>/baseline/results.json
  results_global/<model>/survey/results.json       (dir_a + dir_b in one file)
  results_global/<model>/emotional/results.json    (dir_a + dir_b in one file)
  ...

Metrics (same as run_value.py):
  - Baseline preference: P(choosing value-side option)
  - Steerability: log-odds shift when nudged toward each side
  - Asymmetry: s(B) - s(A) and normalized version
  - Backfiring rate: fraction of nudges that move probability against intended direction

Usage:
    # Run all models for specific primary values (extends previous results)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --all --values honesty safety fairness

    # Run a single model
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --model llama-33-70b --all-conditions --values honesty safety

    # Run all models, all dilemmas (no value filter)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --all

    # Analyze for a specific value (no API calls)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --analyze-only --value honesty

    # Cross-value overview
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --overview

    # Overview filtered, output to CSV
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --overview --values honesty safety --nudge-types user_preference -o overview.csv
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
    load_config,
    model_has_active_reasoning,
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
class GlobalDilemma:
    """A dilemma annotated with its primary values (first value on each side)."""

    dilemma: Dilemma
    primary_value_to_do: str
    primary_value_not_to_do: str


@dataclass
class DilemmaResult:
    dilemma_id: int
    direction: str | None  # "dir_a", "dir_b", or None (baseline)
    to_do_is_a: bool
    primary_value_to_do: str
    primary_value_not_to_do: str
    responses: list[str]
    n_to_do: int
    n_not_to_do: int
    n_unparseable: int


@dataclass
class PromptTag:
    """Associates a prompt with its direction and source dilemma."""

    info: DilemmaPromptInfo
    direction: str | None  # "dir_a", "dir_b", or None (baseline)
    gdilemma: GlobalDilemma

    @property
    def result_key(self) -> str:
        """Composite key for resume: 'dilemma_id' or 'dilemma_id:direction'."""
        if self.direction is None:
            return str(self.info.dilemma_id)
        return f"{self.info.dilemma_id}:{self.direction}"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_dilemmas_global(
    dilemmas: list[Dilemma],
    values: list[str] | None = None,
) -> list[GlobalDilemma]:
    """Keep dilemmas with distinct primary values.

    If *values* is provided, only keep dilemmas where at least one primary
    value is in the set.  Dilemmas whose two primary values are the same are
    always discarded.
    """
    value_set = {v.strip().lower() for v in values} if values else None
    result: list[GlobalDilemma] = []
    for d in dilemmas:
        if not d.values_to_do or not d.values_not_to_do:
            continue
        pv_to_do = d.values_to_do[0].strip().lower()
        pv_not_to_do = d.values_not_to_do[0].strip().lower()
        if pv_to_do == pv_not_to_do:
            continue
        if value_set is not None:
            if pv_to_do not in value_set and pv_not_to_do not in value_set:
                continue
        result.append(
            GlobalDilemma(
                dilemma=d,
                primary_value_to_do=pv_to_do,
                primary_value_not_to_do=pv_not_to_do,
            )
        )
    return result


def filter_rare_primary_values(
    gdilemmas: list[GlobalDilemma],
    threshold: int = 10,
) -> list[GlobalDilemma]:
    """Iteratively remove dilemmas whose primary values are too rare.

    A primary value is "rare" if it appears fewer than *threshold* times
    across all remaining dilemmas (counting both the to_do and not_to_do
    sides).  After each removal pass the counts are recomputed; the process
    repeats until no rare values remain (convergence).
    """
    from collections import Counter

    current = list(gdilemmas)
    iteration = 0
    while True:
        iteration += 1
        counts: Counter[str] = Counter()
        for gd in current:
            counts[gd.primary_value_to_do] += 1
            counts[gd.primary_value_not_to_do] += 1

        rare = {v for v, c in counts.items() if c < threshold}
        if not rare:
            break

        before = len(current)
        current = [
            gd
            for gd in current
            if gd.primary_value_to_do not in rare
            and gd.primary_value_not_to_do not in rare
        ]
        removed = before - len(current)
        print(
            f"  Occurrence filter iter {iteration}: dropped {len(rare)} rare "
            f"values, removed {removed} dilemmas "
            f"({len(current)} remaining)"
        )

    return current


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def get_conditions(config: dict) -> list[dict]:
    """Build the condition list.

    Each condition dict has:
        name: str
        influence_type: str | None

    Influence conditions produce two prompts per dilemma (dir_a and dir_b),
    both stored in the same results file.
    """
    conditions: list[dict] = [
        {"name": "baseline", "influence_type": None},
    ]
    for inf_type in config["influence_types"]:
        conditions.append({"name": inf_type, "influence_type": inf_type})
    return conditions


# ---------------------------------------------------------------------------
# Prompt construction (always value-level)
# ---------------------------------------------------------------------------


def _build_nudge_prompt(
    gd: GlobalDilemma,
    direction: str,
    influence_type: str,
    config: dict,
    all_dilemmas: list[Dilemma],
) -> DilemmaPromptInfo:
    """Build a single nudge prompt for one dilemma in one direction."""
    seed = config["seed"]
    dilemma = gd.dilemma
    influence = INFLUENCE_TYPES[influence_type]

    favours_to_do = direction == "dir_a"
    target_action = "to_do" if favours_to_do else "not_to_do"

    if favours_to_do:
        favoured_value = gd.primary_value_to_do
        disfavoured_value = gd.primary_value_not_to_do
        option_text = dilemma.action_to_do
    else:
        favoured_value = gd.primary_value_not_to_do
        disfavoured_value = gd.primary_value_to_do
        option_text = dilemma.action_not_to_do

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
            target_value=favoured_value,
            favor_value_side=True,
            current_dilemma_id=dilemma.id,
            dilemmas=all_dilemmas,
            n_examples=config.get("n_few_shot_examples", 3),
            seed=seed,
        )
    else:
        nudge_text = render_influence_text(
            influence_type,
            value1=favoured_value,
            value2=disfavoured_value,
            option_text=option_text,
        )

    return build_prompt(
        dilemma=dilemma,
        nudge_text=nudge_text,
        nudge_position=influence.position,
        nudge_brackets=influence.brackets,
        seed=seed,
    )


def build_prompts_for_condition(
    gdilemmas: list[GlobalDilemma],
    condition: dict,
    config: dict,
    all_dilemmas: list[Dilemma],
) -> list[PromptTag]:
    """Build prompts for every dilemma under one condition.

    For baseline, returns one PromptTag per dilemma (direction=None).
    For influence types, returns two PromptTags per dilemma (dir_a + dir_b).
    Nudge text always references the dilemma's own primary values.
    """
    seed = config["seed"]
    influence_type = condition["influence_type"]

    tags: list[PromptTag] = []
    for gd in gdilemmas:
        if influence_type is None:
            info = build_prompt(dilemma=gd.dilemma, seed=seed)
            tags.append(PromptTag(info=info, direction=None, gdilemma=gd))
        else:
            for direction in ("dir_a", "dir_b"):
                info = _build_nudge_prompt(
                    gd,
                    direction,
                    influence_type,
                    config,
                    all_dilemmas,
                )
                tags.append(PromptTag(info=info, direction=direction, gdilemma=gd))

    return tags


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def results_dir() -> Path:
    return EXPERIMENT_DIR / "results_global"


def load_experiment_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _result_key(r: dict) -> str:
    """Composite key for a stored result dict."""
    d = r.get("direction")
    if d is None:
        return str(r["dilemma_id"])
    return f"{r['dilemma_id']}:{d}"


def _count_responses(parsed: list[str], to_do_is_a: bool) -> tuple[int, int, int]:
    """Count (n_to_do, n_not_to_do, n_unparseable) from parsed choices."""
    n_to_do = n_not_to_do = n_unparseable = 0
    for choice in parsed:
        if choice == "unparseable":
            n_unparseable += 1
        elif (choice == "A" and to_do_is_a) or (choice == "B" and not to_do_is_a):
            n_to_do += 1
        else:
            n_not_to_do += 1
    return n_to_do, n_not_to_do, n_unparseable


def _merge_result(existing: dict, new_responses: list[str], to_do_is_a: bool) -> dict:
    """Merge new responses into an existing result dict, updating counts."""
    merged = dict(existing)
    combined = list(existing.get("responses", [])) + list(new_responses)
    n_to_do, n_not_to_do, n_unparseable = _count_responses(combined, to_do_is_a)
    merged["responses"] = combined
    merged["n_to_do"] = n_to_do
    merged["n_not_to_do"] = n_not_to_do
    merged["n_unparseable"] = n_unparseable
    return merged


async def run_condition(
    model: str,
    gdilemmas: list[GlobalDilemma],
    condition: dict,
    config: dict,
    all_dilemmas: list[Dilemma],
    dry_run: bool = False,
) -> list[DilemmaResult] | None:
    """Run a single condition for one model across all dilemmas.

    Supports incremental top-up: if k_per_dilemma was increased, runs only
    the delta for prompts that already have partial results.
    """
    condition_name = condition["name"]
    k = config["k_per_dilemma"]
    max_concurrent = config.get("max_concurrent", 100)

    save_dir = results_dir() / model / condition_name
    results_path = save_dir / "results.json"

    existing_results: dict[str, dict] = {}
    if results_path.exists() and not dry_run:
        with open(results_path) as f:
            existing_data = json.load(f)
        for r in existing_data["results"]:
            existing_results[_result_key(r)] = r

    print(f"\n  Condition: {condition_name}")
    all_tags = build_prompts_for_condition(
        gdilemmas,
        condition,
        config,
        all_dilemmas,
    )

    # Determine what needs running: new prompts get full k, existing
    # prompts with fewer than k responses get the delta.
    tags_with_delta: list[tuple[PromptTag, int]] = []
    n_complete = 0
    for tag in all_tags:
        existing = existing_results.get(tag.result_key)
        if existing is None:
            tags_with_delta.append((tag, k))
        else:
            have = len(existing.get("responses", []))
            delta = k - have
            if delta > 0:
                tags_with_delta.append((tag, delta))
            else:
                n_complete += 1

    if not tags_with_delta and not dry_run:
        print(f"  All {len(all_tags)} prompts already done with K>={k}, skipping")
        return None

    n_new = sum(1 for _, d in tags_with_delta if d == k)
    n_topup = len(tags_with_delta) - n_new
    parts = []
    if n_new:
        parts.append(f"{n_new} new")
    if n_topup:
        parts.append(f"{n_topup} top-up")
    print(f"  {' + '.join(parts)} prompts to run ({n_complete} already complete)")

    if dry_run:
        for tag in all_tags[:2]:
            info = tag.info
            print(
                f"\n  --- Dilemma {info.dilemma_id} "
                f"(dir={tag.direction}, to_do_is_A={info.to_do_is_a}) ---"
            )
            print(f"  System: {info.system_prompt}")
            print(f"  Prompt:\n{info.prompt_text}")
        return None

    uses_reasoning = model_has_active_reasoning(model)
    agent_config_key = "default_with_reasoning" if uses_reasoning else "default"
    agent_config_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "choices"
        / "config"
        / "create_agent.yaml"
    )
    agent_config = load_config(agent_config_path, agent_config_key)
    agent_config["concurrency_limit"] = max_concurrent
    agent = create_agent(model_key=model, **agent_config)

    # Group by delta_k so each API batch uses a uniform K
    from collections import defaultdict

    by_delta: dict[int, list[PromptTag]] = defaultdict(list)
    for tag, delta_k in tags_with_delta:
        by_delta[delta_k].append(tag)

    all_results_by_key = dict(existing_results)
    all_reasoning: dict[str, list] = {}
    all_reasoning_summaries: dict[str, list] = {}
    total_api_calls = 0
    any_new_tag: PromptTag | None = None

    for delta_k, batch_tags in sorted(by_delta.items()):
        system_message = batch_tags[0].info.system_prompt
        prompts = [t.info.prompt_text for t in batch_tags]
        total_api_calls += len(prompts) * delta_k

        print(
            f"  Sending {len(prompts)} prompts x K={delta_k} "
            f"= {len(prompts) * delta_k} API calls..."
        )

        responses_by_prompt = await generate_responses(
            agent=agent,
            prompts=prompts,
            system_message=system_message,
            K=delta_k,
            verbose=True,
            reasoning_mode=ReasoningMode.NONE,
            valid_choices=["A", "B"],
            max_retries=2,
        )

        parsed_responses, reasoning_results, reasoning_summaries = (
            parse_responses_forced_choice(
                responses_by_prompt,
                choices=["A", "B"],
                verbose=True,
            )
        )

        for prompt_idx, tag in enumerate(batch_tags):
            info = tag.info
            parsed = parsed_responses.get(prompt_idx, [])
            gd = tag.gdilemma
            rkey = tag.result_key

            existing = existing_results.get(rkey)
            if existing is not None:
                merged = _merge_result(existing, parsed, info.to_do_is_a)
                all_results_by_key[rkey] = merged
            else:
                n_to_do, n_not_to_do, n_unparseable = _count_responses(
                    parsed,
                    info.to_do_is_a,
                )
                all_results_by_key[rkey] = asdict(
                    DilemmaResult(
                        dilemma_id=info.dilemma_id,
                        direction=tag.direction,
                        to_do_is_a=info.to_do_is_a,
                        primary_value_to_do=gd.primary_value_to_do,
                        primary_value_not_to_do=gd.primary_value_not_to_do,
                        responses=parsed,
                        n_to_do=n_to_do,
                        n_not_to_do=n_not_to_do,
                        n_unparseable=n_unparseable,
                    )
                )

            traces = reasoning_results.get(prompt_idx, [])
            summaries = reasoning_summaries.get(prompt_idx, [])
            if traces:
                all_reasoning[rkey] = traces
            if summaries:
                all_reasoning_summaries[rkey] = summaries

            if any_new_tag is None:
                any_new_tag = tag

    save_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model,
        "condition": condition_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_per_dilemma": k,
            "seed": config["seed"],
            "influence_type": condition.get("influence_type"),
        },
        "results": list(all_results_by_key.values()),
    }
    if all_reasoning:
        output["reasoning"] = all_reasoning
    if all_reasoning_summaries:
        output["reasoning_summaries"] = all_reasoning_summaries
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    n_total = len(all_results_by_key)
    print(
        f"  Saved {n_total} results "
        f"({n_new} new, {n_topup} topped up, {total_api_calls} API calls) "
        f"to {results_path}"
    )

    if any_new_tag is not None:
        example_path = save_dir / "example_prompt.txt"
        tag = any_new_tag
        info = tag.info
        gd = tag.gdilemma
        with open(example_path, "w") as f:
            f.write(f"System Message:\n{info.system_prompt}\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(info.prompt_text)
            f.write(f"\n\n{'=' * 60}\n")
            f.write(f"Dilemma ID: {info.dilemma_id}\n")
            f.write(f"Direction: {tag.direction}\n")
            f.write(f"to_do is Option A: {info.to_do_is_a}\n")
            f.write(f"Condition: {condition_name}\n")
            f.write(
                f"Primary values: {gd.primary_value_to_do} vs "
                f"{gd.primary_value_not_to_do}\n"
            )

    return None


async def run_model(
    model: str,
    gdilemmas: list[GlobalDilemma],
    config: dict,
    all_dilemmas: list[Dilemma],
    conditions: list[dict] | None = None,
    dry_run: bool = False,
) -> None:
    if conditions is None:
        conditions = get_conditions(config)

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Conditions: {len(conditions)}")
    print(f"Dilemmas: {len(gdilemmas)}")
    print(f"{'='*60}")

    for condition in conditions:
        await run_condition(
            model,
            gdilemmas,
            condition,
            config,
            all_dilemmas,
            dry_run,
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

MIN_RESPONSES = 5
ALPHA = 0.05


def _load_condition(model: str, condition: str) -> dict | None:
    path = results_dir() / model / condition / "results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _result_counts_for_value(result: dict, value: str) -> tuple[int, int, bool] | None:
    """Return (n_value, n_non_value, value_is_to_do) or None if insufficient data.

    Determines which side holds *value* from the stored primary values and
    maps n_to_do / n_not_to_do accordingly.
    """
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
    """Return {dilemma_id: result} for dilemmas where *value* is a primary value.

    Only works for results without a direction (i.e. baseline).
    """
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
    """Split influence results into {direction: {dilemma_id: result}}.

    Only keeps dilemmas where *value* is a primary value.
    """
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
        print(f"  No baseline for {model}")
        return None

    baseline_by_id = _filter_results_by_value(
        baseline_data["results"], value, allowed_ids
    )
    if not baseline_by_id:
        print(f"  No baseline dilemmas for value '{value}' in {model}")
        return None

    model_dir = results_dir() / model
    if not model_dir.exists():
        return None

    condition_dirs = [
        d.name for d in model_dir.iterdir() if d.is_dir() and d.name != "baseline"
    ]

    # inf_type → {"dir_a": {id: result}, "dir_b": {id: result}}
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
        print(f"  No nudge conditions for {model}")
        return None

    # --- Baseline preference (aggregated across all matching dilemmas) ---
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

    # --- Per-influence-type metrics ---
    metrics_by_type: dict[str, dict] = {}

    agg_c_0_A = agg_c_0_B = 0
    agg_c_A_A = agg_c_A_B = 0
    agg_c_B_A = agg_c_B_B = 0
    all_backfire_count = all_backfire_total = 0
    all_backfire_sig_count = all_backfire_sig_total = 0

    for inf_type, directions in influence_conditions.items():
        dir_a_by_id = directions.get("dir_a", {})
        dir_b_by_id = directions.get("dir_b", {})

        c_0_A = c_0_B = 0
        c_A_A = c_A_B = 0
        c_B_A = c_B_B = 0
        backfire_count = backfire_total = 0
        backfire_sig_count = backfire_sig_total = 0

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

            # Determine which stored direction is toward_value / toward_non_value
            # dir_a favours the to_do-side value
            if value_is_to_do:
                toward_value_id = "dir_a"
                toward_non_value_id = "dir_b"
            else:
                toward_value_id = "dir_b"
                toward_non_value_id = "dir_a"

            tv_by_id = directions.get(toward_value_id, {})
            tnv_by_id = directions.get(toward_non_value_id, {})

            # Nudge toward value (A)
            if did in tv_by_id:
                tv_counts = _result_counts_for_value(tv_by_id[did], value)
                if tv_counts is not None:
                    c_A_A += tv_counts[0]
                    c_A_B += tv_counts[1]

                    n_base = n_val_base + n_nval_base
                    n_nudge = tv_counts[0] + tv_counts[1]
                    f_base = n_val_base / n_base
                    f_nudge = tv_counts[0] / n_nudge
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
            if did in tnv_by_id:
                tnv_counts = _result_counts_for_value(tnv_by_id[did], value)
                if tnv_counts is not None:
                    c_B_A += tnv_counts[0]
                    c_B_B += tnv_counts[1]

                    n_base = n_val_base + n_nval_base
                    n_nudge = tnv_counts[0] + tnv_counts[1]
                    f_base = n_nval_base / n_base
                    f_nudge = tnv_counts[1] / n_nudge
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


# ---------------------------------------------------------------------------
# Printing / overview helpers
# ---------------------------------------------------------------------------


def _fmt(v, fmt=".3f"):
    return f"{v:{fmt}}" if v is not None else "  n/a"


def _abs(v):
    return abs(v) if v is not None else None


def _safe_mean_of(dicts: list[dict], key: str):
    vals = [d[key] for d in dicts if d.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _safe_mean_abs_of(dicts: list[dict], key: str):
    vals = [abs(d[key]) for d in dicts if d.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def print_analysis(all_metrics: list[dict]) -> None:
    value = all_metrics[0]["selected_value"] if all_metrics else "?"

    print(f"\n{'='*96}")
    print(f'GLOBAL DAILYDILEMMAS RESULTS  —  value = "{value}"')
    print(f"{'='*96}")

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

    if len(all_metrics) > 1:
        overalls = [m["overall"] for m in all_metrics]
        n_total = sum(o.get("n_observations", 0) for o in overalls)
        print("-" * len(header))
        print(
            f"{'  across models':<32} "
            f"{_safe_mean_of(overalls, 'baseline_p_value_side') or 0:<8.3f} "
            f"{_fmt(_safe_mean_of(overalls, 'p_val_toward_value')):<8} "
            f"{_fmt(_safe_mean_of(overalls, 'p_val_toward_non_value')):<8} "
            f"{_fmt(_safe_mean_of(overalls, 'steerability_value')):>8} "
            f"{_fmt(_safe_mean_of(overalls, 'steerability_non_value')):>8} "
            f"{_fmt(_safe_mean_of(overalls, 'asymmetry')):>8} "
            f"{_fmt(_safe_mean_of(overalls, 'normalized_asymmetry')):>8} "
            f"{_safe_mean_of(overalls, 'sig_rate') or 0:>5.1%} "
            f"{_safe_mean_of(overalls, 'backfire_rate') or 0:>5.1%} "
            f"{_safe_mean_of(overalls, 'backfire_sig_rate') or 0:>5.1%} "
            f"{n_total:>6d}"
        )

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
        type_rows: list[dict] = []
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
            type_rows.append(it)
        if len(type_rows) > 1:
            n_total = sum(r.get("n_observations", 0) for r in type_rows)
            print("-" * len(sub_header))
            print(
                f"{'  across models':<32} "
                f"{_fmt(_safe_mean_of(type_rows, 'p_val_toward_value')):<8} "
                f"{_fmt(_safe_mean_of(type_rows, 'p_val_toward_non_value')):<8} "
                f"{_fmt(_safe_mean_of(type_rows, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean_of(type_rows, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean_of(type_rows, 'asymmetry')):>8} "
                f"{_safe_mean_of(type_rows, 'sig_rate') or 0:>5.1%} "
                f"{_safe_mean_of(type_rows, 'backfire_rate') or 0:>5.1%} "
                f"{_safe_mean_of(type_rows, 'backfire_sig_rate') or 0:>5.1%} "
                f"{n_total:>6d}"
            )


def _discover_values(model: str) -> set[str]:
    """Discover all primary values present in a model's baseline results."""
    baseline = _load_condition(model, "baseline")
    if baseline is None:
        return set()
    values: set[str] = set()
    for r in baseline["results"]:
        values.add(r["primary_value_to_do"])
        values.add(r["primary_value_not_to_do"])
    return values


def run_overview(
    models: list[str] | None = None,
    values: list[str] | None = None,
    nudge_types: list[str] | None = None,
    output: str | None = None,
) -> None:
    """Compute and display an overview table across values."""
    rdir = results_dir()
    if not rdir.exists():
        print(f"No results directory at {rdir}")
        sys.exit(1)

    if models is None:
        models = sorted(d.name for d in rdir.iterdir() if d.is_dir())

    if not models:
        print("No model directories found.")
        return

    available_values: set[str] = set()
    for model in models:
        available_values |= _discover_values(model)

    if values:
        value_list = sorted(
            v.strip().lower() for v in values if v.strip().lower() in available_values
        )
    else:
        value_list = sorted(available_values)

    if not value_list:
        print("No matching values found in results.")
        return

    all_metrics: list[dict] = []
    for value in value_list:
        for model in models:
            metrics = analyze_model(model, value)
            if metrics:
                all_metrics.append(metrics)

    if not all_metrics:
        print("No results found.")
        return

    if output:
        _write_overview_csv(all_metrics, output, nudge_types)
    else:
        print_overview(all_metrics, nudge_types)


def _print_overview_across_models_row(
    value: str,
    overalls: list[dict],
    header_len: int,
) -> None:
    n_total = sum(o.get("n_observations", 0) for o in overalls)
    print(
        f"{value:<14} "
        f"{'  across models':<32} "
        f"{'':>4} "
        f"{_safe_mean_of(overalls, 'baseline_p_value_side') or 0:<8.3f} "
        f"{_fmt(_safe_mean_of(overalls, 'steerability_value')):>8} "
        f"{_fmt(_safe_mean_of(overalls, 'steerability_non_value')):>8} "
        f"{_fmt(_safe_mean_abs_of(overalls, 'steerability_value')):>8} "
        f"{_fmt(_safe_mean_abs_of(overalls, 'steerability_non_value')):>8} "
        f"{_fmt(_safe_mean_of(overalls, 'asymmetry')):>8} "
        f"{_fmt(_safe_mean_of(overalls, 'normalized_asymmetry')):>8} "
        f"{_safe_mean_of(overalls, 'sig_rate') or 0:>5.1%} "
        f"{_safe_mean_of(overalls, 'backfire_rate') or 0:>5.1%} "
        f"{n_total:>6d}"
    )
    print("-" * header_len)


def _print_nudge_overall_row(
    model: str,
    rows: list[dict],
    header_len: int,
) -> None:
    def _safe_mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    def _safe_mean_abs(key):
        vals = [abs(r[key]) for r in rows if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    n_total = sum(r.get("n_observations", 0) for r in rows)
    print(
        f"{'  overall':<14} "
        f"{model:<32} "
        f"{_fmt(_safe_mean('steerability_value')):>8} "
        f"{_fmt(_safe_mean('steerability_non_value')):>8} "
        f"{_fmt(_safe_mean_abs('steerability_value')):>8} "
        f"{_fmt(_safe_mean_abs('steerability_non_value')):>8} "
        f"{_fmt(_safe_mean('asymmetry')):>8} "
        f"{_safe_mean('sig_rate') or 0:>5.1%} "
        f"{_safe_mean('backfire_rate') or 0:>5.1%} "
        f"{n_total:>6d}"
    )
    print("-" * header_len)


def print_overview(
    all_metrics: list[dict],
    nudge_types: list[str] | None = None,
) -> None:
    print(f"\n{'='*110}")
    print("CROSS-VALUE OVERVIEW (global)")
    print(f"{'='*110}")

    header = (
        f"{'Value':<14} {'Model':<32} {'n':>4} {'P(val)':<8} "
        f"{'s(val)':>8} {'s(~val)':>8} {'|s(v)|':>8} {'|s(~v)|':>8} "
        f"{'Asym':>8} {'N-Asym':>8} "
        f"{'Sig%':>6} {'BF%':>6} {'N':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    sorted_m = sorted(all_metrics, key=lambda x: (x["selected_value"], x["model"]))
    current_value = None
    value_overalls: list[dict] = []
    for m in sorted_m:
        if (
            current_value is not None
            and m["selected_value"] != current_value
            and len(value_overalls) > 1
        ):
            _print_overview_across_models_row(
                current_value, value_overalls, len(header)
            )
        if m["selected_value"] != current_value:
            value_overalls = []
            current_value = m["selected_value"]
        o = m["overall"]
        print(
            f"{m['selected_value']:<14} "
            f"{m['model']:<32} "
            f"{m['n_dilemmas']:>4d} "
            f"{o['baseline_p_value_side']:<8.3f} "
            f"{_fmt(o['steerability_value']):>8} "
            f"{_fmt(o['steerability_non_value']):>8} "
            f"{_fmt(_abs(o['steerability_value'])):>8} "
            f"{_fmt(_abs(o['steerability_non_value'])):>8} "
            f"{_fmt(o['asymmetry']):>8} "
            f"{_fmt(o['normalized_asymmetry']):>8} "
            f"{o['sig_rate']:>5.1%} "
            f"{o['backfire_rate']:>5.1%} "
            f"{o['n_observations']:>6d}"
        )
        value_overalls.append(o)
    if current_value is not None and len(value_overalls) > 1:
        _print_overview_across_models_row(current_value, value_overalls, len(header))

    inf_types: set[str] = set()
    for m in all_metrics:
        inf_types.update(m["by_influence_type"])

    type_list = sorted(inf_types)
    if nudge_types:
        type_list = [t for t in type_list if t in nudge_types]

    for inf_type in type_list:
        print(f"\n--- {inf_type} ---")
        sub_header = (
            f"{'Value':<14} {'Model':<32} "
            f"{'s(val)':>8} {'s(~val)':>8} {'|s(v)|':>8} {'|s(~v)|':>8} "
            f"{'Asym':>8} "
            f"{'Sig%':>6} {'BF%':>6} {'N':>6}"
        )
        print(sub_header)
        print("-" * len(sub_header))

        sorted_metrics = sorted(
            all_metrics, key=lambda x: (x["model"], x["selected_value"])
        )
        current_model = None
        model_rows: list[dict] = []
        all_model_overall_rows: list[dict] = []
        for m in sorted_metrics:
            it = m["by_influence_type"].get(inf_type)
            if it is None:
                continue
            if current_model is not None and m["model"] != current_model and model_rows:
                _print_nudge_overall_row(current_model, model_rows, len(sub_header))
                all_model_overall_rows.extend(model_rows)
                model_rows = []
            current_model = m["model"]
            print(
                f"{m['selected_value']:<14} "
                f"{m['model']:<32} "
                f"{_fmt(it['steerability_value']):>8} "
                f"{_fmt(it['steerability_non_value']):>8} "
                f"{_fmt(_abs(it['steerability_value'])):>8} "
                f"{_fmt(_abs(it['steerability_non_value'])):>8} "
                f"{_fmt(it['asymmetry']):>8} "
                f"{it['sig_rate']:>5.1%} "
                f"{it['backfire_rate']:>5.1%} "
                f"{it['n_observations']:>6d}"
            )
            model_rows.append(it)
        if current_model is not None and model_rows:
            _print_nudge_overall_row(current_model, model_rows, len(sub_header))
            all_model_overall_rows.extend(model_rows)
        if (
            len(
                {
                    m["model"]
                    for m in sorted_metrics
                    if m["by_influence_type"].get(inf_type)
                }
            )
            > 1
        ):
            n_total = sum(r.get("n_observations", 0) for r in all_model_overall_rows)
            print(
                f"{'  across models':<14} "
                f"{'':<32} "
                f"{_fmt(_safe_mean_of(all_model_overall_rows, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean_of(all_model_overall_rows, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean_abs_of(all_model_overall_rows, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean_abs_of(all_model_overall_rows, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean_of(all_model_overall_rows, 'asymmetry')):>8} "
                f"{_safe_mean_of(all_model_overall_rows, 'sig_rate') or 0:>5.1%} "
                f"{_safe_mean_of(all_model_overall_rows, 'backfire_rate') or 0:>5.1%} "
                f"{n_total:>6d}"
            )
            print("=" * len(sub_header))

    model_names = sorted({m["model"] for m in all_metrics})
    if len(all_metrics) > len(model_names):
        print("\n--- Average across values (per model) ---")
        avg_header = (
            f"{'Model':<32} {'n_vals':>6} "
            f"{'s(val)':>8} {'s(~val)':>8} {'|s(v)|':>8} {'|s(~v)|':>8} "
            f"{'Asym':>8} {'N-Asym':>8} "
            f"{'Sig%':>6} {'BF%':>6}"
        )
        print(avg_header)
        print("-" * len(avg_header))
        for model in model_names:
            mm = [m for m in all_metrics if m["model"] == model]
            n_vals = len(mm)
            vals_s_A = [
                m["overall"]["steerability_value"]
                for m in mm
                if m["overall"]["steerability_value"] is not None
            ]
            vals_s_B = [
                m["overall"]["steerability_non_value"]
                for m in mm
                if m["overall"]["steerability_non_value"] is not None
            ]
            vals_asym = [
                m["overall"]["asymmetry"]
                for m in mm
                if m["overall"]["asymmetry"] is not None
            ]
            vals_nasym = [
                m["overall"]["normalized_asymmetry"]
                for m in mm
                if m["overall"]["normalized_asymmetry"] is not None
            ]
            vals_sig = [m["overall"]["sig_rate"] for m in mm]
            vals_bf = [m["overall"]["backfire_rate"] for m in mm]

            def _mean(vs):
                return sum(vs) / len(vs) if vs else None

            def _mean_abs(vs):
                return sum(abs(v) for v in vs) / len(vs) if vs else None

            print(
                f"{model:<32} {n_vals:>6d} "
                f"{_fmt(_mean(vals_s_A)):>8} "
                f"{_fmt(_mean(vals_s_B)):>8} "
                f"{_fmt(_mean_abs(vals_s_A)):>8} "
                f"{_fmt(_mean_abs(vals_s_B)):>8} "
                f"{_fmt(_mean(vals_asym)):>8} "
                f"{_fmt(_mean(vals_nasym)):>8} "
                f"{_mean(vals_sig) or 0:>5.1%} "
                f"{_mean(vals_bf) or 0:>5.1%}"
            )

        if len(model_names) > 1:
            all_overalls = [m["overall"] for m in all_metrics]
            n_vals_total = len(all_metrics)
            print("-" * len(avg_header))
            print(
                f"{'  across models':<32} {n_vals_total:>6d} "
                f"{_fmt(_safe_mean_of(all_overalls, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean_of(all_overalls, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean_abs_of(all_overalls, 'steerability_value')):>8} "
                f"{_fmt(_safe_mean_abs_of(all_overalls, 'steerability_non_value')):>8} "
                f"{_fmt(_safe_mean_of(all_overalls, 'asymmetry')):>8} "
                f"{_fmt(_safe_mean_of(all_overalls, 'normalized_asymmetry')):>8} "
                f"{_safe_mean_of(all_overalls, 'sig_rate') or 0:>5.1%} "
                f"{_safe_mean_of(all_overalls, 'backfire_rate') or 0:>5.1%}"
            )


def _write_overview_csv(
    all_metrics: list[dict],
    output_path: str,
    nudge_types: list[str] | None = None,
) -> None:
    import csv as csv_mod

    rows = []
    for m in sorted(all_metrics, key=lambda x: (x["selected_value"], x["model"])):
        o = m["overall"]
        base_row = {
            "value": m["selected_value"],
            "model": m["model"],
            "n_dilemmas": m["n_dilemmas"],
            "influence_type": "overall",
            "baseline_p_value": o["baseline_p_value_side"],
            "p_toward_value": o.get("p_val_toward_value"),
            "p_toward_non_value": o.get("p_val_toward_non_value"),
            "steerability_value": o["steerability_value"],
            "steerability_non_value": o["steerability_non_value"],
            "asymmetry": o["asymmetry"],
            "normalized_asymmetry": o["normalized_asymmetry"],
            "sig_rate": o["sig_rate"],
            "backfire_rate": o["backfire_rate"],
            "n_observations": o["n_observations"],
        }
        rows.append(base_row)

        for inf_type, it in sorted(m["by_influence_type"].items()):
            if nudge_types and inf_type not in nudge_types:
                continue
            rows.append(
                {
                    "value": m["selected_value"],
                    "model": m["model"],
                    "n_dilemmas": m["n_dilemmas"],
                    "influence_type": inf_type,
                    "baseline_p_value": o["baseline_p_value_side"],
                    "p_toward_value": it.get("p_val_toward_value"),
                    "p_toward_non_value": it.get("p_val_toward_non_value"),
                    "steerability_value": it["steerability_value"],
                    "steerability_non_value": it["steerability_non_value"],
                    "asymmetry": it["asymmetry"],
                    "normalized_asymmetry": it.get("normalized_asymmetry"),
                    "sig_rate": it["sig_rate"],
                    "backfire_rate": it["backfire_rate"],
                    "n_observations": it["n_observations"],
                }
            )

    fieldnames = [
        "value",
        "model",
        "n_dilemmas",
        "influence_type",
        "baseline_p_value",
        "p_toward_value",
        "p_toward_non_value",
        "steerability_value",
        "steerability_non_value",
        "asymmetry",
        "normalized_asymmetry",
        "sig_rate",
        "backfire_rate",
        "n_observations",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


def run_analysis(
    value: str,
    models: list[str] | None = None,
) -> None:
    rdir = results_dir()
    if not rdir.exists():
        print(f"No results directory at {rdir}")
        sys.exit(1)

    if models is None:
        models = sorted(d.name for d in rdir.iterdir() if d.is_dir())

    all_metrics = []
    for model in models:
        print(f"  Analyzing {model} for value '{value}'...")
        metrics = analyze_model(model, value)
        if metrics:
            all_metrics.append(metrics)

    if all_metrics:
        print_analysis(all_metrics)
        out_path = rdir / f"metrics_{value.replace(' ', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nFull metrics saved to {out_path}")
    else:
        print(f"No results found for value '{value}'.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Global DailyDilemmas nudge experiment (primary-value pairs)",
    )
    parser.add_argument(
        "--value",
        type=str,
        help="Moral value to analyze for (e.g. 'honesty'). "
        "Required for --analyze-only.",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        help="Primary values to filter dilemmas for running (extends previous "
        "results). Also used to filter overview. If omitted during a run, "
        "all dilemmas with distinct primary values are included.",
    )
    parser.add_argument("--model", type=str, help="Model key from config")
    parser.add_argument(
        "--condition",
        type=str,
        help="Single condition name (e.g. baseline, survey, emotional)",
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
        help="Only compute and print metrics (no API calls). "
        "Requires --value; optionally --values to restrict dilemma set.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Print a cross-value overview table. Supports --model, --values, "
        "--nudge-types, and --output filters.",
    )
    parser.add_argument(
        "--nudge-types",
        nargs="+",
        help="List of nudge/influence types to show in per-type breakdown "
        "(default: all). E.g. --nudge-types user_preference emotional",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write overview to a CSV file instead of printing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample prompts without API calls",
    )
    parser.add_argument(
        "--occurrence-threshold",
        type=int,
        default=10,
        help="Minimum number of occurrences for a primary value to be kept. "
        "Dilemmas with a primary value that appears fewer than this many "
        "times are iteratively removed until convergence (default: 10).",
    )
    parser.add_argument(
        "--max-dilemmas",
        type=int,
        help="Limit number of dilemmas (for testing)",
    )
    args = parser.parse_args()

    # --- Overview mode ---
    if args.overview:
        models = [args.model] if args.model else None
        run_overview(
            models=models,
            values=args.values,
            nudge_types=args.nudge_types,
            output=args.output,
        )
        return

    # --- Analyze-only mode ---
    if args.analyze_only:
        if not args.value:
            parser.error("--analyze-only requires --value")
        selected_value = args.value.strip().lower()
        models = [args.model] if args.model else None
        run_analysis(selected_value, models)
        return

    # --- Run mode ---
    config = load_experiment_config()
    all_dilemmas = load_dilemmas()

    run_values = [v.strip().lower() for v in args.values] if args.values else None
    gdilemmas = filter_dilemmas_global(all_dilemmas, run_values)

    if not gdilemmas:
        print("No dilemmas with distinct primary values found.")
        sys.exit(1)

    if args.occurrence_threshold > 0:
        n_before = len(gdilemmas)
        gdilemmas = filter_rare_primary_values(
            gdilemmas, threshold=args.occurrence_threshold
        )
        if len(gdilemmas) < n_before:
            print(
                f"  Occurrence threshold ({args.occurrence_threshold}): "
                f"{n_before} → {len(gdilemmas)} dilemmas"
            )

    if not gdilemmas:
        print("No dilemmas remaining after occurrence filtering.")
        sys.exit(1)

    primary_values = set()
    for gd in gdilemmas:
        primary_values.add(gd.primary_value_to_do)
        primary_values.add(gd.primary_value_not_to_do)
    print(f"Dilemmas: {len(gdilemmas)}  (unique primary values: {len(primary_values)})")
    if run_values:
        print(f"Filtered to values: {run_values}")

    if args.max_dilemmas:
        gdilemmas = gdilemmas[: args.max_dilemmas]
        print(f"Limited to {len(gdilemmas)} dilemmas")

    all_conditions = get_conditions(config)

    if args.all:
        for model in config["models"]:
            await run_model(
                model,
                gdilemmas,
                config,
                all_dilemmas,
                dry_run=args.dry_run,
            )
        if not args.dry_run and args.value:
            run_analysis(args.value.strip().lower())

    elif args.model:
        if args.condition:
            matching = [c for c in all_conditions if c["name"] == args.condition]
            if not matching:
                print(f"Unknown condition: {args.condition}")
                print(f"Available: {[c['name'] for c in all_conditions]}")
                sys.exit(1)
            await run_model(
                args.model,
                gdilemmas,
                config,
                all_dilemmas,
                conditions=matching,
                dry_run=args.dry_run,
            )
        elif args.all_conditions:
            await run_model(
                args.model,
                gdilemmas,
                config,
                all_dilemmas,
                dry_run=args.dry_run,
            )
        else:
            parser.error("Specify --condition, --all-conditions, or --all")

        if not args.dry_run and args.value:
            run_analysis(args.value.strip().lower(), [args.model])
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    asyncio.run(main())
