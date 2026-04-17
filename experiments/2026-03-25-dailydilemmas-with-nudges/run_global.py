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
  - Flip rates: fraction of dilemmas where the majority choice changes

A second independent baseline (baseline_2) can be collected with --baseline-2
for test-retest reliability analysis (see analyze_global.py --baseline-reliability).

Analysis is in analyze_global.py.

Usage:
    # Run all models for specific primary values (extends previous results)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --all --values honesty safety fairness

    # Run a single model
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --model llama-33-70b --all-conditions --values honesty safety

    # Run with instructed reasoning (for non-reasoning models like llama)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --model llama-33-70b --all-conditions --reasoning before

    # Run all models, all dilemmas (no value filter)
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --all

    # Run a second baseline for test-retest reliability
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --model llama-33-70b --condition baseline_2

    # Run everything including baseline_2
    uv run python experiments/2026-03-25-dailydilemmas-with-nudges/run_global.py \
        --all --baseline-2
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


def get_conditions(config: dict, include_baseline_2: bool = False) -> list[dict]:
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
    if include_baseline_2:
        conditions.append({"name": "baseline_2", "influence_type": None})
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
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
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
        reasoning_mode=reasoning_mode,
        seed=seed,
    )


def build_prompts_for_condition(
    gdilemmas: list[GlobalDilemma],
    condition: dict,
    config: dict,
    all_dilemmas: list[Dilemma],
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
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
            info = build_prompt(
                dilemma=gd.dilemma,
                reasoning_mode=reasoning_mode,
                seed=seed,
            )
            tags.append(PromptTag(info=info, direction=None, gdilemma=gd))
        else:
            for direction in ("dir_a", "dir_b"):
                info = _build_nudge_prompt(
                    gd,
                    direction,
                    influence_type,
                    config,
                    all_dilemmas,
                    reasoning_mode=reasoning_mode,
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
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
) -> list[DilemmaResult] | None:
    """Run a single condition for one model across all dilemmas.

    Supports incremental top-up: if k_per_dilemma was increased, runs only
    the delta for prompts that already have partial results.
    """
    condition_name = condition["name"]
    k = config["k_per_dilemma"]
    max_concurrent = config.get("max_concurrent", 100)

    model_dir_name = (
        f"{model}-reasoning" if reasoning_mode == ReasoningMode.BEFORE else model
    )
    save_dir = results_dir() / model_dir_name / condition_name
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
        reasoning_mode=reasoning_mode,
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

    uses_reasoning = reasoning_mode != ReasoningMode.NONE or model_has_active_reasoning(
        model
    )
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
            reasoning_mode=reasoning_mode,
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
            "reasoning_mode": reasoning_mode.value,
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
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
    include_baseline_2: bool = False,
) -> None:
    if conditions is None:
        conditions = get_conditions(config, include_baseline_2=include_baseline_2)

    model_dir_name = (
        f"{model}-reasoning" if reasoning_mode == ReasoningMode.BEFORE else model
    )
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    if reasoning_mode != ReasoningMode.NONE:
        print(f"Reasoning: {reasoning_mode.value}  (results dir: {model_dir_name})")
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
            reasoning_mode=reasoning_mode,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main():
    parser = argparse.ArgumentParser(
        description="Global DailyDilemmas nudge experiment (primary-value pairs)",
    )
    parser.add_argument(
        "--values",
        nargs="+",
        help="Primary values to filter dilemmas for running (extends previous "
        "results). If omitted, all dilemmas with distinct primary values "
        "are included.",
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
        "--dry-run",
        action="store_true",
        help="Print sample prompts without API calls",
    )
    parser.add_argument(
        "--occurrence-threshold",
        type=int,
        default=20,
        help="Minimum number of occurrences for a primary value to be kept. "
        "Dilemmas with a primary value that appears fewer than this many "
        "times are iteratively removed until convergence (default: 20).",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["none", "before", "after"],
        default="none",
        help="Reasoning mode: none, before (reason then answer), after (answer then reason)",
    )
    parser.add_argument(
        "--max-dilemmas",
        type=int,
        help="Limit number of dilemmas (for testing)",
    )
    parser.add_argument(
        "--baseline-2",
        action="store_true",
        help="Include a second independent baseline run (baseline_2) for "
        "test-retest reliability analysis",
    )
    args = parser.parse_args()

    reasoning_mode = ReasoningMode(args.reasoning)

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

    from collections import Counter

    pv_counts: Counter[str] = Counter()
    for gd in gdilemmas:
        pv_counts[gd.primary_value_to_do] += 1
        pv_counts[gd.primary_value_not_to_do] += 1
    print(f"Dilemmas: {len(gdilemmas)}  (unique primary values: {len(pv_counts)})")
    if run_values:
        print(f"Filtered to values: {run_values}")

    if args.dry_run:
        print(f"\nSurviving primary values ({len(pv_counts)}):")
        for v, c in pv_counts.most_common():
            print(f"  {v}: {c}")

    if args.max_dilemmas:
        gdilemmas = gdilemmas[: args.max_dilemmas]
        print(f"Limited to {len(gdilemmas)} dilemmas")

    all_conditions = get_conditions(config, include_baseline_2=args.baseline_2)

    if args.dry_run and not args.all and not args.model:
        return

    if args.all:
        for model in config["models"]:
            await run_model(
                model,
                gdilemmas,
                config,
                all_dilemmas,
                dry_run=args.dry_run,
                reasoning_mode=reasoning_mode,
                include_baseline_2=args.baseline_2,
            )

    elif args.model:
        if args.condition:
            if args.condition == "baseline_2":
                all_conditions = get_conditions(config, include_baseline_2=True)
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
                reasoning_mode=reasoning_mode,
            )
        elif args.all_conditions:
            await run_model(
                args.model,
                gdilemmas,
                config,
                all_dilemmas,
                dry_run=args.dry_run,
                reasoning_mode=reasoning_mode,
                include_baseline_2=args.baseline_2,
            )
        else:
            parser.error("Specify --condition, --all-conditions, or --all")
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    asyncio.run(main())
