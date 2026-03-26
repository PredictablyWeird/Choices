#!/usr/bin/env python3
"""
Main runner for the DailyDilemmas contextual influence experiment.

For each (model, condition), generates prompts for all dilemmas, sends them
via the choices API infrastructure, parses responses, and saves results as JSON.

Usage:
    # Run a single model + condition
    uv run python experiments/dailydilemmas_nudges/run.py \
        --model llama-33-70b --condition baseline

    # Run all conditions for a model
    uv run python experiments/dailydilemmas_nudges/run.py \
        --model llama-33-70b --all-conditions

    # Run everything
    uv run python experiments/dailydilemmas_nudges/run.py --all

    # Dry run: print sample prompts without API calls
    uv run python experiments/dailydilemmas_nudges/run.py \
        --model llama-33-70b --condition baseline --dry-run
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

# Flat imports — add experiment dir to path
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
RESULTS_DIR = EXPERIMENT_DIR / "results"
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"


@dataclass
class DilemmaResult:
    dilemma_id: int
    to_do_is_a: bool
    responses: list[str]  # parsed choices: "A", "B", or "unparseable"
    n_to_do: int
    n_not_to_do: int
    n_unparseable: int


def load_experiment_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_conditions(config: dict, dilemmas: list[Dilemma]) -> list[dict]:
    """
    Build the list of all conditions to run.

    Each condition is a dict with:
        name: str — condition identifier
        influence_type: str | None
        target_action: str | None — "to_do" or "not_to_do"
    """
    conditions = [{"name": "baseline", "influence_type": None, "target_action": None}]

    for inf_type in config["influence_types"]:
        for target_action in ("to_do", "not_to_do"):
            conditions.append(
                {
                    "name": f"{inf_type}_toward_{target_action}",
                    "influence_type": inf_type,
                    "target_action": target_action,
                }
            )

    return conditions


def build_prompts_for_condition(
    dilemmas: list[Dilemma],
    condition: dict,
    config: dict,
) -> list[DilemmaPromptInfo]:
    """Build all prompts for a single condition across all dilemmas."""
    seed = config["seed"]
    influence_type = condition["influence_type"]
    target_action = condition["target_action"]

    prompt_infos = []
    for dilemma in dilemmas:
        nudge_text = None
        nudge_position = "start"
        nudge_brackets = "none"

        if influence_type is not None:
            influence = INFLUENCE_TYPES[influence_type]
            nudge_position = influence.position
            nudge_brackets = influence.brackets

            if influence_type == "few_shot_action":
                nudge_text = generate_few_shot_action_examples(
                    target_action=target_action,
                    current_dilemma_id=dilemma.id,
                    dilemmas=dilemmas,
                    n_examples=config.get("n_few_shot_examples", 3),
                    seed=seed,
                )
            elif influence_type == "few_shot_value":
                target_values = (
                    dilemma.values_to_do
                    if target_action == "to_do"
                    else dilemma.values_not_to_do
                )
                target_value = target_values[0] if target_values else "the first option"
                nudge_text = generate_few_shot_value_examples(
                    target_value=target_value,
                    favor_value_side=(target_action == "to_do"),
                    current_dilemma_id=dilemma.id,
                    dilemmas=dilemmas,
                    n_examples=config.get("n_few_shot_examples", 3),
                    seed=seed,
                )
            else:
                # Determine which value is being nudged toward
                if target_action == "to_do":
                    value1 = (
                        dilemma.values_to_do[0]
                        if dilemma.values_to_do
                        else "the first option"
                    )
                    value2 = (
                        dilemma.values_not_to_do[0]
                        if dilemma.values_not_to_do
                        else "the second option"
                    )
                    option_text = dilemma.action_to_do
                else:
                    value1 = (
                        dilemma.values_not_to_do[0]
                        if dilemma.values_not_to_do
                        else "the second option"
                    )
                    value2 = (
                        dilemma.values_to_do[0]
                        if dilemma.values_to_do
                        else "the first option"
                    )
                    option_text = dilemma.action_not_to_do

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


async def run_condition(
    model: str,
    dilemmas: list[Dilemma],
    condition: dict,
    config: dict,
    dry_run: bool = False,
) -> list[DilemmaResult] | None:
    """Run a single condition for a single model across all dilemmas."""
    condition_name = condition["name"]
    k = config["k_per_dilemma"]
    max_concurrent = config.get("max_concurrent", 100)

    save_dir = RESULTS_DIR / model / condition_name
    results_path = save_dir / "results.json"

    # Load existing results and figure out which dilemmas still need running
    existing_results: dict[int, dict] = {}
    if results_path.exists() and not dry_run:
        with open(results_path) as f:
            existing_data = json.load(f)
        for r in existing_data["results"]:
            existing_results[r["dilemma_id"]] = r

    print(f"\n  Condition: {condition_name}")
    all_prompt_infos = build_prompts_for_condition(dilemmas, condition, config)

    # Filter to only dilemmas we haven't run yet
    prompt_infos = [
        info for info in all_prompt_infos if info.dilemma_id not in existing_results
    ]

    if not prompt_infos and not dry_run:
        print(f"  All {len(all_prompt_infos)} dilemmas already done, skipping")
        return None

    print(
        f"  {len(prompt_infos)} new dilemmas to run ({len(existing_results)} already done)"
    )

    if dry_run:
        # Print a few sample prompts
        for info in all_prompt_infos[:2]:
            print(
                f"\n  --- Dilemma {info.dilemma_id} (to_do_is_A={info.to_do_is_a}) ---"
            )
            print(f"  System: {info.system_prompt}")
            print(f"  Prompt:\n{info.prompt_text}")
        return None

    # Create agent
    agent = create_agent(
        model_key=model,
        temperature=config.get("temperature", 0.7),
        max_tokens=config.get("max_tokens", 16),
        concurrency_limit=max_concurrent,
    )

    # All prompts share the same system message
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

    # Parse responses
    parsed_responses, reasoning_results, reasoning_summaries = (
        parse_responses_forced_choice(
            responses_by_prompt, choices=["A", "B"], verbose=True
        )
    )

    # Build results
    results = []
    for prompt_idx, info in enumerate(prompt_infos):
        parsed = parsed_responses.get(prompt_idx, [])

        # Map A/B to to_do/not_to_do
        n_to_do = 0
        n_not_to_do = 0
        n_unparseable = 0
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
                responses=parsed,
                n_to_do=n_to_do,
                n_not_to_do=n_not_to_do,
                n_unparseable=n_unparseable,
            )
        )

    # Merge with existing results and save
    all_results_by_id = dict(existing_results)  # start with existing
    for r in results:
        all_results_by_id[r.dilemma_id] = asdict(r)  # new results overwrite

    save_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "model": model,
        "condition": condition_name,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "k_per_dilemma": k,
            "seed": config["seed"],
            "influence_type": condition.get("influence_type"),
            "target_action": condition.get("target_action"),
        },
        "results": list(all_results_by_id.values()),
    }
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    n_new = len(results)
    n_total = len(all_results_by_id)
    print(f"  Saved {n_total} results ({n_new} new) to {results_path}")

    # Save an example prompt for inspection
    example_path = save_dir / "example_prompt.txt"
    if prompt_infos:
        info = prompt_infos[0]
        with open(example_path, "w") as f:
            f.write(f"System Message:\n{info.system_prompt}\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(info.prompt_text)
            f.write(f"\n\n{'=' * 60}\n")
            f.write(f"Dilemma ID: {info.dilemma_id}\n")
            f.write(f"to_do is Option A: {info.to_do_is_a}\n")
            f.write(f"Condition: {condition_name}\n")

    return results


async def run_model(
    model: str,
    dilemmas: list[Dilemma],
    config: dict,
    conditions: list[dict] | None = None,
    dry_run: bool = False,
) -> None:
    """Run all (or specified) conditions for a single model."""
    if conditions is None:
        conditions = get_conditions(config, dilemmas)

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Conditions: {len(conditions)}")
    print(f"Dilemmas: {len(dilemmas)}")
    print(f"{'='*60}")

    for condition in conditions:
        await run_condition(model, dilemmas, condition, config, dry_run)


async def main():
    parser = argparse.ArgumentParser(
        description="DailyDilemmas contextual influence experiment"
    )
    parser.add_argument("--model", type=str, help="Model key from models.yaml")
    parser.add_argument(
        "--condition",
        type=str,
        help="Condition name (e.g., baseline, survey_toward_to_do)",
    )
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all conditions for the specified model",
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all models and conditions"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print sample prompts without API calls"
    )
    parser.add_argument(
        "--max-dilemmas", type=int, help="Limit number of dilemmas (for testing)"
    )
    args = parser.parse_args()

    config = load_experiment_config()
    dilemmas = load_dilemmas()

    if args.max_dilemmas:
        dilemmas = dilemmas[: args.max_dilemmas]
        print(f"Limited to {len(dilemmas)} dilemmas")

    all_conditions = get_conditions(config, dilemmas)

    if args.all:
        for model in config["models"]:
            await run_model(model, dilemmas, config, dry_run=args.dry_run)

    elif args.model:
        if args.condition:
            # Find the matching condition
            matching = [c for c in all_conditions if c["name"] == args.condition]
            if not matching:
                print(f"Unknown condition: {args.condition}")
                print(f"Available: {[c['name'] for c in all_conditions]}")
                sys.exit(1)
            await run_model(
                args.model, dilemmas, config, conditions=matching, dry_run=args.dry_run
            )
        elif args.all_conditions:
            await run_model(args.model, dilemmas, config, dry_run=args.dry_run)
        else:
            parser.error("Specify --condition, --all-conditions, or --all")
    else:
        parser.error("Specify --model or --all")


if __name__ == "__main__":
    asyncio.run(main())
