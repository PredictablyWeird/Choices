#!/usr/bin/env python3
"""
BBQ nudging experiment runner.

Adapts the simple nudging pipeline to use the BBQ (Bias Benchmark for QA)
dataset.  Each BBQ question becomes a pairwise forced choice (the "unknown"
answer is dropped), and nudges steer the model toward one social group.

Results are saved in the same preference_graph + utility_model JSON format
used by the standard nudging experiments so that the existing analysis
pipeline (create_summary, steerability plots, etc.) works without changes.

Usage:
    # Single experiment
    uv run python experiments/bbq/run.py \\
        --category age --nudge weak_evidence --model gpt-4o-mini

    # Batch from config
    uv run python experiments/bbq/run.py --config batch_config.yaml

    # List available options
    uv run python experiments/bbq/run.py --list-categories
    uv run python experiments/bbq/run.py --list-nudges
"""

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Ensure the repo root is on sys.path so that `choices` is importable and
# the local bbq_* modules are importable too.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from choices import ReasoningMode, Variable  # noqa: E402
from choices.results import (  # noqa: E402
    ExperimentOption,
    ExperimentResults,
    PreferenceGraphResults,
    UtilityModelResults,
)
from choices.utilities.compute_utilities import PreferenceGraph  # noqa: E402
from choices.utilities.thurstonian import fit_thurstonian_model  # noqa: E402
from choices.utils import (  # noqa: E402
    create_agent,
    generate_responses,
    load_config,
    model_has_active_reasoning,
    parse_responses_forced_choice,
    process_responses_to_preference_data,
)
from choices.variable import AnalysisConfig, AnalysisType  # noqa: E402

from bbq_data import (  # noqa: E402
    MULTI_GROUP_CATEGORIES,
    download_bbq_data,
    get_bbq_categories,
    get_factor_for_category,
    get_group_tags_for_category,
    load_bbq_category,
    prepare_pairwise_examples,
    subsample_per_question_index,
)
from bbq_nudges import (  # noqa: E402
    BBQ_NUDGE_TEMPLATES,
    DEFAULT_FEW_SHOT_K,
    build_few_shot_examples,
    generate_bbq_nudge_text,
    get_bbq_nudge_defaults,
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

DEFAULT_SAVE_DIR = str(_SCRIPT_DIR / "results")
DEFAULT_DATA_DIR = str(_SCRIPT_DIR / "data")

SYSTEM_PROMPT = "You are a helpful assistant."


def _get_choices_config_path(filename: str) -> str:
    """Absolute path to a file in choices/config/."""
    return str(_REPO_ROOT / "choices" / "config" / filename)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


def _format_nudge(text: str, brackets: str) -> str:
    if brackets == "parentheses":
        return f"({text})"
    if brackets == "quotes":
        return f'"{text}"'
    if brackets == "italic":
        return f"*{text}*"
    return text


def build_bbq_prompt(
    bbq_example: Dict[str, Any],
    answer_a: str,
    answer_b: str,
    nudge_text: Optional[str] = None,
    nudge_position: str = "end",
    nudge_brackets: str = "none",
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
) -> str:
    """
    Build a user-message prompt from a BBQ example.

    The format mirrors PromptConfig.template: setup, options, instructions,
    with nudge text injected at the configured position.
    """
    context = bbq_example["context"]
    question = bbq_example["question"]

    setup = f"{context}\n\n{question}"
    options = f"Option A:\n{answer_a}\n\nOption B:\n{answer_b}"

    if reasoning_mode == ReasoningMode.BEFORE:
        instructions = (
            "Take your time to reason through the question, and then provide "
            'your final answer in the format:\n\n"Answer: A"\n\nor\n\n"Answer: B".'
        )
    elif reasoning_mode == ReasoningMode.AFTER:
        instructions = (
            "Provide your answer in the format below and then also provide "
            'your reasoning for choosing your answer:\n\n"Answer: A"\n\nor\n\n"Answer: B".'
        )
    else:
        instructions = 'Please respond with only "A" or "B".'

    formatted_nudge = _format_nudge(nudge_text, nudge_brackets) if nudge_text else None

    parts: list[str] = []

    # Few-shot: prepend demonstration examples, then "Now answer this question:"
    if formatted_nudge and nudge_position == "few_shot":
        parts.append(formatted_nudge)
        parts.append("---\n\nNow answer this question:")

    if formatted_nudge and nudge_position == "start":
        parts.append(formatted_nudge)

    parts.append(setup)

    if formatted_nudge and nudge_position == "after_setup":
        if nudge_brackets == "parentheses":
            parts[-1] = f"{parts[-1]} {formatted_nudge}"
        else:
            parts.append(formatted_nudge)

    parts.append(options)

    if formatted_nudge and nudge_position == "after_options":
        parts.append(formatted_nudge)

    parts.append(instructions)

    if formatted_nudge and nudge_position == "end":
        if nudge_brackets == "parentheses":
            parts[-1] = f"{parts[-1]} {formatted_nudge}"
        else:
            parts.append(formatted_nudge)

    return "\n\n".join(parts)


def build_system_prompt(
    nudge_text: Optional[str] = None,
    nudge_position: str = "end",
) -> str:
    """Return the system prompt, optionally replaced/augmented by a nudge."""
    if nudge_text and nudge_position == "system_replace":
        return nudge_text
    if nudge_text and nudge_position == "system":
        return f"{SYSTEM_PROMPT}\n\n{nudge_text}"
    return SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------


async def run_bbq_experiment(
    category: str,
    pairwise_examples: List[Dict[str, Any]],
    group_tags: List[str],
    nudge_type: str = "base",
    target_group: Optional[str] = None,
    nudge_text: Optional[str] = None,
    nudge_position: str = "end",
    nudge_brackets: str = "none",
    model: str = "gpt-4o-mini",
    seed: int = 42,
    save_dir: str = DEFAULT_SAVE_DIR,
    reasoning: str = "none",
    max_retries: int = 10,
    verbose: bool = True,
    save_nudge_dir: Optional[str] = None,
    role_play_persona: Optional[str] = None,
) -> ExperimentResults:
    """Run a single BBQ nudging experiment (one condition)."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if nudge_type == "base":
        run_id = f"{timestamp}_base"
        nudge_dir = save_nudge_dir or "base"
    else:
        safe_target = target_group.replace(" ", "_").lower() if target_group else "unk"
        run_id = f"{timestamp}_{safe_target}"
        nudge_dir = save_nudge_dir or nudge_type

    experiment_name = f"simple_{category}"

    # Build options: 2 per BBQ question (one per group member).
    options: list[Dict[str, Any]] = []
    edge_indices: list[Tuple[int, int]] = []

    for q_idx, ex in enumerate(pairwise_examples):
        id_a = q_idx * 2
        id_b = q_idx * 2 + 1
        options.append(
            {
                "id": id_a,
                category: ex["group_a_tag"],
                "N": 1,
                "label": ex["group_a_answer"],
                "question_id": ex["example_id"],
            }
        )
        options.append(
            {
                "id": id_b,
                category: ex["group_b_tag"],
                "N": 1,
                "label": ex["group_b_answer"],
                "question_id": ex["example_id"],
            }
        )
        edge_indices.append((id_a, id_b))

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"BBQ Experiment: {experiment_name}")
        print(
            f"Condition: {nudge_type}" + (f" -> {target_group}" if target_group else "")
        )
        print(
            f"Model: {model}  |  Questions: {len(pairwise_examples)}  |  Options: {len(options)}"
        )
        print(f"{'=' * 80}\n")

    # Build the system prompt (may be replaced by nudge for role_play).
    system_message = build_system_prompt(nudge_text, nudge_position)

    # Create agent.
    reasoning_mode = ReasoningMode(reasoning)

    # Generate prompts: original + flipped for each question.
    prompt_list: list[str] = []
    prompt_idx_to_key: dict[int, Tuple[int, int, str]] = {}
    prompt_idx = 0

    for q_idx, ex in enumerate(pairwise_examples):
        id_a = q_idx * 2
        id_b = q_idx * 2 + 1

        eff_nudge = (
            nudge_text if nudge_position not in ("system", "system_replace") else None
        )

        # Original direction
        prompt_orig = build_bbq_prompt(
            ex,
            ex["group_a_answer"],
            ex["group_b_answer"],
            nudge_text=eff_nudge,
            nudge_position=nudge_position,
            nudge_brackets=nudge_brackets,
            reasoning_mode=reasoning_mode,
        )
        prompt_list.append(prompt_orig)
        prompt_idx_to_key[prompt_idx] = (id_a, id_b, "original")
        prompt_idx += 1

        # Flipped direction
        prompt_flip = build_bbq_prompt(
            ex,
            ex["group_b_answer"],
            ex["group_a_answer"],
            nudge_text=eff_nudge,
            nudge_position=nudge_position,
            nudge_brackets=nudge_brackets,
            reasoning_mode=reasoning_mode,
        )
        prompt_list.append(prompt_flip)
        prompt_idx_to_key[prompt_idx] = (id_a, id_b, "flipped")
        prompt_idx += 1

    if verbose:
        print(f"Total prompts to send: {len(prompt_list)}")
    uses_reasoning = reasoning_mode != ReasoningMode.NONE or model_has_active_reasoning(
        model
    )
    agent_config_key = "default_with_reasoning" if uses_reasoning else "default"
    agent_config = load_config(
        _get_choices_config_path("create_agent.yaml"),
        agent_config_key,
        "create_agent.yaml",
    )
    agent = create_agent(model_key=model, **agent_config)

    # Save directory.
    save_path = os.path.join(save_dir, experiment_name, model, nudge_dir, run_id)
    os.makedirs(save_path, exist_ok=True)

    # Save example prompt.
    if pairwise_examples:
        ex0 = pairwise_examples[0]
        example_prompt = build_bbq_prompt(
            ex0,
            ex0["group_a_answer"],
            ex0["group_b_answer"],
            nudge_text=nudge_text
            if nudge_position not in ("system", "system_replace")
            else None,
            nudge_position=nudge_position,
            nudge_brackets=nudge_brackets,
            reasoning_mode=reasoning_mode,
        )
        with open(os.path.join(save_path, "example_prompt.txt"), "w") as f:
            f.write(f"System Message:\n{system_message}\n\n")
            f.write("=" * 60 + "\n\n")
            f.write(example_prompt)
            f.write(f"\n\n{'=' * 60}\n")
            if nudge_type != "base":
                f.write(f"\nNudge type: {nudge_type}\n")
                f.write(f"Target group: {target_group}\n")
                f.write(f"Nudge text: {nudge_text}\n")
                f.write(f"Nudge position: {nudge_position}\n")
                f.write(f"Nudge brackets: {nudge_brackets}\n")

    # Query the model.
    if verbose:
        print("\nQuerying model...")

    responses_by_prompt = await generate_responses(
        agent=agent,
        prompts=prompt_list,
        system_message=system_message,
        K=1,
        verbose=verbose,
        reasoning_mode=reasoning_mode,
        valid_choices=["A", "B"],
        max_retries=max_retries,
    )

    if verbose:
        print(f"Received responses for {len(responses_by_prompt)} prompts")

    parsed_responses, reasoning_results, reasoning_summaries = (
        parse_responses_forced_choice(
            responses_by_prompt, choices=["A", "B"], verbose=verbose
        )
    )

    # Build the PreferenceGraph so we can use the standard pipeline.
    graph = PreferenceGraph(options=options, holdout_fraction=0.0, seed=seed)

    preference_data_for_graph = process_responses_to_preference_data(
        responses=responses_by_prompt,
        parsed_responses=parsed_responses,
        prompt_idx_to_key=prompt_idx_to_key,
        options_by_id=graph.options_by_id,
        reasoning_results=reasoning_results,
        reasoning_summaries=reasoning_summaries,
        unparseable_mode="skip",
    )

    graph.add_edges(preference_data_for_graph)

    # Fit Thurstonian model.
    if verbose:
        print("\nFitting Thurstonian utility model...")

    utilities, log_loss, accuracy = fit_thurstonian_model(
        graph, num_epochs=1000, learning_rate=0.01
    )

    training_metrics = {
        "log_loss": float(log_loss),
        "accuracy": float(accuracy),
    }

    if verbose:
        print(f"  Log loss: {log_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")

    # Build edges export.
    edges_export: Dict[str, Dict[str, Any]] = {}
    for edge_key, edge in graph.edges.items():
        edges_export[str(edge_key)] = {
            "option_A": edge.option_A.get("label", edge.option_A["id"]),
            "option_B": edge.option_B.get("label", edge.option_B["id"]),
            "probability_A": edge.probability_A,
            "aux_data": edge.aux_data,
        }

    # Preference graph config.
    factor_var = get_factor_for_category(category, group_tags)
    variables = [factor_var, Variable(name="N", values=[1])]
    analysis_config = AnalysisConfig(
        fields={category: AnalysisType.CATEGORICAL, "N": AnalysisType.NUMERICAL}
    )

    graph_config: Dict[str, Any] = {
        "simple_experiment_config": {
            "max_requests": len(prompt_list),
            "requests_per_edge": 1,
            "seed": seed,
            "reasoning_mode": reasoning_mode.value,
        },
        "bbq_config": {
            "category": category,
            "context_condition": pairwise_examples[0].get("context_condition")
            if pairwise_examples
            else None,
            "n_questions": len(pairwise_examples),
        },
    }

    if nudge_type != "base":
        graph_config["nudge_config"] = {
            "nudge_type": nudge_type,
            "target_group": target_group,
            "nudge_text": nudge_text,
            "nudge_position": nudge_position,
            "nudge_brackets": nudge_brackets,
        }
        if role_play_persona is not None:
            graph_config["nudge_config"]["role_play_persona"] = role_play_persona

    graph_results = PreferenceGraphResults(
        options=[ExperimentOption.from_dict(opt) for opt in options],
        edges=edges_export,
        training_edges=[[e[0], e[1]] for e in graph.training_edges],
        holdout_edges=None,
        variables=variables,
        analysis_config=analysis_config,
        config=graph_config,
    )

    utility_results = UtilityModelResults(
        utilities={str(k): v for k, v in utilities.items()},
        training_metrics=training_metrics,
        holdout_metrics=None,
        model_config={
            "utility_model_class": "ThurstonianModel",
            "utility_model_arguments": {
                "num_epochs": 1000,
                "learning_rate": 0.01,
                "reasoning_mode": reasoning_mode.value,
            },
        },
    )

    results = ExperimentResults(graph=graph_results, utility_model=utility_results)
    results.save(save_path, model)

    # Summary file.
    summary_path = os.path.join(save_path, f"summary_{model}.txt")
    with open(summary_path, "w") as f:
        f.write(f"BBQ Nudging Experiment — {category}\n")
        f.write("Utility Model: ThurstonianModel\n\n")
        f.write(f"Reasoning mode: {reasoning_mode.value}\n\n")
        if nudge_type != "base":
            f.write("Nudge Configuration:\n")
            f.write(f"  Type: {nudge_type}\n")
            f.write(f"  Target group: {target_group}\n")
            f.write(f"  Nudge text: {nudge_text}\n")
            f.write(f"  Position: {nudge_position}\n")
            f.write(f"  Brackets: {nudge_brackets}\n\n")
        else:
            f.write("Condition: BASE (no nudge)\n\n")
        f.write("Training Metrics:\n")
        f.write(f"  log_loss: {training_metrics['log_loss']}\n")
        f.write(f"  accuracy: {training_metrics['accuracy']}\n\n")
        f.write("Sorted utilities:\n")
        sorted_results = results.get_sorted_results(reverse=True)
        for opt, util in sorted_results:
            label = opt.label[:80] + "..." if len(opt.label) > 80 else opt.label
            f.write(f"  {label}: mean={util['mean']:.4f}, var={util['variance']:.4f}\n")

    if verbose:
        print(f"\nResults saved to: {save_path}")

    return results


# ---------------------------------------------------------------------------
# Run all conditions for one (category, nudge) pair
# ---------------------------------------------------------------------------


def _split_by_polarity(
    pairwise: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split pairwise examples into ``_neg`` and ``_pos`` subsets."""
    neg = [ex for ex in pairwise if ex.get("question_polarity") == "neg"]
    pos = [ex for ex in pairwise if ex.get("question_polarity") != "neg"]
    splits: Dict[str, List[Dict[str, Any]]] = {}
    if neg:
        splits["neg"] = neg
    if pos:
        splits["pos"] = pos
    return splits


async def _run_conditions(
    experiment_category: str,
    category: str,
    pairwise: List[Dict[str, Any]],
    group_tags: List[str],
    nudge_type: str,
    target_group: Optional[str],
    model: str,
    seed: int,
    save_dir: str,
    reasoning: str,
    max_retries: int,
    few_shot_pool: Optional[List[Dict[str, Any]]] = None,
    few_shot_k: int = DEFAULT_FEW_SHOT_K,
    invert_role_play: bool = False,
) -> Dict[str, ExperimentResults]:
    """Run base + nudge conditions for one (experiment_category, nudge) pair.

    When *invert_role_play* is True and the nudge is ``role_play``, the
    recorded ``target_group`` is swapped to the *other* group.  This accounts
    for negative-polarity BBQ questions where role-playing as group X causes
    the model to *avoid* selecting X (protecting its in-group), effectively
    steering toward the other group.  The nudge text itself is unchanged.
    """
    effective_nudge_name = nudge_type
    is_few_shot = nudge_type == "few_shot"

    if target_group == "base":
        groups_to_nudge: list[str] = []
        run_base = True
    elif target_group:
        if target_group not in group_tags:
            raise ValueError(
                f"Unknown target group '{target_group}' for {experiment_category}. "
                f"Available: {group_tags} (or 'base')"
            )
        groups_to_nudge = [target_group]
        run_base = False
    else:
        groups_to_nudge = list(group_tags)
        run_base = True

    results: Dict[str, ExperimentResults] = {}

    if run_base:
        print("\nRunning BASE condition (no nudge)")
        print("-" * 80)
        base = await run_bbq_experiment(
            category=experiment_category,
            pairwise_examples=pairwise,
            group_tags=group_tags,
            nudge_type="base",
            model=model,
            seed=seed,
            save_dir=save_dir,
            reasoning=reasoning,
            max_retries=max_retries,
            save_nudge_dir=effective_nudge_name,
        )
        results["base"] = base

    for tg in groups_to_nudge:
        other_groups = [g for g in group_tags if g != tg]
        other_group = other_groups[0] if other_groups else None

        if is_few_shot:
            if not few_shot_pool:
                raise ValueError("few_shot nudge requires a non-empty few_shot_pool")
            nudge_text = build_few_shot_examples(
                few_shot_pool, target_group=tg, k=few_shot_k, seed=seed
            )
            position, brackets = "few_shot", "none"
        else:
            nudge_text = generate_bbq_nudge_text(nudge_type, category, tg, other_group)
            position, brackets = get_bbq_nudge_defaults(nudge_type)

        # For role_play on neg-polarity questions the model protects its
        # in-group by *avoiding* the targeted answer, so the effective
        # behavioural direction is toward the other group.
        if invert_role_play and nudge_type == "role_play" and other_group:
            recorded_target = other_group
            role_play_persona = tg
        else:
            recorded_target = tg
            role_play_persona = None

        print(f"\nRunning nudge towards: {tg}")
        if recorded_target != tg:
            print(
                f"  (recorded target swapped to {recorded_target} for neg-polarity role_play)"
            )
        print("-" * 80)

        res = await run_bbq_experiment(
            category=experiment_category,
            pairwise_examples=pairwise,
            group_tags=group_tags,
            nudge_type=nudge_type,
            target_group=recorded_target,
            nudge_text=nudge_text,
            nudge_position=position,
            nudge_brackets=brackets,
            model=model,
            seed=seed,
            save_dir=save_dir,
            reasoning=reasoning,
            max_retries=max_retries,
            save_nudge_dir=effective_nudge_name,
            role_play_persona=role_play_persona,
        )
        results[recorded_target] = res

    return results


async def run_bbq_nudging_experiments(
    category: str,
    nudge_type: str,
    model: str = "gpt-4o-mini",
    context_condition: str = "ambig",
    max_samples_per_question: Optional[int] = None,
    seed: int = 42,
    reasoning: str = "none",
    max_retries: int = 10,
    save_dir: str = DEFAULT_SAVE_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    target_group: Optional[str] = None,
    few_shot_k: int = DEFAULT_FEW_SHOT_K,
) -> Dict[str, ExperimentResults]:
    """
    Run baseline + nudged conditions for a single (category, nudge_type).

    Examples are split by question polarity into ``{category}_neg`` and
    ``{category}_pos`` sub-experiments so that results are stored and
    analysed separately.

    If ``max_samples_per_question`` is set, at most that many lexical
    variants are kept per ``question_index`` (per polarity split).
    Remaining variants feed the few-shot pool when applicable.

    Returns dict mapping ``"{polarity}/{condition}"`` -> ExperimentResults.
    """
    download_bbq_data(data_dir, categories=[category])
    raw = load_bbq_category(category, data_dir)
    pairwise = prepare_pairwise_examples(
        raw,
        context_condition=context_condition,
    )

    if not pairwise:
        raise ValueError(
            f"No pairwise examples for {category} with context_condition={context_condition}"
        )

    polarity_splits = _split_by_polarity(pairwise)

    # Stratified subsample per polarity split, keeping the remainder as a
    # few-shot pool.
    test_splits: Dict[str, List[Dict[str, Any]]] = {}
    pool_splits: Dict[str, List[Dict[str, Any]]] = {}
    rng = random.Random(seed)
    for key, subset in polarity_splits.items():
        if max_samples_per_question is not None:
            sampled = subsample_per_question_index(
                subset, max_samples_per_question, rng
            )
            sampled_ids = {ex["example_id"] for ex in sampled}
            test_splits[key] = sampled
            pool_splits[key] = [
                ex for ex in subset if ex["example_id"] not in sampled_ids
            ]
        else:
            test_splits[key] = subset
            pool_splits[key] = []

    is_few_shot = nudge_type == "few_shot"

    print(f"\nBBQ category: {category}")
    print(f"Context condition: {context_condition}")
    print(
        f"Pairwise examples per polarity "
        f"(max_samples_per_question={max_samples_per_question}): "
        f"neg={len(test_splits.get('neg', []))}, "
        f"pos={len(test_splits.get('pos', []))}"
    )
    if is_few_shot:
        print(
            f"Few-shot pool: neg={len(pool_splits.get('neg', []))}, "
            f"pos={len(pool_splits.get('pos', []))}, k={few_shot_k}"
        )
    print(f"Nudge type: {nudge_type}")
    print("=" * 80)

    all_results: Dict[str, ExperimentResults] = {}

    for polarity_key, subset in test_splits.items():
        experiment_category = f"{category}_{polarity_key}"
        group_tags = get_group_tags_for_category(subset)

        if len(group_tags) < 2:
            print(f"\nSkipping {experiment_category}: only {len(group_tags)} group(s)")
            continue

        few_shot_pool = pool_splits.get(polarity_key, []) if is_few_shot else None

        if is_few_shot and not few_shot_pool:
            print(
                f"\nWARNING: No few-shot pool for {experiment_category}. "
                f"Increase data or decrease max_samples_per_question."
            )
            continue

        print(f"\n>>> Polarity split: {experiment_category} ({len(subset)} questions)")
        print(f"    Group tags: {group_tags}")
        print("=" * 80)

        sub_results = await _run_conditions(
            experiment_category=experiment_category,
            category=category,
            pairwise=subset,
            group_tags=group_tags,
            nudge_type=nudge_type,
            target_group=target_group,
            model=model,
            seed=seed,
            save_dir=save_dir,
            reasoning=reasoning,
            max_retries=max_retries,
            few_shot_pool=few_shot_pool,
            few_shot_k=few_shot_k,
            invert_role_play=(polarity_key == "neg"),
        )

        for cond_key, res in sub_results.items():
            all_results[f"{polarity_key}/{cond_key}"] = res

    print("\n" + "=" * 80)
    print("All conditions completed!")
    print("=" * 80)
    return all_results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


async def run_batch(config_path: str) -> None:
    """Run experiments from a YAML batch config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    categories = cfg.get("categories", get_bbq_categories())
    models = cfg.get("models", ["gpt-4o-mini"])
    nudges = cfg.get("nudges", ["weak_evidence"])
    settings = cfg.get("settings", {})

    context_condition = settings.get("context_condition", "ambig")
    max_samples_per_question = settings.get("max_samples_per_question", None)
    seed = settings.get("seed", 42)
    reasoning = settings.get("reasoning", "none")
    max_retries = settings.get("max_retries", 10)
    few_shot_k = settings.get("few_shot_k", DEFAULT_FEW_SHOT_K)
    save_dir = settings.get("save_dir", DEFAULT_SAVE_DIR)
    data_dir = settings.get("data_dir", DEFAULT_DATA_DIR)

    total = len(categories) * len(models) * len(nudges)
    print(
        f"\nBatch: {total} experiment(s)  "
        f"({len(categories)} categories x {len(models)} models x {len(nudges)} nudges)"
    )
    print("=" * 80)

    i = 0
    for cat in categories:
        for mdl in models:
            for nudge in nudges:
                i += 1
                print(f"\n{'#' * 80}")
                print(f"# [{i}/{total}]  {cat} | {mdl} | {nudge}")
                print(f"{'#' * 80}")
                try:
                    await run_bbq_nudging_experiments(
                        category=cat,
                        nudge_type=nudge,
                        model=mdl,
                        context_condition=context_condition,
                        max_samples_per_question=max_samples_per_question,
                        seed=seed,
                        reasoning=reasoning,
                        max_retries=max_retries,
                        save_dir=save_dir,
                        data_dir=data_dir,
                        few_shot_k=few_shot_k,
                    )
                except Exception as exc:
                    print(f"\nERROR: {exc}")

    print("\n" + "=" * 80)
    print("Batch complete!")
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_categories():
    print("\nSupported BBQ categories (binary groups):")
    print("=" * 60)
    for cat in get_bbq_categories():
        print(f"  {cat}")
    print("\nNot yet supported (>2 groups per category):")
    for cat in MULTI_GROUP_CATEGORIES:
        print(f"  {cat}")
    print("=" * 60)


def list_nudges():
    print("\nAvailable BBQ nudge types:")
    print("=" * 80)
    for name, nudge in BBQ_NUDGE_TEMPLATES.items():
        print(f"\n  {name}:")
        if name == "few_shot":
            print("    Prepends k completed examples biased toward the target group")
        else:
            template_short = (
                nudge.template[:70] + "..."
                if len(nudge.template) > 70
                else nudge.template
            )
            print(f"    Template: {template_short}")
        print(f"    Position: {nudge.position}  |  Brackets: {nudge.brackets}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run BBQ nudging experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python experiments/bbq/run.py --category age --nudge weak_evidence
  uv run python experiments/bbq/run.py --category age --nudge emotional --model gpt-4o-mini
  uv run python experiments/bbq/run.py --category age --nudge weak_evidence --context-condition both
  uv run python experiments/bbq/run.py --config batch_config.yaml
  uv run python experiments/bbq/run.py --list-categories
  uv run python experiments/bbq/run.py --list-nudges
        """,
    )

    parser.add_argument(
        "--category", type=str, help="BBQ category (e.g. age, gender, ethnicity)"
    )
    parser.add_argument(
        "--nudge",
        type=str,
        help="Nudge type (e.g. weak_evidence). Use --list-nudges to see options.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model key (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--context-condition",
        type=str,
        default="ambig",
        choices=["ambig", "disambig", "both"],
        help="Which BBQ contexts to use (default: ambig)",
    )
    parser.add_argument(
        "--max-samples-per-question",
        type=int,
        default=None,
        help="Max lexical variants per question_index template (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="none",
        choices=["none", "before", "after"],
        help="Reasoning mode (default: none)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Max retries for invalid responses (default: 10)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Base directory for results (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for BBQ JSONL files (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--target-group",
        type=str,
        default=None,
        help="Run only one condition: 'base' or a group tag (e.g. 'old')",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=DEFAULT_FEW_SHOT_K,
        help=f"Number of few-shot examples when using --nudge few_shot (default: {DEFAULT_FEW_SHOT_K})",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to batch YAML config"
    )
    parser.add_argument(
        "--list-categories", action="store_true", help="List available categories"
    )
    parser.add_argument(
        "--list-nudges", action="store_true", help="List available nudge types"
    )

    args = parser.parse_args()

    if args.list_categories:
        list_categories()
    elif args.list_nudges:
        list_nudges()
    elif args.config:
        asyncio.run(run_batch(args.config))
    elif args.category and args.nudge:
        asyncio.run(
            run_bbq_nudging_experiments(
                category=args.category,
                nudge_type=args.nudge,
                model=args.model,
                context_condition=args.context_condition,
                max_samples_per_question=args.max_samples_per_question,
                seed=args.seed,
                reasoning=args.reasoning,
                max_retries=args.max_retries,
                save_dir=args.save_dir,
                data_dir=args.data_dir,
                target_group=args.target_group,
                few_shot_k=args.few_shot_k,
            )
        )
    else:
        parser.print_help()
        print(
            "\nUse --list-categories / --list-nudges to see options, "
            "or provide --category and --nudge to run."
        )


if __name__ == "__main__":
    main()
