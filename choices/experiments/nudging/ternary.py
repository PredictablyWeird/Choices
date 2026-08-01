#!/usr/bin/env python3
"""
Ternary (K=3) direction-rotated influence audit.

The binary audit runs each item under three conditions -- no influence, a cue
toward A, a cue toward B. With three options the pair becomes a *rotation*:
no influence, plus one cue aimed at each of the three groups.

This module is a standalone runner. It deliberately does NOT go through
``Experiment``/``compute_utilities``/``PreferenceGraph``, all of which assume
pairwise edges. It reuses the shared pieces that are already K-agnostic
(``create_agent``, ``generate_responses``) and emits one tidy CSV row per
trial, which is what the K=3 estimands actually need.

Design (per condition):
    size-multisets drawn from sizes 1-5   (default: all 35 of them)
      x all distinct size->group assignments  (6 when sizes are distinct,
                                               fewer when tied; 125 ordered
                                               assignments over the full grid)
      x 6 display permutations             (full counterbalancing of A/B/C)
      x reps

    Defaults (all 35 multisets, 2 reps) give 1500 trials per condition --
    a full factorial over ordered size assignments from 1-5.

Usage:
    uv run python -m choices.experiments.nudging.ternary --list-factors
    uv run python -m choices.experiments.nudging.ternary \\
        --factor age3 --model gpt-5-2-non-reasoning --influence weak_evidence --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from choices.experiments.nudging.ternary_factors import (
    CHOICE_LETTERS,
    DEFAULT_INFLUENCE_TYPES,
    DEFAULT_SYSTEM_PROMPT,
    N_VALUES_DEFAULT,
    SETUPS,
    TERNARY_FACTORS,
    TernaryFactor,
    build_influence_text,
    influence_defaults,
    list_factors,
    option_text,
)
from choices.experiments.simple_rates import _get_config_path
from choices.utils import create_agent, generate_responses, load_config
from choices.utils import model_has_active_reasoning
from choices.variable import ReasoningMode


DEFAULT_OUT_DIR = "results_ternary"


# ============================================================================
# Explicit (calibrated) trial design
# ============================================================================


def build_trials_explicit(
    factor: TernaryFactor,
    assignments: List[Dict[str, int]],
    reps: int,
) -> List[Dict[str, Any]]:
    """
    Build trials from explicit, pre-chosen per-group size assignments.

    This is the calibrated-sampling counterpart to ``build_trials``: each
    assignment already specifies exactly which size goes to which group (e.g.
    picked by ``choices.analysis.ternary_calibration.neutral_triples`` to
    offset a known intrinsic preference), so it is used as-is rather than
    being rotated across groups the way generic size-triples are. Rotating a
    calibrated assignment would reassign the compensating sizes to the wrong
    groups and undo the calibration.

    Display order is still fully counterbalanced (all 6 permutations), same
    as ``build_trials``.
    """
    groups = list(factor.values)
    trials: List[Dict[str, Any]] = []

    for triple_id, assignment in enumerate(assignments):
        for perm_id, order in enumerate(itertools.permutations(groups)):
            for rep in range(reps):
                trials.append(
                    {
                        "triple_id": triple_id,
                        "size_perm_id": 0,
                        "perm_id": perm_id,
                        "rep": rep,
                        "assignment": dict(assignment),
                        "display": list(order),
                    }
                )

    return trials


# ============================================================================
# Trial design
# ============================================================================


def sample_size_triples(
    n_triples: Optional[int],
    n_values: List[int],
    seed: int,
    equal_sizes: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    Sample size-multisets for the design.

    Args:
        n_triples: How many distinct size-multisets to draw; None means all.
        n_values: Pool of group sizes.
        seed: RNG seed (design is fully deterministic given the seed).
        equal_sizes: If True, use only all-equal triples (n, n, n). This drops
            the size trade-off entirely so the baseline is a pure three-way
            group preference -- a cheap, clean sanity arm.

    Returns:
        List of sorted (s0, s1, s2) tuples.
    """
    if equal_sizes:
        pool = [(n, n, n) for n in n_values]
    else:
        pool = sorted(itertools.combinations_with_replacement(sorted(n_values), 3))

    if n_triples is None or n_triples >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n_triples))


def build_trials(
    factor: TernaryFactor,
    size_triples: List[Tuple[int, int, int]],
    reps: int,
) -> List[Dict[str, Any]]:
    """
    Expand size-multisets into fully counterbalanced trials.

    Each multiset is assigned to groups in every distinct order (6 ways for
    all-distinct sizes, fewer when tied), so over the full multiset pool this
    is a full factorial over ordered size assignments. Each trial records
    which group holds which size (``assignment``) and which group is displayed
    at which letter (``display``).
    """
    groups = list(factor.values)
    trials: List[Dict[str, Any]] = []

    for triple_id, sizes in enumerate(size_triples):
        # All distinct assignments of the sizes onto groups (set() drops
        # duplicate orderings arising from tied sizes).
        for size_perm_id, sized in enumerate(
            sorted(set(itertools.permutations(sizes)))
        ):
            assignment = dict(zip(groups, sized))

            # All 6 display orders of the three groups onto letters A/B/C.
            for perm_id, order in enumerate(itertools.permutations(groups)):
                for rep in range(reps):
                    trials.append(
                        {
                            "triple_id": triple_id,
                            "size_perm_id": size_perm_id,
                            "perm_id": perm_id,
                            "rep": rep,
                            "assignment": assignment,
                            "display": list(order),
                        }
                    )

    return trials


# ============================================================================
# Prompt assembly
# ============================================================================


def build_instructions(reasoning_mode: ReasoningMode) -> str:
    """Response-format instructions for a three-way forced choice."""
    if reasoning_mode == ReasoningMode.BEFORE:
        return (
            "Take your time to reason through the question, and then provide "
            "your final answer in the format:\n\n"
            '"Answer: A"\n\nor\n\n"Answer: B"\n\nor\n\n"Answer: C".'
        )
    if reasoning_mode == ReasoningMode.AFTER:
        return (
            "Provide your answer in the format below and then also provide "
            "your reasoning for choosing your answer:\n\n"
            '"Answer: A"\n\nor\n\n"Answer: B"\n\nor\n\n"Answer: C".'
        )
    return 'Please respond with only "A", "B", or "C".'


def format_influence(text: Optional[str], brackets: str) -> Optional[str]:
    """Wrap influence text in the configured bracket style."""
    if not text:
        return None
    if brackets == "parentheses":
        return f"({text})"
    if brackets == "quotes":
        return f'"{text}"'
    if brackets == "italic":
        return f"*{text}*"
    return text


def build_few_shot_ending(
    factor: TernaryFactor,
    target: str,
    n_values: List[int],
    num_examples: int,
    seed: int,
) -> str:
    """
    Build few-shot examples in which the target group is chosen *despite*
    having the fewest people -- the K=3 analogue of the binary few_shot cue.
    """
    rng = random.Random(seed)
    others = [v for v in factor.values if v != target]
    blocks = ["Here are some examples of previous decisions:"]

    for _ in range(num_examples):
        target_n = rng.choice([n for n in n_values if n <= max(n_values) - 2])
        other_ns = [rng.choice([n for n in n_values if n > target_n]) for _ in others]

        assignment = {target: target_n}
        for g, n in zip(others, other_ns):
            assignment[g] = n

        order = list(factor.values)
        rng.shuffle(order)

        lines = []
        answer = None
        for letter, group in zip(CHOICE_LETTERS, order):
            lines.append(
                f"Option {letter}:\n{option_text(factor, group, assignment[group])}"
            )
            if group == target:
                answer = letter

        blocks.append("\n\n".join(lines) + f"\n\nAnswer: {answer}")

    return "\n\n".join(blocks)


def build_prompt(
    factor: TernaryFactor,
    trial: Dict[str, Any],
    setup: str,
    reasoning_mode: ReasoningMode,
    influence_text: Optional[str],
    influence_position: str,
    influence_brackets: str,
    ending: Optional[str] = None,
) -> str:
    """
    Assemble a full user prompt. Mirrors the part ordering and bracket/spacing
    conventions of ``choices.experiment.PromptConfig.template`` so that the
    ternary prompts differ from the binary ones only in option count.
    """
    nudge = format_influence(influence_text, influence_brackets)
    parts: List[str] = []

    def append_nudge(text: str) -> None:
        # parentheses attach to the previous block with a single newline;
        # everything else becomes its own block.
        if influence_brackets == "parentheses" and parts:
            parts[-1] = parts[-1] + "\n" + text
        else:
            parts.append(text)

    if nudge and influence_position == "start":
        parts.append(nudge)

    parts.append(setup)

    if nudge and influence_position == "after_setup":
        append_nudge(nudge)

    option_block = "\n\n".join(
        f"Option {letter}:\n"
        f"{option_text(factor, group, trial['assignment'][group])}"
        for letter, group in zip(CHOICE_LETTERS, trial["display"])
    )
    parts.append(option_block)

    if nudge and influence_position == "after_options":
        append_nudge(nudge)

    parts.append(build_instructions(reasoning_mode))

    if nudge and influence_position == "end":
        append_nudge(nudge)

    if ending:
        parts.append(ending)

    return "\n\n".join(parts)


# ============================================================================
# K-ary response parsing
# ============================================================================

_ANSWER_PATTERN = re.compile(rf"Answer:\s*({'|'.join(CHOICE_LETTERS)})", re.IGNORECASE)
_LETTER_PATTERNS = {
    letter: re.compile(rf"(?:^|[^\w])({re.escape(letter)})(?:[^\w]|$)")
    for letter in CHOICE_LETTERS
}


def parse_choice(
    content: Optional[str],
    reasoning_mode: ReasoningMode,
) -> Tuple[Optional[str], str]:
    """
    Parse a single response into a letter.

    ``choices.utils.parse_responses_forced_choice`` asserts exactly two
    choices, so K=3 needs its own parser. This one also distinguishes *why* a
    response was invalid, because the invalid-rate breakdown is a fair
    question to ask of a three-option design.

    Returns:
        (letter or None, status) where status is one of
        "ok", "empty", "no_choice", "multiple_choices".
    """
    if content is None or not content.strip():
        return None, "empty"

    if reasoning_mode in (ReasoningMode.BEFORE, ReasoningMode.AFTER):
        match = _ANSWER_PATTERN.search(content)
        if match:
            return match.group(1).upper(), "ok"
        return None, "no_choice"

    matched = [
        letter
        for letter, pattern in _LETTER_PATTERNS.items()
        if pattern.search(content)
    ]
    if len(matched) == 1:
        return matched[0], "ok"
    if not matched:
        return None, "no_choice"
    return None, "multiple_choices"


# ============================================================================
# Runner
# ============================================================================


async def run_condition(
    factor: TernaryFactor,
    model: str,
    influence_type: str,
    direction: Optional[str],
    trials: List[Dict[str, Any]],
    setup: str,
    reasoning_mode: ReasoningMode,
    out_dir: str,
    run_id: str,
    seed: int,
    few_shot_examples: int,
    max_retries: int,
    dry_run: bool,
    verbose: bool = True,
) -> Optional[str]:
    """
    Run one condition (base, or a cue toward one group) and write trials.csv.

    Returns the path to the written CSV, or None on a dry run.
    """
    is_base = direction is None
    condition_name = "base" if is_base else direction

    # ---- influence text -------------------------------------------------
    influence_text: Optional[str] = None
    ending: Optional[str] = None
    position, brackets = influence_defaults(influence_type)
    system_prompt = DEFAULT_SYSTEM_PROMPT

    if not is_base:
        if influence_type.startswith("few_shot"):
            ending = build_few_shot_ending(
                factor, direction, N_VALUES_DEFAULT, few_shot_examples, seed
            )
        else:
            influence_text = build_influence_text(factor, influence_type, direction)
            if influence_text is None:
                raise ValueError(
                    f"Influence type '{influence_type}' produced no text; "
                    "it is not usable as a directed cue."
                )
            if position == "system_replace":
                # role_play: the cue replaces the system prompt entirely.
                system_prompt = format_influence(influence_text, brackets)
                influence_text = None
            elif position == "system":
                system_prompt = (
                    f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                    f"{format_influence(influence_text, brackets)}"
                )
                influence_text = None

    # ---- prompts --------------------------------------------------------
    prompts = [
        build_prompt(
            factor=factor,
            trial=trial,
            setup=setup,
            reasoning_mode=reasoning_mode,
            influence_text=influence_text,
            influence_position=position,
            influence_brackets=brackets,
            ending=ending,
        )
        for trial in trials
    ]

    save_path = os.path.join(
        out_dir, factor.name, model, influence_type, condition_name, run_id
    )
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "example_prompt.txt"), "w") as f:
        f.write(f"System message:\n{system_prompt}\n\n")
        f.write("=" * 70 + "\n\n")
        f.write(prompts[0])
        f.write("\n\n" + "=" * 70 + "\n")
        f.write(f"factor:      {factor.name} ({' / '.join(factor.values)})\n")
        f.write(f"influence:   {influence_type}\n")
        f.write(f"direction:   {condition_name}\n")
        f.write(f"position:    {position}\n")
        f.write(f"brackets:    {brackets}\n")
        f.write(f"n_trials:    {len(prompts)}\n")

    if verbose:
        print(
            f"  [{factor.name} | {influence_type} | -> {condition_name}] "
            f"{len(prompts)} prompts"
        )

    if dry_run:
        print("\n" + "-" * 70)
        print(f"DRY RUN: {factor.name} / {influence_type} / -> {condition_name}")
        print("-" * 70)
        print(f"System message: {system_prompt}\n")
        print(prompts[0])
        print("-" * 70 + "\n")
        return None

    # ---- query ----------------------------------------------------------
    uses_reasoning = reasoning_mode != ReasoningMode.NONE or model_has_active_reasoning(
        model
    )
    agent_config_key = "default_with_reasoning" if uses_reasoning else "default"
    agent_config = load_config(
        _get_config_path("create_agent.yaml"), agent_config_key, "create_agent.yaml"
    )
    agent = create_agent(model_key=model, **agent_config)

    responses = await generate_responses(
        agent=agent,
        prompts=prompts,
        system_message=system_prompt,
        K=1,
        verbose=verbose,
        reasoning_mode=reasoning_mode,
        valid_choices=CHOICE_LETTERS,
        max_retries=max_retries,
    )

    # ---- write tidy CSV -------------------------------------------------
    csv_path = os.path.join(save_path, "trials.csv")
    status_counts: Dict[str, int] = {}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "model",
                "reasoning_mode",
                "factor",
                "influence",
                "direction",
                "triple_id",
                "size_perm_id",
                "perm_id",
                "rep",
                "sizes",
                "display_order",
                "chosen_letter",
                "chosen_group",
                "status",
                "raw_response",
            ]
        )

        for idx, trial in enumerate(trials):
            resp_list = responses.get(idx) or []
            resp = resp_list[0] if resp_list else None
            content = getattr(resp, "content", resp) if resp is not None else None

            letter, status = parse_choice(content, reasoning_mode)
            status_counts[status] = status_counts.get(status, 0) + 1

            chosen_group = (
                trial["display"][CHOICE_LETTERS.index(letter)] if letter else ""
            )

            writer.writerow(
                [
                    run_id,
                    model,
                    reasoning_mode.value,
                    factor.name,
                    influence_type,
                    condition_name,
                    trial["triple_id"],
                    trial["size_perm_id"],
                    trial["perm_id"],
                    trial["rep"],
                    ";".join(f"{g}={trial['assignment'][g]}" for g in factor.values),
                    ",".join(trial["display"]),
                    letter or "",
                    chosen_group,
                    status,
                    (content or "").replace("\n", " ")[:500],
                ]
            )

    n_ok = status_counts.get("ok", 0)
    with open(os.path.join(save_path, "meta.json"), "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "reasoning_mode": reasoning_mode.value,
                "factor": factor.name,
                "groups": factor.values,
                "influence": influence_type,
                "direction": condition_name,
                "influence_text": influence_text,
                "system_prompt": system_prompt,
                "position": position,
                "brackets": brackets,
                "n_trials": len(trials),
                "n_valid": n_ok,
                "invalid_breakdown": {
                    k: v for k, v in status_counts.items() if k != "ok"
                },
                "seed": seed,
            },
            f,
            indent=2,
        )

    if verbose:
        pct = 100.0 * n_ok / len(trials) if trials else 0.0
        print(f"      valid: {n_ok}/{len(trials)} ({pct:.1f}%) -> {csv_path}")

    return csv_path


async def run_audit(
    factor_name: str,
    model: str,
    influence_types: List[str],
    n_triples: Optional[int],
    reps: int,
    reasoning: str,
    setup_key: str,
    out_dir: str,
    seed: int,
    equal_sizes: bool,
    few_shot_examples: int,
    max_retries: int,
    skip_base: bool,
    dry_run: bool,
    calibrate_from: Optional[List[str]] = None,
    calib_max_n: int = 10,
) -> None:
    """Run the full rotation: base + one cue per group, for each influence type."""
    if factor_name not in TERNARY_FACTORS:
        raise ValueError(f"Unknown factor: {factor_name}\nAvailable:\n{list_factors()}")

    factor = TERNARY_FACTORS[factor_name]
    reasoning_mode = ReasoningMode.from_value(reasoning)
    setup = SETUPS.get(setup_key, setup_key)

    if calibrate_from:
        import pandas as pd

        from choices.analysis.ternary_calibration import (
            fit_luce_model,
            neutral_triples,
            print_calibration_report,
        )

        paths = calibrate_from if isinstance(calibrate_from, list) else [calibrate_from]
        base_dfs = [pd.read_csv(p) for p in paths]
        fit = fit_luce_model(base_dfs, factor.values, per_group_beta=True)
        print_calibration_report(fit)

        calib_n_values = list(range(1, calib_max_n + 1))
        # Calibration selects the most-neutral assignments first, so "all"
        # is not meaningful here; fall back to 20 when no count was given.
        calib_n_triples = n_triples if n_triples is not None else 20
        assignments = neutral_triples(fit, calib_n_values, calib_n_triples)
        print(
            f"\n  selected {len(assignments)} calibrated size assignments "
            f"(worst max-deviation: {max(max(abs(fit.predict(a)[g] - 1/3) for g in factor.values) for a in assignments):.3f})"
        )
        trials = build_trials_explicit(factor, assignments, reps)
        n_size_settings = len(assignments)
    else:
        size_triples = sample_size_triples(
            n_triples, N_VALUES_DEFAULT, seed, equal_sizes
        )
        trials = build_trials(factor, size_triples, reps)
        n_size_settings = len(size_triples)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_conditions = len(influence_types) * 3 + (0 if skip_base else 1)
    print(f"\n{'=' * 70}")
    print(f"Ternary influence audit: {factor.name} ({' / '.join(factor.values)})")
    print(f"{'=' * 70}")
    print(f"  model:        {model}  (reasoning={reasoning_mode.value})")
    print(f"  influences:   {', '.join(influence_types)}")
    print(
        f"  size settings:{n_size_settings}  ({'calibrated' if calibrate_from else 'generic'})  reps: {reps}"
    )
    print(f"  trials/cond:  {len(trials)}")
    print(f"  conditions:   {n_conditions}")
    print(f"  total calls:  {len(trials) * n_conditions}")
    print(f"  run_id:       {run_id}")
    print(f"{'=' * 70}\n")

    # Baseline is shared across influence types, so it is run once under the
    # reserved influence name "base".
    if not skip_base:
        await run_condition(
            factor=factor,
            model=model,
            influence_type="base",
            direction=None,
            trials=trials,
            setup=setup,
            reasoning_mode=reasoning_mode,
            out_dir=out_dir,
            run_id=run_id,
            seed=seed,
            few_shot_examples=few_shot_examples,
            max_retries=max_retries,
            dry_run=dry_run,
        )

    for influence_type in influence_types:
        for direction in factor.values:
            await run_condition(
                factor=factor,
                model=model,
                influence_type=influence_type,
                direction=direction,
                trials=trials,
                setup=setup,
                reasoning_mode=reasoning_mode,
                out_dir=out_dir,
                run_id=run_id,
                seed=seed,
                few_shot_examples=few_shot_examples,
                max_retries=max_retries,
                dry_run=dry_run,
            )

    print(f"\nDone. Results under: {os.path.join(out_dir, factor.name, model)}")
    if not dry_run:
        print("\nNext:")
        print(
            f"  uv run python -m choices.analysis.analyze_ternary "
            f"--factor {factor.name} --model {model}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ternary (K=3) direction-rotated influence audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--factor", help="Ternary factor (see --list-factors)")
    # gpt-5-2-non-reasoning is the highest-backfire model in the binary paper
    # (73% of significant BBQ effects, 75% on triage), so it is the default.
    parser.add_argument("--model", default="gpt-5-2-non-reasoning", help="Model key")
    parser.add_argument(
        "--influence",
        nargs="+",
        default=DEFAULT_INFLUENCE_TYPES,
        help=f"Influence types (default: {' '.join(DEFAULT_INFLUENCE_TYPES)})",
    )
    parser.add_argument(
        "--n-triples",
        type=int,
        default=None,
        help="How many size-multisets to sample (default: all, i.e. a full "
        "factorial over ordered size assignments)",
    )
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument(
        "--reasoning", default="none", choices=["none", "before", "after"]
    )
    parser.add_argument("--setup", default="original", help="Setup key or literal text")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--equal-sizes",
        action="store_true",
        help="Use only all-equal size triples (pure group preference, no size trade-off)",
    )
    parser.add_argument("--few-shot-examples", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip the baseline condition (reuse an earlier baseline run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one example prompt per condition; make no API calls",
    )
    parser.add_argument("--list-factors", action="store_true")
    parser.add_argument(
        "--calibrate-from",
        metavar="TRIALS_CSV",
        nargs="+",
        help=(
            "Path(s) to existing base-condition trials.csv files (pass "
            "several to pool base data from more than one run, which widens "
            "the size range the fit is calibrated over). Fits a per-group "
            "Luce model to the observed choices and picks --n-triples "
            "explicit size assignments that make the *predicted* baseline "
            "close to uniform, instead of the generic balanced grid. Use "
            "this after a pilot run showed a skewed baseline, to get a "
            "clean asymmetry/backfire read on the re-run."
        ),
    )
    parser.add_argument(
        "--calib-max-n",
        type=int,
        default=10,
        help=(
            "Upper bound of the group-size search grid used by "
            "--calibrate-from (default 10, matching the paper's grid). "
            "Raise this for factors whose intrinsic bias is too strong to "
            "neutralize within 1-10 (e.g. an exchange rate requiring more "
            "than 10x as many of one group as another)."
        ),
    )

    args = parser.parse_args()

    if args.list_factors:
        print("Available ternary factors:\n")
        print(list_factors())
        return

    if not args.factor:
        parser.error("--factor is required (or use --list-factors)")

    asyncio.run(
        run_audit(
            factor_name=args.factor,
            model=args.model,
            influence_types=args.influence,
            n_triples=args.n_triples,
            reps=args.reps,
            reasoning=args.reasoning,
            setup_key=args.setup,
            out_dir=args.out,
            seed=args.seed,
            equal_sizes=args.equal_sizes,
            few_shot_examples=args.few_shot_examples,
            max_retries=args.max_retries,
            skip_base=args.skip_base,
            dry_run=args.dry_run,
            calibrate_from=args.calibrate_from,
            calib_max_n=args.calib_max_n,
        )
    )


if __name__ == "__main__":
    main()
