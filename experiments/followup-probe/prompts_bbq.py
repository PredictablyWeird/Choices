"""
Reconstruct BBQ prompt configurations from saved data.

Mirrors what `experiments/bbq/run.py` does at trial time, but for
graphs that already exist on disk. Uses the BBQ JSONL as the source of truth
for context + question text (since the saved option only stores the answer
label and `question_id`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_BBQ_RUNNER_DIR = Path(__file__).resolve().parents[1] / "2026-04-14-bbq"
if str(_BBQ_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(_BBQ_RUNNER_DIR))

from bbq_data import (  # noqa: E402
    load_bbq_category,
    prepare_pairwise_examples,
)
from bbq_nudges import (  # noqa: E402
    DEFAULT_FEW_SHOT_K,
    build_few_shot_examples,
    generate_bbq_nudge_text,
    get_bbq_nudge_defaults,
)
from run import build_bbq_prompt, build_system_prompt  # noqa: E402

from choices.variable import ReasoningMode  # noqa: E402


# Maps BBQ factor name in saved layout → (BBQ category, polarity)
FACTOR_TO_CATEGORY: dict[str, tuple[str, str]] = {
    "age_neg": ("age", "neg"),
    "age_pos": ("age", "pos"),
    "ses_neg": ("ses", "neg"),
    "ses_pos": ("ses", "pos"),
}


_BBQ_DATA_DIR = _BBQ_RUNNER_DIR / "data"
_examples_cache: dict[tuple[str, str], dict[int, dict]] = {}
_few_shot_pool_cache: dict[tuple[str, str], list[dict]] = {}


def _load_examples_for_factor(factor_name: str) -> tuple[dict[int, dict], list[dict]]:
    """Return (examples_by_id, full_pairwise_list_for_factor)."""
    if factor_name not in FACTOR_TO_CATEGORY:
        raise ValueError(f"Unknown BBQ factor: {factor_name}")
    category, polarity = FACTOR_TO_CATEGORY[factor_name]
    cache_key = (category, polarity)
    if cache_key in _examples_cache:
        return _examples_cache[cache_key], _few_shot_pool_cache[cache_key]

    raw = load_bbq_category(category, str(_BBQ_DATA_DIR))
    pairwise = prepare_pairwise_examples(raw, context_condition="ambig")
    if polarity == "neg":
        pairwise_polarity = [
            ex for ex in pairwise if ex.get("question_polarity") == "neg"
        ]
    else:
        pairwise_polarity = [
            ex for ex in pairwise if ex.get("question_polarity") != "neg"
        ]
    by_id = {ex["example_id"]: ex for ex in pairwise_polarity}
    _examples_cache[cache_key] = by_id
    _few_shot_pool_cache[cache_key] = pairwise_polarity
    return by_id, pairwise_polarity


def get_pairwise_for_question_id(factor_name: str, question_id: int) -> dict | None:
    by_id, _ = _load_examples_for_factor(factor_name)
    return by_id.get(question_id)


def build_prompts_for_condition(
    factor_name: str,
    nudge_type: str,
    target_group: str | None,
    reasoning: str = "none",
    seed: int = 42,
    few_shot_k: int = DEFAULT_FEW_SHOT_K,
):
    """
    Returns a callable `make_prompt(option_a_dict, option_b_dict) -> str` and
    the system prompt string. The callable looks up the BBQ example via
    option_a's `question_id` (== option_b's question_id), then renders the
    prompt with appropriate nudge text.
    """
    if factor_name not in FACTOR_TO_CATEGORY:
        raise ValueError(f"Unknown BBQ factor: {factor_name}")
    category, _polarity = FACTOR_TO_CATEGORY[factor_name]

    is_few_shot = nudge_type == "few_shot"

    if target_group in (None, "base"):
        nudge_text = None
        position = "end"
        brackets = "none"
    elif is_few_shot:
        # Need the few-shot pool for this category+polarity
        _, pool = _load_examples_for_factor(factor_name)
        nudge_text = build_few_shot_examples(
            pool, target_group=target_group, k=few_shot_k, seed=seed
        )
        position, brackets = "few_shot", "none"
    else:
        # Discover other_group from the saved factor's two values
        # (we don't have group_tags here, but the templates use {other_group_label};
        # the runner determines it from group_tags). We compute it from the BBQ
        # examples themselves.
        _, pool = _load_examples_for_factor(factor_name)
        all_tags: set[str] = set()
        for ex in pool:
            all_tags.add(ex["group_a_tag"])
            all_tags.add(ex["group_b_tag"])
        others = [t for t in all_tags if t != target_group]
        other_group = others[0] if others else None
        nudge_text = generate_bbq_nudge_text(
            nudge_type, category, target_group, other_group
        )
        position, brackets = get_bbq_nudge_defaults(nudge_type)

    system_prompt = build_system_prompt(nudge_text, position)
    reasoning_mode = ReasoningMode(reasoning)

    by_id, _ = _load_examples_for_factor(factor_name)

    eff_nudge = nudge_text if position not in ("system", "system_replace") else None

    def make_prompt(option_a: dict, option_b: dict) -> str:
        # Both options are from the same question; take A's question_id
        qid = option_a.get("question_id")
        ex = by_id.get(qid)
        if ex is None:
            raise KeyError(
                f"BBQ example_id={qid} not in loaded {factor_name} pool "
                f"(size={len(by_id)})"
            )
        return build_bbq_prompt(
            ex,
            option_a["label"],
            option_b["label"],
            nudge_text=eff_nudge,
            nudge_position=position,
            nudge_brackets=brackets,
            reasoning_mode=reasoning_mode,
        )

    return make_prompt, system_prompt
