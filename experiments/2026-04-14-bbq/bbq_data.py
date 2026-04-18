"""
BBQ dataset loading, parsing, and pairwise example preparation.

Downloads BBQ (Bias Benchmark for QA) JSONL files from the official GitHub
repository and converts them into pairwise forced-choice examples suitable
for the nudging experiment framework.

Reference: Parrish et al., "BBQ: A Hand-Built Bias Benchmark for Question
Answering", Findings of ACL 2022.  https://arxiv.org/abs/2110.08193
"""

import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from choices.variable import Variable

BBQ_GITHUB_RAW = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"

# Short lowercase category names -> upstream BBQ JSONL filenames (without .jsonl).
CATEGORY_TO_BBQ_FILE: Dict[str, str] = {
    "age": "Age",
    "disability": "Disability_status",
    "gender": "Gender_identity",
    "nationality": "Nationality",
    "appearance": "Physical_appearance",
    "ethnicity": "Race_ethnicity",
    "religion": "Religion",
    "ses": "SES",
    "sexuality": "Sexual_orientation",
    # Intersectional
    "race_x_ses": "Race_x_SES",
    "race_x_gender": "Race_x_gender",
}

# Categories with exactly two groups, suitable for binary nudging experiments.
PRIMARY_CATEGORIES = [
    "age",
    "disability",
    "ses",
]

# Categories with >2 groups in the actual data.  Nudging is ambiguous when
# the target group may not appear in every question, so these are excluded
# for now.  Future work could filter questions per nudge target or generate
# per-question nudge text.
MULTI_GROUP_CATEGORIES = [
    "gender",
    "nationality",
    "appearance",
    "ethnicity",
    "religion",
    "sexuality",
]

INTERSECTIONAL_CATEGORIES = [
    "race_x_ses",
    "race_x_gender",
]

ALL_CATEGORIES = PRIMARY_CATEGORIES + MULTI_GROUP_CATEGORIES + INTERSECTIONAL_CATEGORIES

# Human-readable group labels used for nudge text and display.
# Keyed by the short category name.
GROUP_LABELS: Dict[str, Dict[str, str]] = {
    "age": {"old": "elderly people", "nonOld": "young people"},
    "disability": {
        "disabled": "disabled people",
        "nonDisabled": "non-disabled people",
    },
    "ses": {"lowSES": "low-income people", "highSES": "high-income people"},
}


def _bbq_file(category: str) -> str:
    """Resolve a short category name to the upstream BBQ filename stem."""
    bbq = CATEGORY_TO_BBQ_FILE.get(category)
    if bbq is None:
        raise ValueError(
            f"Unknown category: {category}. "
            f"Available: {list(CATEGORY_TO_BBQ_FILE.keys())}"
        )
    return bbq


def get_bbq_categories() -> List[str]:
    """Return the supported BBQ categories (binary-group only)."""
    return list(PRIMARY_CATEGORIES)


def download_bbq_data(
    data_dir: str,
    categories: Optional[List[str]] = None,
    verbose: bool = True,
) -> None:
    """Download BBQ JSONL files from GitHub if not already present."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = ALL_CATEGORIES

    for category in categories:
        bbq_name = _bbq_file(category)
        dest = data_path / f"{bbq_name}.jsonl"
        if dest.exists():
            if verbose:
                print(f"  {bbq_name}.jsonl already exists, skipping")
            continue
        url = f"{BBQ_GITHUB_RAW}/{bbq_name}.jsonl"
        if verbose:
            print(f"  Downloading {bbq_name}.jsonl ...")
        urllib.request.urlretrieve(url, dest)


def load_bbq_category(category: str, data_dir: str) -> List[Dict[str, Any]]:
    """Load all examples from a single BBQ category JSONL file."""
    bbq_name = _bbq_file(category)
    path = Path(data_dir) / f"{bbq_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"BBQ data file not found: {path}. Run download_bbq_data() first."
        )
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _extract_pairwise(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single BBQ example into a pairwise forced-choice record.

    Identifies the two non-"unknown" answers and their group tags, dropping
    the "unknown" option. Returns None if the example cannot be converted
    (e.g. if group tags are missing or there aren't exactly 2 non-unknown answers).
    """
    answer_info = example.get("answer_info", {})

    non_unknown = []
    for ans_key in ("ans0", "ans1", "ans2"):
        info = answer_info.get(ans_key, [])
        if len(info) < 2:
            continue
        _name, group_tag = info[0], info[1]
        if group_tag == "unknown":
            continue
        ans_idx = int(ans_key[-1])
        non_unknown.append(
            {
                "ans_key": ans_key,
                "ans_idx": ans_idx,
                "answer_text": example.get(ans_key, ""),
                "group_tag": group_tag,
            }
        )

    if len(non_unknown) != 2:
        return None

    a, b = non_unknown[0], non_unknown[1]
    stereotyped = example.get("additional_metadata", {}).get("stereotyped_groups", [])

    return {
        "example_id": example["example_id"],
        "question_index": example.get("question_index"),
        "question_polarity": example.get("question_polarity"),
        "context_condition": example.get("context_condition"),
        "category": example.get("category"),
        "context": example.get("context", ""),
        "question": example.get("question", ""),
        "group_a_answer": a["answer_text"],
        "group_a_tag": a["group_tag"],
        "group_a_ans_idx": a["ans_idx"],
        "group_b_answer": b["answer_text"],
        "group_b_tag": b["group_tag"],
        "group_b_ans_idx": b["ans_idx"],
        "stereotyped_groups": stereotyped,
        "label": example.get("label"),
    }


def subsample_per_question_index(
    examples: List[Dict[str, Any]],
    max_samples_per_question: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Stratified subsample: keep at most *max_samples_per_question* rows
    per ``question_index``, chosen randomly via *rng*."""
    by_qi: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        by_qi[ex["question_index"]].append(ex)
    result: List[Dict[str, Any]] = []
    for group in by_qi.values():
        if len(group) > max_samples_per_question:
            result.extend(rng.sample(group, max_samples_per_question))
        else:
            result.extend(group)
    return result


def prepare_pairwise_examples(
    examples: List[Dict[str, Any]],
    context_condition: str = "ambig",
) -> List[Dict[str, Any]]:
    """
    Filter and convert BBQ examples into pairwise forced-choice records.

    Args:
        examples: Raw BBQ examples from load_bbq_category.
        context_condition: Which contexts to include.
            "ambig"    – ambiguous only (default).
            "disambig" – disambiguated only.
            "both"     – no filtering by context condition.

    Returns:
        List of pairwise dicts ready for prompt generation.
    """
    filtered = []
    for ex in examples:
        if context_condition != "both":
            if ex.get("context_condition") != context_condition:
                continue
        record = _extract_pairwise(ex)
        if record is not None:
            filtered.append(record)

    return filtered


def get_group_tags_for_category(
    examples: List[Dict[str, Any]],
) -> List[str]:
    """
    Discover the set of non-unknown group tags present in pairwise examples.

    Returns a sorted list so factor Variable values are deterministic.
    """
    tags = set()
    for ex in examples:
        tags.add(ex["group_a_tag"])
        tags.add(ex["group_b_tag"])
    return sorted(tags)


def get_factor_for_category(category: str, group_tags: List[str]) -> Variable:
    """
    Build a Variable for a BBQ category's bias dimension.

    The variable name is the short category name (e.g. "age", "gender").
    """
    return Variable(name=category, values=group_tags)
