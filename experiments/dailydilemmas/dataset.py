"""
Download and parse the DailyDilemmas dataset from HuggingFace.

Each dilemma is a binary moral choice (to_do vs not_to_do) with associated
human values in tension. This module pivots the 2-row-per-dilemma format
into a single Dilemma dataclass and groups dilemmas by their dominant value pair.

Dataset: kellycyy/daily_dilemmas (Chiu et al., ICLR 2025 spotlight)
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "kellycyy/daily_dilemmas"
DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Dilemma:
    id: int
    situation: str
    action_to_do: str
    action_not_to_do: str
    consequence_to_do: str
    consequence_not_to_do: str
    values_to_do: tuple[str, ...]
    values_not_to_do: tuple[str, ...]
    topic: int
    topic_group: str

    @property
    def value_pair(self) -> tuple[str, str]:
        """Dominant value pair: (primary to_do value, primary not_to_do value)."""
        v1 = self.values_to_do[0] if self.values_to_do else "unknown"
        v2 = self.values_not_to_do[0] if self.values_not_to_do else "unknown"
        return (v1, v2)

    @property
    def canonical_value_pair(self) -> tuple[str, str]:
        """Sorted value pair for grouping (order-independent)."""
        return tuple(sorted(self.value_pair))


def _download_dataset() -> pd.DataFrame:
    """Download the dilemma_to_action_to_values_aggregated CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_DIR / "dilemmas.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename="dilemma_to_action_to_values_aggregated.csv",
        repo_type="dataset",
    )
    df = pd.read_csv(path)
    df.to_csv(cache_path, index=False)
    return df


def _parse_values(values_str: str) -> tuple[str, ...]:
    """Parse the values_aggregated string (Python list literal) into a tuple."""
    try:
        values = ast.literal_eval(values_str)
        if isinstance(values, list):
            return tuple(values)
    except (ValueError, SyntaxError):
        pass
    return ()


def load_dilemmas() -> list[Dilemma]:
    """
    Download and parse all dilemmas into Dilemma objects.

    Returns a list of Dilemma objects, one per unique dilemma (pivoted from
    the 2-row-per-dilemma format in the dataset).
    """
    df = _download_dataset()

    # Pivot: group by dilemma_idx, separate to_do and not_to_do rows
    dilemmas = []
    for dilemma_idx, group in df.groupby("dilemma_idx"):
        to_do = group[group["action_type"] == "to_do"]
        not_to_do = group[group["action_type"] == "not_to_do"]

        if to_do.empty or not_to_do.empty:
            continue

        to_do_row = to_do.iloc[0]
        not_to_do_row = not_to_do.iloc[0]

        dilemmas.append(
            Dilemma(
                id=int(dilemma_idx),
                situation=to_do_row["dilemma_situation"],
                action_to_do=to_do_row["action"],
                action_not_to_do=not_to_do_row["action"],
                consequence_to_do=to_do_row["negative_consequence"],
                consequence_not_to_do=not_to_do_row["negative_consequence"],
                values_to_do=_parse_values(to_do_row["values_aggregated"]),
                values_not_to_do=_parse_values(not_to_do_row["values_aggregated"]),
                topic=int(to_do_row["topic"]),
                topic_group=to_do_row["topic_group"],
            )
        )

    return dilemmas


def group_by_value_pair(
    dilemmas: list[Dilemma],
) -> dict[tuple[str, str], list[Dilemma]]:
    """Group dilemmas by their canonical (sorted) value pair."""
    groups: dict[tuple[str, str], list[Dilemma]] = {}
    for d in dilemmas:
        key = d.canonical_value_pair
        groups.setdefault(key, []).append(d)
    return groups


def print_value_pair_distribution(dilemmas: list[Dilemma]) -> None:
    """Print the distribution of value pairs across dilemmas."""
    pairs = [d.canonical_value_pair for d in dilemmas]
    counts = Counter(pairs)

    print(f"\nTotal dilemmas: {len(dilemmas)}")
    print(f"Unique value pairs: {len(counts)}")

    all_values = set()
    for d in dilemmas:
        all_values.update(d.values_to_do)
        all_values.update(d.values_not_to_do)
    print(f"Unique values: {len(all_values)}")

    print(f"\nTop 30 value pairs (of {len(counts)}):")
    for (v1, v2), count in counts.most_common(30):
        pct = count / len(dilemmas) * 100
        print(f"  {v1:30s} vs {v2:30s}  {count:4d} ({pct:.1f}%)")

    # Cumulative coverage
    print("\nCumulative coverage:")
    cumulative = 0
    for i, ((v1, v2), count) in enumerate(counts.most_common(), 1):
        cumulative += count
        if i in (5, 10, 15, 20, 30, 50) or i == len(counts):
            print(
                f"  Top {i:3d} pairs: {cumulative:4d} dilemmas ({cumulative/len(dilemmas)*100:.1f}%)"
            )


def print_sample_dilemmas(dilemmas: list[Dilemma], n: int = 3) -> None:
    """Print a few sample dilemmas for inspection."""
    print(f"\n{'='*80}")
    print(f"Sample dilemmas (first {n}):")
    print(f"{'='*80}")
    for d in dilemmas[:n]:
        print(f"\n--- Dilemma {d.id} (topic: {d.topic_group}) ---")
        print(f"Situation: {d.situation[:200]}...")
        print(f"  To do:     {d.action_to_do}")
        print(f"  Not to do: {d.action_not_to_do}")
        print(f"  Values (to_do):     {d.values_to_do}")
        print(f"  Values (not_to_do): {d.values_not_to_do}")
        print(f"  Value pair: {d.value_pair}")


if __name__ == "__main__":
    dilemmas = load_dilemmas()
    print_value_pair_distribution(dilemmas)
    print_sample_dilemmas(dilemmas)
