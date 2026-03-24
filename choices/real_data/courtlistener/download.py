"""Download CourtListener data: judge demographics from S3 + opinions from HuggingFace.

Judge data (people, races, political affiliations) is tiny (<1MB total).
Opinions are streamed from harvard-lil/cold-cases on HuggingFace and sampled
to a manageable size (default 10k).
"""

from __future__ import annotations

import bz2
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "courtlistener"
S3_BASE = "https://com-courtlistener-storage.s3-us-west-2.amazonaws.com/bulk-data"

# Latest available snapshot
SNAPSHOT = "2025-12-31"

JUDGE_FILES = {
    "people.csv": f"people-db-people-{SNAPSHOT}.csv.bz2",
    "races.csv": f"people-db-races-{SNAPSHOT}.csv.bz2",
    "political_affiliations.csv": f"people-db-political-affiliations-{SNAPSHOT}.csv.bz2",
}


def _download_bz2_csv(url: str, dest: Path) -> None:
    """Download a bz2-compressed CSV from S3, decompress, and save."""
    print(f"  Downloading {url.split('/')[-1]}...")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        compressed = resp.read()
    decompressed = bz2.decompress(compressed)
    dest.write_bytes(decompressed)
    # Count lines for progress
    n_lines = decompressed.count(b"\n") - 1
    print(f"    -> {dest.name} ({n_lines} rows)")


def download_judges() -> None:
    """Download judge demographic CSVs from CourtListener S3."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for local_name, s3_name in JUDGE_FILES.items():
        dest = DATA_DIR / local_name
        if dest.exists():
            print(f"  Already have {local_name}")
            continue
        url = f"{S3_BASE}/{s3_name}"
        _download_bz2_csv(url, dest)


def download_opinions(n: int = 10000, seed: int = 42) -> None:
    """Download a sample of opinions from harvard-lil/cold-cases (HuggingFace).

    Streams the dataset and saves n opinions that have an author_id,
    so they can be joined to judge demographics.
    """
    dest = DATA_DIR / "opinions.jsonl"
    if dest.exists():
        print("  Already have opinions.jsonl")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets is required. Install with: pip install datasets")

    print(f"  Streaming cold-cases from HuggingFace (sampling {n} with author_id)...")
    ds = load_dataset("harvard-lil/cold-cases", split="train", streaming=True)

    collected = []
    seen = 0
    for row in ds:
        seen += 1
        if seen % 10000 == 0:
            print(f"    Scanned {seen}, collected {len(collected)}/{n}...")

        # Each row has 'opinions' which is a list of opinion dicts
        opinions = row.get("opinions") or []
        cluster_info = {
            "case_name": row.get("case_name", ""),
            "case_name_short": row.get("case_name_short", ""),
            "date_filed": row.get("date_filed", ""),
            "court_short_name": row.get("court_short_name", ""),
            "precedential_status": row.get("precedential_status", ""),
            "judges": row.get("judges", ""),
        }

        for op in opinions:
            author_id = op.get("author_id")
            if not author_id:
                continue

            text = (op.get("opinion_text") or "").strip()
            if not text or len(text) < 200:
                continue

            collected.append(
                {
                    **cluster_info,
                    "opinion_id": op.get("opinion_id"),
                    "author_id": author_id,
                    "author_str": op.get("author_str", ""),
                    "type": op.get("type", ""),
                    "per_curiam": op.get("per_curiam", False),
                    "text": text,
                }
            )

            if len(collected) >= n:
                break

        if len(collected) >= n:
            break

    print(f"    Scanned {seen} clusters total, collected {len(collected)} opinions")

    with open(dest, "w") as f:
        for rec in collected:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"    -> opinions.jsonl ({len(collected)} records)")


def download(n_opinions: int = 10000) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading judge demographics from CourtListener S3...")
    download_judges()

    print("Downloading opinions from HuggingFace cold-cases...")
    download_opinions(n=n_opinions)

    print(f"Dataset ready at {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-opinions",
        type=int,
        default=10000,
        help="Number of opinions to sample (default: 10000)",
    )
    args = parser.parse_args()
    download(n_opinions=args.n_opinions)
