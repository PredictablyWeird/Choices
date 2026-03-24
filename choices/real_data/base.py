"""Base classes for real-data dataset downloading and conversion.

Each dataset module (okcupid/, resumes/, medical/, etc.) implements:
  - A download function that fetches raw data into real_data/data/<source>/
  - A convert function that produces standardized JSON records

Standard record format:
{
    "source": "okcupid",           # dataset identifier
    "id": "okcupid_0",            # unique record id
    "profile_type": "dating",      # category: dating, resume, medical, etc.
    "modifiable_fields": {         # demographics that can be swapped
        "sex": "m",
        "age": 25,
        ...
    },
    "text_fields": {               # free-text content
        "self_summary": "...",
        ...
    },
    "prompt_template": "..."       # renderable template with {field} placeholders
}
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "converted"


class DatasetConverter(ABC):
    """Base class for converting a raw dataset to the standard format."""

    source: str  # e.g. "okcupid", "resume", "medical"
    profile_type: str  # e.g. "dating", "resume", "medical_report"

    @abstractmethod
    def raw_data_path(self) -> Path:
        """Path to the raw downloaded data."""
        ...

    @abstractmethod
    def convert(self) -> list[dict[str, Any]]:
        """Convert raw data to list of standard records."""
        ...

    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.source}.jsonl"

    def run(self) -> Path:
        """Convert and write to JSONL."""
        records = self.convert()
        out = self.output_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote {len(records)} records to {out}")
        return out


def load_dataset(source: str) -> list[dict[str, Any]]:
    """Load a converted dataset by source name."""
    path = OUTPUT_DIR / f"{source}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No converted data for '{source}'. Run the conversion first."
        )
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def serve(
    source: str, modifications: dict[str, Any] | None = None, index: int = 0
) -> str:
    """Serve a single record with optional field modifications.

    Example:
        serve("okcupid", {"sex": "f", "age": 30}, index=5)
    """
    records = load_dataset(source)
    if index >= len(records):
        raise IndexError(
            f"Index {index} out of range (dataset has {len(records)} records)"
        )
    record = records[index]
    fields = {**record.get("modifiable_fields", {}), **record.get("text_fields", {})}
    if modifications:
        fields.update(modifications)
    return record["prompt_template"].format_map(fields)
