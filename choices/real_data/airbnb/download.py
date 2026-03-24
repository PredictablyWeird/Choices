"""Download Inside Airbnb NYC listings from Kaggle (dominoweir/inside-airbnb-nyc).

Contains the full detailed listings.csv with 74 columns including host_name,
description, neighbourhood, price, reviews, etc.
"""

from __future__ import annotations

import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "airbnb"
DATASET_SLUG = "dominoweir/inside-airbnb-nyc"


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if we already have a listings CSV
    existing = list(DATA_DIR.glob("*listings*.csv"))
    if existing:
        print(f"Already downloaded: {existing[0]}")
        return DATA_DIR

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub is required. Install with: pip install kagglehub")

    print(f"Downloading {DATASET_SLUG} from Kaggle...")
    downloaded_path = Path(kagglehub.dataset_download(DATASET_SLUG))

    # Copy CSV files to our data dir
    for csv_file in downloaded_path.rglob("*.csv"):
        dest = DATA_DIR / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
            print(f"  {csv_file.name} -> {dest}")

    # Also copy any gz files
    for gz_file in downloaded_path.rglob("*.csv.gz"):
        dest = DATA_DIR / gz_file.name
        if not dest.exists():
            shutil.copy2(gz_file, dest)
            print(f"  {gz_file.name} -> {dest}")

    print(f"Dataset ready at {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download()
