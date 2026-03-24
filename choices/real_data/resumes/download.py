"""Download the Resume Dataset (snehaanbhawal/resume-dataset from Kaggle)."""

from __future__ import annotations

import shutil
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "resumes"


def download() -> Path:
    """Download resume dataset using kagglehub.

    Returns path to the directory containing the CSV.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = DATA_DIR / "Resume.csv"
    if csv_path.exists():
        print(f"Already downloaded: {csv_path}")
        return DATA_DIR

    try:
        import kagglehub
    except ImportError:
        raise RuntimeError("kagglehub is required. Install with: pip install kagglehub")

    print("Downloading snehaanbhawal/resume-dataset from Kaggle...")
    downloaded_path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    downloaded_path = Path(downloaded_path)

    # kagglehub downloads to a cache dir; copy CSV to our data dir
    for csv_file in downloaded_path.rglob("*.csv"):
        dest = DATA_DIR / csv_file.name
        if not dest.exists():
            shutil.copy2(csv_file, dest)
            print(f"Copied {csv_file.name} -> {dest}")

    if not csv_path.exists():
        # Try case-insensitive match
        csvs = list(DATA_DIR.glob("*.csv"))
        if csvs:
            print(f"Found: {[c.name for c in csvs]}")
        else:
            raise FileNotFoundError(
                f"No CSV found after download. "
                f"Files in cache: {list(downloaded_path.rglob('*'))}"
            )

    print(f"Dataset ready at {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download()
