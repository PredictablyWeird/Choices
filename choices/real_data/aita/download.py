"""Download the AITA dataset from the Iterative/DVC public remote."""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "aita"

# DVC remote stores files as md5[:2]/md5[2:]
DVC_REMOTE = "https://remote.dvc.org/aita_dataset"
FILE_MD5 = "e89dc2d659cdf8f2f6da84f3636a260e"


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "aita_clean.csv"

    if csv_path.exists():
        print(f"Already downloaded: {csv_path}")
        return DATA_DIR

    url = f"{DVC_REMOTE}/{FILE_MD5[:2]}/{FILE_MD5[2:]}"
    print(f"Downloading AITA dataset from {url}...")
    urllib.request.urlretrieve(url, csv_path)
    print(f"Dataset ready at {csv_path}")
    return DATA_DIR


if __name__ == "__main__":
    download()
