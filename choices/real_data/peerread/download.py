"""Download PeerRead JSON files from GitHub using sparse checkout (skips PDFs).

The full repo is ~1.2GB due to PDFs. We use git sparse-checkout to clone
only the JSON metadata files (~20MB).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "peerread"
REPO_URL = "https://github.com/allenai/PeerRead.git"


def download() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    reviews_path = DATA_DIR / "reviews.jsonl"
    papers_path = DATA_DIR / "parsed_pdfs.jsonl"

    if reviews_path.exists() and papers_path.exists():
        print(f"Already downloaded: {DATA_DIR}")
        return DATA_DIR

    tmpdir = tempfile.mkdtemp()
    clone_dir = Path(tmpdir) / "PeerRead"

    try:
        print("Sparse cloning PeerRead (JSON only, no PDFs)...")
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--depth",
                "1",
                "--filter=blob:none",
                REPO_URL,
                str(clone_dir),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "sparse-checkout",
                "set",
                "--no-cone",
                "data/**/reviews/*.json",
                "data/**/parsed_pdfs/*.json",
            ],
            cwd=clone_dir,
            check=True,
        )
        subprocess.run(
            ["git", "checkout"],
            cwd=clone_dir,
            check=True,
        )

        # Collect JSON files
        reviews = []
        papers = []
        json_files = sorted(clone_dir.rglob("*.json"))
        for i, path in enumerate(json_files):
            print(f"\r  Processing {i + 1}/{len(json_files)}", end="", flush=True)
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if "/reviews/" in str(path):
                reviews.append(data)
            elif "/parsed_pdfs/" in str(path):
                papers.append(data)
        print()

        with open(reviews_path, "w") as f:
            for r in reviews:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  {len(reviews)} reviews -> {reviews_path.name}")

        with open(papers_path, "w") as f:
            for p in papers:
                f.write(json.dumps(p, default=str) + "\n")
        print(f"  {len(papers)} papers -> {papers_path.name}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"Dataset ready at {DATA_DIR}")
    return DATA_DIR


if __name__ == "__main__":
    download()
