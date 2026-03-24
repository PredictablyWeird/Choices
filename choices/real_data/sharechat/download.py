"""Download ShareChat conversations from HuggingFace.

Streams the dataset and saves the first user message of each conversation,
filtered to English and substantive messages (>50 chars). Groups turn-level
rows by URL to extract conversation openers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sharechat"

# Platform configs with model filters (only newer models).
# For platforms where version field is unreliable (claude, perplexity), we skip.
PLATFORM_CONFIGS = {
    "chatgpt": {
        "models": {
            "o3",
            "o3-mini",
            "o3-mini-high",
            "gpt-4-5",
            "gpt-4-1-mini",
            "o1",
            "o1-pro",
        },
        "version_key": "version",
    },
    "gemini": {
        "models": {
            "2.5 Pro",
            "2.5 Flash",
            "2.5 Pro (experimental)",
            "2.5 Pro (preview)",
            "2.5 Flash (preview)",
            "2.5 Flash (experimental)",
        },
        "version_key": "version",
    },
    "grok": {
        "models": {"grok-3", "grok-3-reasoning", "grok-latest"},
        "version_key": "version",
    },
}


def download(n_per_platform: int = 2000) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dest = DATA_DIR / "conversations.jsonl"
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return DATA_DIR

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("datasets is required. Install with: pip install datasets")

    token = os.getenv("HF_TOKEN")

    all_records = []
    for platform, config in PLATFORM_CONFIGS.items():
        allowed_models = config["models"]
        version_key = config["version_key"]
        print(f"Streaming {platform} (filtering to {len(allowed_models)} model(s))...")
        ds = load_dataset(
            "tucnguyen/ShareChat",
            platform,
            streaming=True,
            split="train",
            token=token,
        )

        # Group turns by URL, collect first user message
        current_url = None
        current_conv = []
        collected = 0
        scanned = 0

        for row in ds:
            scanned += 1
            url = row.get("url", "")

            # New conversation
            if url != current_url:
                # Process previous conversation
                if current_conv:
                    rec = _extract_record(
                        current_conv, platform, allowed_models, version_key
                    )
                    if rec:
                        all_records.append(rec)
                        collected += 1
                        if collected % 200 == 0:
                            print(
                                f"  {platform}: {collected}/{n_per_platform} "
                                f"(scanned {scanned} rows)"
                            )
                        if collected >= n_per_platform:
                            break

                current_url = url
                current_conv = []

            current_conv.append(row)

        # Don't forget last conversation
        if current_conv and collected < n_per_platform:
            rec = _extract_record(current_conv, platform, allowed_models, version_key)
            if rec:
                all_records.append(rec)
                collected += 1

        print(
            f"  {platform}: {collected} conversations collected "
            f"(scanned {scanned} rows)"
        )

    with open(dest, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Total: {len(all_records)} conversations -> {dest}")
    return DATA_DIR


def _extract_record(
    turns: list[dict],
    platform: str,
    allowed_models: set[str] | None = None,
    version_key: str = "version",
) -> dict | None:
    """Extract a record from a conversation's turns."""
    # Find first user message and check model version
    first_user = None
    first_reply = None
    model_version = ""

    for t in turns:
        # Grab model version from any turn
        v = t.get(version_key) or t.get("model") or ""
        if isinstance(v, list):
            v = ""
        if v:
            model_version = str(v)

        if t["role"] == "user" and first_user is None:
            first_user = t
        elif (
            t["role"] in ("llm", "ASSISTANT", "assistant")
            and first_user is not None
            and first_reply is None
        ):
            first_reply = t

    if not first_user:
        return None

    # Filter by model
    if allowed_models and model_version not in allowed_models:
        return None

    text = (first_user.get("plain_text") or "").strip()

    # Filter: English, substantive
    lang = first_user.get("detected_language_final", "")
    if lang != "English":
        return None
    if len(text) < 50:
        return None
    # Skip code-only messages
    if text.startswith("```") or text.startswith("def ") or text.startswith("import "):
        return None

    record = {
        "platform": platform,
        "url": first_user.get("url", ""),
        "user_message": text,
        "model_version": model_version,
        "turns_count": first_user.get("turns_count", 0),
        "detected_language": lang,
    }

    if first_reply:
        reply_text = (first_reply.get("plain_text") or "").strip()
        if reply_text:
            record["model_reply"] = reply_text

    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-platform", type=int, default=2000)
    args = parser.parse_args()
    download(n_per_platform=args.n_per_platform)
