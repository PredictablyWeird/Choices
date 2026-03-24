"""LLM-based rewriting for datasets where demographics are embedded in text.

For datasets like AITA where you can't just swap a format-string field, this
module sends each record to an LLM judge that produces counterfactual variants
(e.g. male/female versions) with minimal changes, or skips unsuitable examples.

Self-contained: uses the OpenAI SDK directly, no dependency on choices.llm_agent.

Presets live in choices/real_data/presets/*.yaml — add a new YAML file to create
a new rewrite type. No code changes needed.

Usage:
    # List available presets
    python -m choices.real_data.llm_rewrite --list-presets

    # Run a rewrite
    python -m choices.real_data.llm_rewrite \\
        --source aita --preset gender_aita \\
        --model gpt-5.2 --reasoning low --n 100

    # Programmatic
    from choices.real_data.llm_rewrite import load_preset, RewriteJob, run_rewrite
    preset = load_preset("gender_aita")
    job = RewriteJob(source="aita", **preset, model="gpt-5.2", n=100)
    asyncio.run(run_rewrite(job))
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

REWRITE_DIR = Path(__file__).parent / "converted" / "rewritten"
PRESETS_DIR = Path(__file__).parent / "presets"


# =============================================================================
# Preset loading
# =============================================================================
def load_preset(name: str) -> dict:
    """Load a preset YAML file by name. Returns dict with name, variants, instruction."""
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        available = list_presets()
        raise FileNotFoundError(
            f"Preset '{name}' not found at {path}.\n"
            f"Available: {', '.join(available)}"
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    for key in ("name", "variants", "instruction"):
        if key not in data:
            raise ValueError(f"Preset '{name}' missing required key: {key}")
    return data


def list_presets() -> list[str]:
    """List available preset names (YAML files in presets/)."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


# =============================================================================
# Core
# =============================================================================
def _build_prompt(
    record: dict,
    instruction: str,
    variant: str,
    rewrite_fields: list[str] | None = None,
) -> str:
    """Build the full prompt for a single rewrite request.

    If rewrite_fields is specified, only those text_fields are sent to the LLM
    (avoids sending metadata headers that shouldn't be rewritten).
    Otherwise falls back to rendering the full prompt_template.
    """
    if rewrite_fields:
        text_parts = []
        for field in rewrite_fields:
            val = record.get("text_fields", {}).get(field, "")
            if val:
                text_parts.append(val)
        original_text = "\n\n".join(text_parts)
    else:
        fields = {
            **record.get("modifiable_fields", {}),
            **record.get("text_fields", {}),
        }
        original_text = record["prompt_template"].format_map(fields)
    formatted_instruction = instruction.format(variant=variant)
    return f"{formatted_instruction}\n\nORIGINAL TEXT:\n{original_text}"


@dataclass
class RewriteJob:
    """Configuration for an LLM rewrite job.

    Can be constructed from a preset:
        preset = load_preset("gender_aita")
        job = RewriteJob(source="aita", **preset, model="gpt-5.2")
    """

    source: str
    name: str
    variants: list[str]
    instruction: str
    model: str = "gpt-4o-mini"
    reasoning: dict | None = None
    n: int = 100
    seed: int = 42
    max_tokens: int = 4096
    concurrency: int = 30
    filter_fn: Callable | None = None
    description: str = ""
    rewrite_fields: list[str] | None = None

    def output_path(self) -> Path:
        return REWRITE_DIR / f"{self.source}_{self.name}_{self.model}.jsonl"


# =============================================================================
# API callers
# =============================================================================
async def _call_responses_api(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    reasoning: dict | None,
) -> str | None:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    if reasoning:
        kwargs["reasoning"] = reasoning
    response = await client.responses.create(**kwargs)
    return response.output_text


async def _call_chat_api(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
) -> str | None:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# =============================================================================
# Runner
# =============================================================================
async def run_rewrite(job: RewriteJob) -> list[dict]:
    """Run a rewrite job and save results.

    Returns list of result dicts, each containing:
        - id: original record id
        - variants: dict mapping variant name -> rewritten text (or None if skipped)
        - skipped: bool, True if ALL variants were skipped
    """
    from choices.real_data.base import load_dataset

    records = load_dataset(job.source)

    if job.filter_fn:
        records = [r for r in records if job.filter_fn(r)]

    rng = random.Random(job.seed)
    if len(records) > job.n:
        records = rng.sample(records, job.n)

    print(
        f"Rewriting {len(records)} records x {len(job.variants)} variants "
        f"using {job.model} (reasoning={job.reasoning})..."
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    semaphore = asyncio.Semaphore(job.concurrency)

    work_items = []
    for rec_idx, record in enumerate(records):
        for variant in job.variants:
            prompt = _build_prompt(record, job.instruction, variant, job.rewrite_fields)
            work_items.append((rec_idx, variant, prompt))

    results_map: dict[int, dict] = {}
    errors = {"count": 0}

    async def process(item_idx: int):
        rec_idx, variant, prompt = work_items[item_idx]
        max_retries = 5
        delay = 2.0
        text = None

        for attempt in range(max_retries):
            try:
                async with semaphore:
                    if job.reasoning is not None:
                        text = await _call_responses_api(
                            client, prompt, job.model, job.max_tokens, job.reasoning
                        )
                    else:
                        text = await _call_chat_api(
                            client, prompt, job.model, job.max_tokens
                        )
                break
            except Exception as e:
                errors["count"] += 1
                if attempt == max_retries - 1:
                    print(f"[Failed] item {item_idx} after {max_retries} attempts: {e}")
                    text = None
                else:
                    await asyncio.sleep(delay + random.random())
                    delay *= 2

        if rec_idx not in results_map:
            results_map[rec_idx] = {
                "id": records[rec_idx]["id"],
                "source": job.source,
                "rewrite_name": job.name,
                "model": job.model,
                "variants": {},
            }
        if text and text.strip().upper() != "SKIP":
            results_map[rec_idx]["variants"][variant] = text.strip()
        else:
            results_map[rec_idx]["variants"][variant] = None

    tasks = [asyncio.create_task(process(i)) for i in range(len(work_items))]
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Rewriting"):
        await coro

    if errors["count"]:
        print(f"Total errors (across retries): {errors['count']}")

    results = []
    n_skipped = 0
    for rec_idx in sorted(results_map):
        entry = results_map[rec_idx]
        if all(v is None for v in entry["variants"].values()):
            n_skipped += 1
        else:
            results.append(entry)

    out_path = job.output_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for entry in results:
            f.write(json.dumps(entry) + "\n")

    print(f"Done. {len(results)} usable, {n_skipped} skipped. Saved to {out_path}")

    return results


def load_rewritten(source: str, name: str, model: str) -> list[dict]:
    """Load rewritten results."""
    path = REWRITE_DIR / f"{source}_{name}_{model}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No rewritten data at {path}")
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


# =============================================================================
# Filtering
# =============================================================================


def _build_filter_fn(preset: dict, min_text: int) -> Callable | None:
    """Build a filter function from the preset's filter_pattern + min text length."""
    fns: list[Callable] = []

    if min_text > 0:

        def text_len(record: dict) -> bool:
            return (
                sum(len(v) for v in record.get("text_fields", {}).values()) > min_text
            )

        fns.append(text_len)

    if pattern := preset.get("filter_pattern"):
        compiled = re.compile(pattern)

        def pattern_match(record: dict) -> bool:
            text = " ".join(record.get("text_fields", {}).values())
            return bool(compiled.search(text))

        fns.append(pattern_match)

    if max_text := preset.get("max_text_length"):

        def text_max(record: dict) -> bool:
            return (
                sum(len(v) for v in record.get("text_fields", {}).values()) <= max_text
            )

        fns.append(text_max)

    if not fns:
        return None

    def combined(record: dict) -> bool:
        return all(fn(record) for fn in fns)

    return combined


# =============================================================================
# CLI
# =============================================================================
def main():
    import argparse

    available = list_presets()

    parser = argparse.ArgumentParser(
        description="Rewrite dataset records using an LLM.",
    )
    parser.add_argument(
        "--list-presets", action="store_true", help="List available presets and exit"
    )
    parser.add_argument("--source", help="Dataset source (e.g. aita, okcupid, resumes)")
    parser.add_argument(
        "--preset", choices=available, help="Instruction preset (from presets/*.yaml)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--reasoning",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort (uses Responses API). Omit for Chat API.",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=200,
        help="Skip records with fewer chars of text",
    )
    args = parser.parse_args()

    if args.list_presets:
        for name in available:
            preset = load_preset(name)
            desc = preset.get("description", "").strip().replace("\n", " ")[:80]
            variants = ", ".join(preset["variants"])
            fp = preset.get("filter_pattern", "")
            print(f"  {name:35s} [{variants}]")
            print(f"    {desc}")
            if fp:
                print(f"    filter: {fp}")
        return

    if not args.source or not args.preset:
        parser.error("--source and --preset are required (or use --list-presets)")

    preset = load_preset(args.preset)
    reasoning = None
    if args.reasoning:
        reasoning = {"effort": args.reasoning, "summary": "detailed"}

    job = RewriteJob(
        source=args.source,
        name=preset["name"],
        variants=preset["variants"],
        instruction=preset["instruction"],
        model=args.model,
        reasoning=reasoning,
        n=args.n,
        seed=args.seed,
        concurrency=args.concurrency,
        filter_fn=_build_filter_fn(preset, args.min_text_length),
        rewrite_fields=preset.get("rewrite_fields"),
    )

    results = asyncio.run(run_rewrite(job))
    print(f"\nSummary: {len(results)} usable records")


if __name__ == "__main__":
    main()
