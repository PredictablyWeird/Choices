"""Run the eval-awareness confabulation experiment.

Calls Kimi K2.5 via OpenRouter with reasoning enabled across all
(question, condition, repetition) triples. Saves results as JSONL.
"""

import asyncio
import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from prompts import Condition, build_messages
from questions import QUESTIONS

load_dotenv()

RESULTS_DIR = Path("results")
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.jsonl"
MODEL = "moonshotai/kimi-k2.5"
REPS = 3
MAX_CONCURRENT = 50
TEMPERATURE = 1.0


class TrialResult(BaseModel):
    question_id: str
    condition: str
    repetition: int
    response_text: str
    reasoning_text: str | None
    timestamp: float
    model: str
    usage: dict | None


def get_client() -> AsyncOpenAI:
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set — check .env")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def load_completed(path: Path) -> set[tuple[str, str, int]]:
    """Load already-completed trial keys from existing results file."""
    completed = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                completed.add((r["question_id"], r["condition"], r["repetition"]))
    return completed


def extract_reasoning(response) -> str | None:
    """Extract reasoning content from OpenRouter response."""
    msg = response.choices[0].message
    # OpenRouter may put reasoning in different places
    # 1. message.reasoning_content (common)
    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
        return msg.reasoning_content
    # 2. message.reasoning (alternative field name)
    if hasattr(msg, "reasoning") and msg.reasoning:
        return msg.reasoning
    # 3. Check model_extra for non-standard fields
    if hasattr(msg, "model_extra") and msg.model_extra:
        for key in ("reasoning_content", "reasoning", "reasoning_details"):
            val = msg.model_extra.get(key)
            if val:
                if isinstance(val, list):
                    # reasoning_details is a list of objects
                    return "\n".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in val
                    )
                return str(val)
    return None


async def run_trial(
    client: AsyncOpenAI,
    question_id: str,
    question_text: str,
    condition: Condition,
    repetition: int,
    semaphore: asyncio.Semaphore,
) -> TrialResult:
    """Run a single trial and return the result."""
    prompt = build_messages(condition, question_text)

    kwargs: dict = {
        "model": MODEL,
        "messages": prompt.messages,
        "temperature": TEMPERATURE,
        "max_tokens": 4096,
        "extra_body": {
            "reasoning": {"effort": "high"},
        },
    }
    if prompt.tools:
        kwargs["tools"] = prompt.tools

    async with semaphore:
        response = await client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    usage = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    # Handle tool call responses (agentic condition may trigger write_file)
    response_text = msg.content or ""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_parts = []
        for tc in msg.tool_calls:
            tool_parts.append(
                f"[tool_call: {tc.function.name}({tc.function.arguments})]"
            )
        if response_text:
            response_text += "\n" + "\n".join(tool_parts)
        else:
            response_text = "\n".join(tool_parts)

    return TrialResult(
        question_id=question_id,
        condition=condition.value,
        repetition=repetition,
        response_text=response_text,
        reasoning_text=extract_reasoning(response),
        timestamp=time.time(),
        model=MODEL,
        usage=usage,
    )


async def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    client = get_client()
    completed = load_completed(RAW_RESULTS_FILE)

    # Build trial list
    trials = []
    for q in QUESTIONS:
        for condition in Condition:
            for rep in range(REPS):
                key = (q.id, condition.value, rep)
                if key not in completed:
                    trials.append((q, condition, rep))

    if not trials:
        print("All trials already completed!")
        return

    random.shuffle(trials)
    print(f"Running {len(trials)} trials ({len(completed)} already completed)")

    # Sanity check: verify we get reasoning back
    print("Sanity check: testing API connection and reasoning...")
    test_q = QUESTIONS[0]
    test_cond = Condition.BARE_EVAL
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    test_result = await run_trial(
        client,
        test_q.id,
        test_q.text,
        test_cond,
        -1,
        semaphore,
    )
    if test_result.reasoning_text:
        print(f"  Reasoning trace received ({len(test_result.reasoning_text)} chars)")
    else:
        print("  WARNING: No reasoning trace in response! Check API parameters.")
        print(f"  Response preview: {test_result.response_text[:200]}")
        resp = input("  Continue anyway? [y/N] ")
        if resp.lower() != "y":
            return

    # Run all trials
    results_file = open(RAW_RESULTS_FILE, "a")

    async def run_and_save(q, condition, rep):
        try:
            result = await run_trial(
                client,
                q.id,
                q.text,
                condition,
                rep,
                semaphore,
            )
            results_file.write(result.model_dump_json() + "\n")
            results_file.flush()
            return True
        except Exception as e:
            print(f"\nError on {q.id}/{condition.value}/{rep}: {e}")
            return False

    pbar = tqdm(total=len(trials), desc="Trials")
    # Run in batches to avoid overwhelming the API
    batch_size = MAX_CONCURRENT * 2
    for i in range(0, len(trials), batch_size):
        batch = trials[i : i + batch_size]
        tasks = [run_and_save(q, c, r) for q, c, r in batch]
        results = await asyncio.gather(*tasks)
        pbar.update(sum(1 for r in results if r))

    pbar.close()
    results_file.close()
    print(f"Done! Results saved to {RAW_RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
