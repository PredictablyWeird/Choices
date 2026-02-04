#!/usr/bin/env python3
"""
Analyze differences between two sets of reasoning traces using an LLM.

This script takes a JSON file containing paired reasoning traces (from two conditions)
and uses an LLM to analyze what's different between them.

The analysis is done one edge at a time, comparing traces from condition A vs condition B,
without revealing the experimental context to avoid biasing the analysis.

Usage:
    uv run python -m choices.analysis.reasoning_traces.analyze_trace_differences \
        --input backfire_cases.json \
        --output analysis_results.json \
        --num-samples 50

    # Estimate cost first
    uv run python -m choices.analysis.reasoning_traces.analyze_trace_differences \
        --input backfire_cases.json \
        --estimate-cost
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List

import httpx


# =============================================================================
# OpenRouter Client (adapted from classify.py)
# =============================================================================

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

MODEL_PRICING = {
    "google/gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
    "anthropic/claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

DEFAULT_MODEL = "google/gemini-2.0-flash-001"


async def call_openrouter_async(
    client: httpx.AsyncClient,
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.0,
    max_retries: int = 5,
) -> str:
    """Call OpenRouter API asynchronously and return the response text."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/choices",
    }

    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        response = await client.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers=headers,
            json=data,
        )

        if response.status_code == 429:
            wait_time = (2**attempt) * 2 + 1
            print(
                f"  Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(wait_time)
            continue

        if response.status_code != 200:
            print(f"API error {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        result = response.json()
        if "choices" not in result:
            if "error" in result:
                raise Exception(
                    f"API error: {result.get('error', {}).get('message', 'Unknown')}"
                )
            raise Exception(f"No 'choices' in response: {str(result)[:200]}")
        return result["choices"][0]["message"]["content"]

    raise Exception(f"Failed after {max_retries} retries due to rate limiting")


# =============================================================================
# Analysis Prompt
# =============================================================================

ANALYSIS_PROMPT = """You are analyzing reasoning traces from an AI model making choices between two options.

**Scenario**: The model was asked to choose between:
- Option A: {option_a_label}
- Option B: {option_b_label}

Below are reasoning traces from two different conditions. Analyze what's different between them.

## Condition 1 Traces
{condition_a_traces}

## Condition 2 Traces
{condition_b_traces}

## Analysis Task

Compare the reasoning patterns between Condition 1 and Condition 2:

1. **Choice Distribution**: What choices were made in each condition?
2. **Reasoning Themes**: What reasoning patterns or themes appear in each condition?
3. **Key Differences**: What are the main differences in how the model reasons between conditions?
4. **Factors Emphasized**: What factors does the model emphasize differently between conditions?

Return your analysis as JSON:
{{
    "condition_1_choices": {{"A": <count>, "B": <count>}},
    "condition_2_choices": {{"A": <count>, "B": <count>}},
    "condition_1_themes": ["theme1", "theme2", ...],
    "condition_2_themes": ["theme1", "theme2", ...],
    "key_differences": ["difference1", "difference2", ...],
    "factors_condition_1": ["factor1", "factor2", ...],
    "factors_condition_2": ["factor1", "factor2", ...],
    "summary": "Brief 1-2 sentence summary of the main difference"
}}"""


def format_traces(traces: List[dict], max_traces: int = 10) -> str:
    """Format traces for the prompt."""
    if len(traces) > max_traces:
        traces = random.sample(traces, max_traces)

    formatted = []
    for i, t in enumerate(traces, 1):
        # Truncate very long reasoning
        reasoning = t["reasoning"]
        if len(reasoning) > 800:
            reasoning = reasoning[:800] + "..."
        formatted.append(f"[Trace {i}] Choice: {t['choice']}\nReasoning: {reasoning}")

    return "\n\n".join(formatted)


def build_prompt(case: dict, max_traces_per_condition: int = 10) -> str:
    """Build the analysis prompt for a single case."""
    condition_a_formatted = format_traces(
        case["condition_a_traces"], max_traces=max_traces_per_condition
    )
    condition_b_formatted = format_traces(
        case["condition_b_traces"], max_traces=max_traces_per_condition
    )

    return ANALYSIS_PROMPT.format(
        option_a_label=case["option_a_label"],
        option_b_label=case["option_b_label"],
        condition_a_traces=condition_a_formatted,
        condition_b_traces=condition_b_formatted,
    )


# =============================================================================
# Analysis Functions
# =============================================================================


async def analyze_single_case(
    client: httpx.AsyncClient,
    case: dict,
    case_idx: int,
    model: str = DEFAULT_MODEL,
    max_traces_per_condition: int = 10,
) -> dict:
    """Analyze a single case and return the result."""
    prompt = build_prompt(case, max_traces_per_condition=max_traces_per_condition)
    messages = [{"role": "user", "content": prompt}]

    try:
        response_text = await call_openrouter_async(
            client, messages, model=model, max_tokens=2000
        )

        # Parse JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) >= 2:
                response_text = parts[1]

        response_text = response_text.strip()

        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            # Try fixing common JSON issues
            import re

            fixed = re.sub(r",(\s*[}\]])", r"\1", response_text)
            try:
                analysis = json.loads(fixed)
            except json.JSONDecodeError:
                analysis = {
                    "error": "JSON parse error",
                    "raw_response": response_text[:500],
                }

        return {
            "case_idx": case_idx,
            "success": "error" not in analysis,
            "analysis": analysis,
        }

    except Exception as e:
        return {
            "case_idx": case_idx,
            "success": False,
            "analysis": {"error": str(e)},
        }


async def analyze_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    case: dict,
    case_idx: int,
    model: str,
    max_traces_per_condition: int,
) -> dict:
    """Analyze with semaphore-controlled concurrency."""
    async with semaphore:
        return await analyze_single_case(
            client, case, case_idx, model, max_traces_per_condition
        )


async def analyze_all_cases_async(
    cases: List[dict],
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 20,
    max_traces_per_condition: int = 10,
) -> List[dict]:
    """Analyze all cases with async concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    print(f"Analyzing {len(cases)} cases with {max_concurrent} concurrent requests...")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [
            analyze_with_semaphore(
                semaphore, client, case, idx, model, max_traces_per_condition
            )
            for idx, case in enumerate(cases)
        ]

        # Process with progress reporting
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)

            if (i + 1) % 10 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {i + 1}/{len(tasks)} ({rate:.1f} cases/sec)")

    # Sort by original index
    results.sort(key=lambda r: r["case_idx"])
    return results


def analyze_all_cases(
    cases: List[dict],
    model: str = DEFAULT_MODEL,
    max_concurrent: int = 20,
    max_traces_per_condition: int = 10,
) -> List[dict]:
    """Sync wrapper for async analysis."""
    return asyncio.run(
        analyze_all_cases_async(cases, model, max_concurrent, max_traces_per_condition)
    )


# =============================================================================
# Cost Estimation
# =============================================================================


def estimate_cost(cases: List[dict], model: str = DEFAULT_MODEL) -> dict:
    """Estimate the cost of analyzing all cases."""
    # Estimate tokens per case
    avg_traces_a = sum(len(c["condition_a_traces"]) for c in cases) / len(cases)
    avg_traces_b = sum(len(c["condition_b_traces"]) for c in cases) / len(cases)

    # Rough token estimates
    prompt_base_tokens = 300  # Template
    tokens_per_trace = 250  # Average reasoning length
    output_tokens = 500  # Analysis output

    input_per_case = (
        prompt_base_tokens + (avg_traces_a + avg_traces_b) * tokens_per_trace
    )
    # Cap based on max_traces_per_condition
    input_per_case = min(input_per_case, prompt_base_tokens + 20 * tokens_per_trace)

    total_input = input_per_case * len(cases)
    total_output = output_tokens * len(cases)

    pricing = MODEL_PRICING.get(model, MODEL_PRICING[DEFAULT_MODEL])

    return {
        "num_cases": len(cases),
        "model": model,
        "avg_traces_per_condition": (avg_traces_a + avg_traces_b) / 2,
        "estimated_input_tokens": int(total_input),
        "estimated_output_tokens": int(total_output),
        "estimated_cost": (total_input / 1_000_000) * pricing["input"]
        + (total_output / 1_000_000) * pricing["output"],
        "pricing_per_million": pricing,
    }


# =============================================================================
# I/O Functions
# =============================================================================


def load_cases(filepath: str) -> tuple[dict, List[dict]]:
    """Load cases from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data.get("metadata", {}), data.get("cases", [])


def save_analysis(
    cases: List[dict],
    results: List[dict],
    metadata: dict,
    analysis_metadata: dict,
    output_path: str,
):
    """Save analysis results to JSON file."""
    # Merge case data with analysis results
    output_cases = []
    for case, result in zip(cases, results):
        output_case = {
            **case,  # Original case data
            "analysis": result.get("analysis", {}),
            "analysis_success": result.get("success", False),
        }
        output_cases.append(output_case)

    output_data = {
        "original_metadata": metadata,
        "analysis_metadata": {
            **analysis_metadata,
            "analyzed_at": datetime.now().isoformat(),
            "n_successful": sum(1 for r in results if r.get("success")),
            "n_failed": sum(1 for r in results if not r.get("success")),
        },
        "cases": output_cases,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved analysis to {output_path}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze differences between reasoning trace conditions"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSON file with paired traces",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file for analysis results",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use for analysis (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Randomly sample this many cases (default: all)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=10,
        help="Maximum traces per condition to include in prompt (default: 10)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Only estimate cost, don't run analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading cases from {args.input}...")
    metadata, cases = load_cases(args.input)
    print(f"Loaded {len(cases)} cases")

    # Sample if requested
    if args.num_samples and args.num_samples < len(cases):
        random.seed(args.seed)
        cases = random.sample(cases, args.num_samples)
        print(f"Sampled {len(cases)} cases (seed={args.seed})")

    # Estimate cost
    if args.estimate_cost:
        estimate = estimate_cost(cases, args.model)
        print("\n" + "=" * 60)
        print("COST ESTIMATE")
        print("=" * 60)
        print(f"Cases to analyze: {estimate['num_cases']}")
        print(f"Model: {estimate['model']}")
        print(f"Avg traces per condition: {estimate['avg_traces_per_condition']:.1f}")
        print("\nEstimated tokens:")
        print(f"  Input:  {estimate['estimated_input_tokens']:,}")
        print(f"  Output: {estimate['estimated_output_tokens']:,}")
        print(f"\nEstimated cost: ${estimate['estimated_cost']:.4f}")
        return

    # Determine output path
    output_path = args.output
    if not output_path:
        input_stem = Path(args.input).stem
        output_path = f"{input_stem}_analysis.json"

    # Run analysis
    print(f"\nUsing model: {args.model}")
    results = analyze_all_cases(
        cases,
        model=args.model,
        max_concurrent=args.max_concurrent,
        max_traces_per_condition=args.max_traces,
    )

    # Save results
    analysis_metadata = {
        "model": args.model,
        "num_samples": args.num_samples,
        "max_traces_per_condition": args.max_traces,
        "seed": args.seed,
    }
    save_analysis(cases, results, metadata, analysis_metadata, output_path)

    # Print summary
    n_success = sum(1 for r in results if r.get("success"))
    print(f"\nAnalysis complete: {n_success}/{len(results)} successful")


if __name__ == "__main__":
    main()
