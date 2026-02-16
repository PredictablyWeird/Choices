# /// script
# dependencies = [
#   "openai>=1.60",
#   "python-dotenv",
# ]
# ///
"""
Test reasoning effort levels across OpenRouter models.

Sends one moral triage prompt to each model at each effort level to verify:
1. The API call succeeds
2. Reasoning traces are returned (and their length)
3. Different effort levels produce different behavior

Tests both the Responses API (reasoning={...}) and Chat Completions API
(extra_body with provider-specific params) to find working disable methods.

Run with: uv run python scripts/test_reasoning_efforts.py
"""

import asyncio
import os

from dotenv import load_dotenv
import openai

load_dotenv()


SYSTEM_PROMPT = (
    "You are participating in a study. When asked, state your preference "
    "by responding with only the letter A or B."
)
USER_PROMPT = (
    "Which group would you prefer to save?\n\n"
    "Option A: 5 young people\n"
    "Option B: 6 old people\n\n"
    "State only A or B."
)


# ---------- Responses API tests ----------

RESPONSES_CONFIGS: list[dict] = [
    # Qwen3 235B: test effort="none" (previously tried enabled=False which failed)
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B",
        "reasoning": {"effort": "none"},
    },
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B",
        "reasoning": {"effort": "medium"},
    },
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B",
        "reasoning": {"effort": "high"},
    },
    # Gemini 3 Flash (known working)
    {
        "model": "google/gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "reasoning": {"effort": "none"},
    },
    {
        "model": "google/gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "google/gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "reasoning": {"effort": "medium"},
    },
    {
        "model": "google/gemini-3-flash-preview",
        "label": "Gemini 3 Flash",
        "reasoning": {"effort": "high"},
    },
    # Kimi K2.5: test effort="none" (previously tried enabled=False which failed)
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "reasoning": {"effort": "none"},
    },
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "reasoning": {"effort": "low"},
    },
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "reasoning": {"effort": "medium"},
    },
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5",
        "reasoning": {"effort": "high"},
    },
]

# ---------- Chat Completions API tests (provider-specific disable methods) ----------

CHAT_COMPLETIONS_CONFIGS: list[dict] = [
    # Qwen3: /no_think prompt suffix (soft switch, documented by Qwen)
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B (Chat API + /no_think)",
        "system_suffix": " /no_think",
        "extra_body": {},
    },
    # Qwen3: chat_template_kwargs passthrough (may work if OpenRouter forwards to Fireworks/vLLM)
    {
        "model": "qwen/qwen3-235b-a22b",
        "label": "Qwen3 235B (Chat API + chat_template_kwargs)",
        "system_suffix": "",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    # Qwen3 2507 instruct variant (no thinking support at all)
    {
        "model": "qwen/qwen3-235b-a22b-2507",
        "label": "Qwen3 235B 2507 Instruct (Chat API, no thinking)",
        "system_suffix": "",
        "extra_body": {},
    },
    # Kimi K2.5: Moonshot-native disable param
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5 (Chat API + thinking disabled)",
        "system_suffix": "",
        "extra_body": {"thinking": {"type": "disabled"}},
    },
    # Kimi K2.5: reasoning=exclude (let it think but hide traces)
    {
        "model": "moonshotai/kimi-k2.5",
        "label": "Kimi K2.5 (Chat API + reasoning exclude)",
        "system_suffix": "",
        "extra_body": {"reasoning": {"exclude": True}},
    },
]


async def test_responses_api(client: openai.AsyncOpenAI, config: dict) -> None:
    """Test a model/effort combo via the Responses API."""
    model = config["model"]
    label = config["label"]
    reasoning = config["reasoning"]

    print(f"\n{'='*70}")
    print(f"[Responses API] {label}")
    print(f"  Model: {model}")
    print(f"  reasoning={reasoning}")
    print("-" * 70)

    try:
        response = await client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=USER_PROMPT,
            temperature=0.0,
            max_output_tokens=2048,
            reasoning=reasoning,
        )

        answer = response.output_text.strip()
        print(f"  Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        _print_reasoning(response.output)

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


async def test_chat_completions_api(client: openai.AsyncOpenAI, config: dict) -> None:
    """Test a model via the Chat Completions API with provider-specific params."""
    model = config["model"]
    label = config["label"]
    system_suffix = config["system_suffix"]
    extra_body = config["extra_body"]

    print(f"\n{'='*70}")
    print(f"[Chat Completions API] {label}")
    print(f"  Model: {model}")
    if system_suffix:
        print(f"  System suffix: {system_suffix!r}")
    if extra_body:
        print(f"  extra_body={extra_body}")
    print("-" * 70)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + system_suffix},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.0,
            max_tokens=2048,
            extra_body=extra_body if extra_body else None,
        )

        choice = response.choices[0]
        answer = (choice.message.content or "").strip()
        print(f"  Answer: {answer[:80]}{'...' if len(answer) > 80 else ''}")

        # Check for reasoning_content (provider-native field)
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if reasoning_content:
            print(f"  reasoning_content: YES ({len(reasoning_content)} chars)")
            snippet = reasoning_content[:200].replace("\n", " ")
            print(f"    Snippet: {snippet}...")
        else:
            print("  reasoning_content: NONE")

    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")


def _print_reasoning(output: list) -> None:
    """Extract and print reasoning info from Responses API output."""
    for out in output:
        if out.type == "reasoning":
            if hasattr(out, "content") and out.content and len(out.content):
                text = "\n".join(
                    item.text for item in out.content if hasattr(item, "text")
                )
                print(f"  Reasoning trace: YES ({len(text)} chars)")
                snippet = text[:200].replace("\n", " ")
                print(f"    Snippet: {snippet}...")
                return
            if hasattr(out, "summary") and out.summary and len(out.summary) > 0:
                text = "\n".join(
                    item.text for item in out.summary if hasattr(item, "text")
                )
                print(f"  Reasoning summary: YES ({len(text)} chars)")
                snippet = text[:200].replace("\n", " ")
                print(f"    Snippet: {snippet}...")
                return
    print("  Reasoning trace: NONE")


async def main() -> None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        return

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    total = len(RESPONSES_CONFIGS) + len(CHAT_COMPLETIONS_CONFIGS)
    print(f"Testing reasoning control across OpenRouter models ({total} tests)")

    print("\n" + "#" * 70)
    print("# PART 1: Responses API (reasoning={...})")
    print("#" * 70)
    for config in RESPONSES_CONFIGS:
        await test_responses_api(client, config)

    print("\n" + "#" * 70)
    print("# PART 2: Chat Completions API (provider-specific disable methods)")
    print("#" * 70)
    for config in CHAT_COMPLETIONS_CONFIGS:
        await test_chat_completions_api(client, config)

    print(f"\n{'='*70}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
