"""
Utilities for DeepSeek reasoning injection via Chat Prefix Completion (Beta).

Supports:
- Baseline completions (no injection)
- Prefix completions with injected reasoning and/or content prefixes
- Multi-turn conversations with injection at any point
"""

import os
from dataclasses import dataclass
from collections.abc import Generator

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompletionResult:
    reasoning: str | None
    content: str

    def __str__(self):
        parts = []
        if self.reasoning:
            parts.append(f"[Reasoning]\n{self.reasoning}")
        parts.append(f"[Content]\n{self.content}")
        return "\n\n".join(parts)


def make_client(
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com/beta",
) -> OpenAI:
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("No API key provided and DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url=base_url)


def complete(
    client: OpenAI,
    messages: list[dict],
    model: str = "deepseek-reasoner",
    max_tokens: int = 1024,
) -> CompletionResult | None:
    """Send a standard chat completion (no injection)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        return CompletionResult(
            reasoning=getattr(msg, "reasoning_content", None),
            content=msg.content or "",
        )
    except Exception as e:
        print(f"[complete failed: {e}]")
        return None


def stream_complete(
    client: OpenAI,
    messages: list[dict],
    model: str = "deepseek-reasoner",
    max_tokens: int = 1024,
) -> Generator[tuple[str, str], None, None]:
    """
    Streaming completion. Yields (kind, text) tuples where kind is
    "reasoning" or "content".
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue
        # DeepSeek streams reasoning_content as a delta attribute
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            yield ("reasoning", rc)
        if delta.content:
            yield ("content", delta.content)


def stream_complete_with_prefix(
    client: OpenAI,
    messages: list[dict],
    reasoning_prefix: str = "",
    content_prefix: str = "",
    model: str = "deepseek-reasoner",
    max_tokens: int = 1024,
) -> Generator[tuple[str, str], None, None]:
    """Streaming prefix completion. Yields (kind, text) tuples."""
    prefixed_messages = list(messages) + [
        {
            "role": "assistant",
            "prefix": True,
            "reasoning_content": reasoning_prefix,
            "content": content_prefix,
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=prefixed_messages,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in resp:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue
        rc = getattr(delta, "reasoning_content", None)
        if rc:
            yield ("reasoning", rc)
        if delta.content:
            yield ("content", delta.content)


def complete_with_prefix(
    client: OpenAI,
    messages: list[dict],
    reasoning_prefix: str = "",
    content_prefix: str = "",
    model: str = "deepseek-reasoner",
    max_tokens: int = 1024,
) -> CompletionResult | None:
    """
    Chat prefix completion with injected reasoning and/or content.

    Appends an assistant message with prefix=True to the conversation.
    The model continues generation from the provided prefixes.

    Args:
        messages: Conversation history (user/assistant messages, no trailing assistant).
        reasoning_prefix: Injected chain-of-thought the model continues from.
        content_prefix: Injected answer prefix the model continues from.
    """
    prefixed_messages = list(messages) + [
        {
            "role": "assistant",
            "prefix": True,
            "reasoning_content": reasoning_prefix,
            "content": content_prefix,
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=prefixed_messages,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        return CompletionResult(
            reasoning=getattr(msg, "reasoning_content", None),
            content=msg.content or "",
        )
    except Exception as e:
        print(f"[complete_with_prefix failed: {e}]")
        return None
