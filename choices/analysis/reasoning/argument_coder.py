"""
LLM-based argument coding for reasoning traces.

Uses async LLM calls to code each reasoning trace for the presence
of arguments defined in a codebook.
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

from .codebook import Argument, Codebook
from .extract_traces import ReasoningTrace

load_dotenv()


@dataclass
class CodedTrace:
    """
    A reasoning trace with argument coding results.

    Attributes:
        trace: The original reasoning trace
        argument_vector: List of binary codes (1=present, 0=absent, -1=error)
                        in the order of codebook arguments
        argument_codes: Dict mapping argument_id to code
    """

    trace: ReasoningTrace
    argument_vector: list[int]
    argument_codes: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "edge_key": self.trace.edge_key,
            "response_index": self.trace.response_index,
            "trace_type": self.trace.trace_type,
            "argument_vector": self.argument_vector,
            "argument_codes": self.argument_codes,
            "original_trace": self.trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodedTrace":
        """Create from dictionary."""
        trace = ReasoningTrace.from_dict(data["original_trace"])
        return cls(
            trace=trace,
            argument_vector=data["argument_vector"],
            argument_codes=data.get("argument_codes", {}),
        )

    def has_argument(self, argument_id: str) -> bool:
        """Check if a specific argument is present."""
        return self.argument_codes.get(argument_id, 0) == 1

    def has_any_argument(self, argument_ids: list[str]) -> bool:
        """Check if any of the specified arguments are present."""
        return any(self.argument_codes.get(aid, 0) == 1 for aid in argument_ids)

    def has_all_arguments(self, argument_ids: list[str]) -> bool:
        """Check if all of the specified arguments are present."""
        return all(self.argument_codes.get(aid, 0) == 1 for aid in argument_ids)


def _create_coding_prompt(argument: Argument, reasoning: str) -> str:
    """Create a prompt for the LLM to code whether an argument is present."""
    return f"""You are a research assistant coding reasoning traces for argument analysis.

ARGUMENT TO CODE: {argument.name}

DEFINITION: {argument.to_coding_description()}

TASK: Does the following reasoning trace use this argument? Answer with ONLY "YES" or "NO".

REASONING TRACE:
{reasoning}

Answer (YES or NO):"""


class ArgumentCoder:
    """
    Async argument coder using LLM calls.

    Attributes:
        codebook: The codebook defining arguments to code
        model: Model identifier for coding (default: gemini-2.0-flash via OpenRouter)
        max_concurrent: Maximum concurrent API requests
        client: AsyncOpenAI client
    """

    def __init__(
        self,
        codebook: Codebook,
        model: str = "google/gemini-2.0-flash-001",
        max_concurrent: int = 100,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize the argument coder.

        Args:
            codebook: Codebook defining arguments to code
            model: Model identifier (OpenRouter format)
            max_concurrent: Maximum concurrent API requests
            api_key: OpenRouter API key (default: from OPENROUTER_API_KEY env var)
            base_url: API base URL (default: OpenRouter)
        """
        self.codebook = codebook
        self.model = model
        self.max_concurrent = max_concurrent

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it in .env or pass api_key parameter."
            )

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def _code_single_argument(
        self,
        argument: Argument,
        reasoning: str,
        semaphore: asyncio.Semaphore,
    ) -> int:
        """Code a single argument for a single reasoning trace."""
        async with semaphore:
            try:
                prompt = _create_coding_prompt(argument, reasoning)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=50,
                )

                raw_answer = response.choices[0].message.content or ""
                answer = raw_answer.strip().strip("\"'").strip().lower()

                if answer == "yes":
                    return 1
                elif answer == "no":
                    return 0
                else:
                    print(f"Warning: Unexpected response: {repr(raw_answer)}")
                    return -1

            except Exception as e:
                print(f"Error coding argument {argument.id}: {e}")
                return -1

    async def _code_trace(
        self,
        trace: ReasoningTrace,
        semaphore: asyncio.Semaphore,
    ) -> CodedTrace:
        """Code all arguments for a single trace."""
        argument_vector = []
        argument_codes = {}

        for argument in self.codebook.arguments:
            code = await self._code_single_argument(
                argument, trace.reasoning, semaphore
            )
            argument_vector.append(code)
            argument_codes[argument.id] = code

        return CodedTrace(
            trace=trace,
            argument_vector=argument_vector,
            argument_codes=argument_codes,
        )

    async def code_traces(
        self,
        traces: list[ReasoningTrace],
        show_progress: bool = True,
    ) -> list[CodedTrace]:
        """
        Code all traces for all arguments in the codebook.

        Args:
            traces: List of reasoning traces to code
            show_progress: Whether to show progress bar

        Returns:
            List of CodedTrace objects with coding results
        """
        print(
            f"Coding {len(traces)} traces with {len(self.codebook.arguments)} arguments"
        )
        print(f"Model: {self.model}")
        print(f"Max concurrent requests: {self.max_concurrent}")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [self._code_trace(trace, semaphore) for trace in traces]

        if show_progress:
            coded_traces = await tqdm.gather(*tasks, desc="Coding traces")
        else:
            coded_traces = await asyncio.gather(*tasks)

        # Print summary statistics
        print("\nArgument frequency:")
        for i, argument in enumerate(self.codebook.arguments):
            count = sum(1 for ct in coded_traces if ct.argument_vector[i] == 1)
            pct = 100 * count / len(coded_traces) if coded_traces else 0
            print(f"  {argument.name}: {count}/{len(coded_traces)} ({pct:.1f}%)")

        return coded_traces


def save_coded_traces(
    coded_traces: list[CodedTrace],
    codebook: Codebook,
    model_used: str,
    output_path: str,
) -> None:
    """Save coded traces to JSON file."""
    import json

    output = {
        "codebook": codebook.to_dict(),
        "model_used": model_used,
        "total_traces": len(coded_traces),
        "coded_traces": [ct.to_dict() for ct in coded_traces],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved {len(coded_traces)} coded traces to {output_path}")


def load_coded_traces(input_path: str) -> tuple[list[CodedTrace], Codebook, str]:
    """
    Load coded traces from JSON file.

    Returns:
        Tuple of (coded_traces, codebook, model_used)
    """
    import json

    with open(input_path) as f:
        data = json.load(f)

    codebook = Codebook.from_dict(data["codebook"])
    model_used = data["model_used"]
    coded_traces = [CodedTrace.from_dict(ct) for ct in data["coded_traces"]]

    return coded_traces, codebook, model_used
