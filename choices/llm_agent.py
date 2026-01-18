import asyncio
import imghdr
import json
import os
import random
import string
import time
from abc import ABC, abstractmethod
from base64 import b64encode
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import openai
from dotenv import load_dotenv
from litellm import acompletion as litellm_acompletion
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


# =================== Response Interface ===================
@dataclass
class LLMResponse:
    """
    Standard response interface for all LLM agents.

    Attributes:
        content: The main response content from the LLM.
        reasoning: Optional full reasoning traces (e.g., from deepseek reasoning models).
        reasoning_summary: Optional summary of the LLM's reasoning (e.g., from o1/o3 models).
        reasoning_type: The type of reasoning used. One of:
            - "NO_REASONING": No reasoning was requested
            - "REASONING_BEFORE": Reasoning appears before the answer
            - "REASONING_AFTER": Reasoning appears after the answer
    """

    content: Optional[str]
    reasoning: Optional[str] = None
    reasoning_summary: Optional[str] = None
    reasoning_type: Literal["NO_REASONING", "REASONING_BEFORE", "REASONING_AFTER"] = (
        "NO_REASONING"
    )


# =================== Utils ===================
def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return b64encode(image_file.read()).decode("utf-8")


class LLMAgent(ABC):
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        retry_times: int = 3,
        accepts_system_message: bool = True,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_outputs = "Sorry, I can not satisfy that request."
        self.retry_times = retry_times
        self.accepts_system_message = accepts_system_message

    @abstractmethod
    def _completions(self, messages) -> str:
        raise NotImplementedError

    def _completions_batch(self, messages) -> List[str]:
        raise NotImplementedError

    async def _async_completions(self, messages) -> str:
        raise NotImplementedError

    async def _completions_stream(self, messages: List[Dict]) -> str:
        raise NotImplementedError

    async def completions_stream(self, messages: List[Dict]) -> str:
        try:
            response = self._completions_stream(messages)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    def completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = self._completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    def completions_batch(self, messages: List[Dict], **kwargs) -> List[str]:
        try:
            response = self._completions_batch(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")

    async def async_completions(self, messages: List[Dict], **kwargs) -> str:
        try:
            response = await self._async_completions(messages, **kwargs)
            return response
        except Exception as e:
            raise Exception(f"Exception: {str(e)}")


class OpenAIAgent(LLMAgent):
    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: str = "gpt-4o-mini",
        concurrency_limit: int = 100,
    ):
        super().__init__(temperature, max_tokens)
        self.model = model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.completions_kwargs = {
            # "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        self.concurrency_limit = concurrency_limit

    def _preprocess_messages(self, messages: List[Dict]) -> List[Dict]:
        for message in messages:
            if image_path := message.get("image_path"):
                image_data = _encode_image(image_path)
                image_type = (
                    imghdr.what(image_path) or "jpeg"
                )  # Default to 'jpeg' if type can't be determined
                message["content"] = [
                    {"type": "text", "text": message["content"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{image_data}"
                        },
                    },
                ]
                del message["image_path"]
        return messages

    def _completions(self, messages: List[Dict]) -> str:
        messages = self._preprocess_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, **self.completions
        )
        response = response.choices[0].message.content
        return response

    def _completions_batch(self, messages: List[List[Dict]], **kwargs) -> List[str]:
        # Create logs directory if it doesn't exist
        os.makedirs("logs/input_file", exist_ok=True)
        os.makedirs("logs/output_file", exist_ok=True)

        # Generate random ID for the input file
        random_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        input_filename = f"{self.model}_{random_id}.jsonl"
        input_filepath = os.path.join("logs/input_file", input_filename)

        # Prepare input file
        with open(input_filepath, "w") as f:
            for i, message in enumerate(messages):
                request = {
                    "custom_id": f"request-{random_id}-{i + 1}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": message,
                        **self.completions_kwargs,
                    },
                }
                json.dump(request, f)
                f.write("\n")

        batch_input_file = self.client.files.create(
            file=open(input_filepath, "rb"), purpose="batch"
        )

        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        start_time = time.time()
        batch_status = None
        while batch_status is None or batch_status.status not in [
            "completed",
            "failed",
            "cancelled",
        ]:
            batch_status = self.client.batches.retrieve(batch.id)
            print(
                f"Time elapsed: {(time.time() - start_time):.2f} seconds. Current status: {batch_status.status}"
            )
            time.sleep(10)

        if batch_status.status != "completed":
            raise Exception(
                f"Batch failed with status: {batch_status.status} | Error: {batch_status.errors}"
            )

        output_file_content = self.client.files.content(
            batch_status.output_file_id
        ).text
        output_filepath = os.path.join("logs/output_file", f"output_{input_filename}")

        with open(output_filepath, "w") as f:
            print("Writing output to file:", output_filepath)
            f.write(output_file_content)

        results = []
        for line in output_file_content.splitlines():
            data = json.loads(line)
            results.append(
                data.get("response", {})
                .get("body", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", None)
            )

        return results

    async def _async_completions(self, messages: List[Dict]) -> str:
        response = await self.async_client.chat.completions.create(
            model=self.model, messages=messages, **self.completions_kwargs
        )
        return response.choices[0].message.content

    async def async_completions_batch(
        self,
        messages: List[List[Dict]],
        timeout: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> List[str]:
        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0}
        results = {}

        async def process_message(message_idx):
            message = messages[message_idx]
            retry_delay = timeout
            current_timeout = timeout
            max_retries = 5

            for attempt in range(max_retries):
                try:
                    async with semaphore:
                        # Use asyncio.wait_for to enforce hard timeout at asyncio level
                        completion = await asyncio.wait_for(
                            self.async_client.chat.completions.create(
                                model=self.model,
                                messages=message,
                                max_tokens=self.max_tokens,
                                temperature=self.temperature,
                                timeout=current_timeout,
                            ),
                            timeout=current_timeout,
                        )
                    response = completion.choices[0].message.content.strip()
                    break  # Exit the retry loop on success
                except asyncio.TimeoutError:
                    counts["timeouts"] += 1
                    if attempt == max_retries - 1:
                        if verbose:
                            print(
                                f"Max retries exceeded (timeouts) for prompt {message_idx} "
                                f"after {max_retries} attempts."
                            )
                        response = None  # final
                    else:
                        current_timeout *= 2
                        if verbose:
                            print(
                                f"Timeout after {current_timeout // 2}s for prompt {message_idx}, attempt {attempt + 1}. "
                                f"Increasing timeout to {current_timeout}s..."
                            )
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Max retries exceeded for message index: {message_idx}")
                        print(e)
                        response = None  # Or handle as you see fit
                    else:
                        print(
                            f"Error occurred for message index: {message_idx}. Retrying in {retry_delay} seconds."
                        )
                        print(e)
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff

            results[message_idx] = LLMResponse(content=response)

        # Create actual tasks (not just coroutines) so we can cancel them if needed
        tasks = [
            asyncio.create_task(process_message(message_idx))
            for message_idx in range(len(messages))
        ]

        # Wrap in try-except to properly clean up tasks on error or cancellation
        try:
            for f in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
                await f  # Wait for each task to complete
        except (Exception, asyncio.CancelledError):
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish (with cancellation)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise the original exception

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")

        return [results[i] for i in range(len(messages))]

    async def _completions_stream(self, messages: List):
        # messages = self.system + messages
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, **self.completions_kwargs
        )

        for chunk in stream:
            if (text := chunk.choices[0].delta.content) is not None:
                yield text


class ReasoningAgent(OpenAIAgent):
    """
    Agent for models with reasoning capabilities (e.g., o1, gpt-5, deepseek).
    Use this agent when reasoning_effort is configured in the model config.
    """

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: str = "gpt-5",
        model_type: str = "openai",
        concurrency_limit: int = 100,
        timeout: int = 5,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ):
        super().__init__(temperature, max_tokens)
        self.model = (
            model[len("openrouter/") :] if model.startswith("openrouter/") else model
        )  # For reasoning models we aren't using litellm
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Infer API key and base URL from model_type
        if model_type == "openai":
            base_url = "https://api.openai.com/v1"
            api_key = os.getenv("OPENAI_API_KEY")
        elif model_type == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")
        else:
            raise ValueError(
                f"ReasoningAgent does not support model_type '{model_type}'. "
                "Must be 'openai' or 'openrouter'."
            )

        self.async_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.concurrency_limit = concurrency_limit
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity
        self.max_retries = 5
        self.use_jitter = True
        self.max_delay = 10.0
        self.base_delay = 1.0

    async def async_completions(
        self, messages: List[List[Dict]], verbose: bool = True, **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]
            retry_delay = 2.0
            current_timeout = self.timeout
            response = None
            reasoning = None
            reasoning_summary = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    try:
                        completion_kwargs = {
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_tokens,
                        }
                        # Add reasoning_effort and text_verbosity for OpenAI reasoning models (o1, gpt-5)
                        if self.reasoning_effort:
                            completion_kwargs["reasoning"] = self.reasoning_effort
                        if self.text_verbosity:
                            completion_kwargs["text"] = self.text_verbosity
                        # print(f"Making request for message {message_idx}")
                        # Use asyncio.wait_for to enforce hard timeout at asyncio level
                        completion_res = await asyncio.wait_for(
                            self.async_client.responses.create(
                                model=self.model,
                                instructions=message[0]["content"],
                                input=message[1]["content"],
                                **completion_kwargs,
                            ),
                            timeout=current_timeout,
                        )
                        # print(f"Request for message {message_idx} completed")
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(
                                    f"Max retries (timeouts) reached for message index {message_idx}."
                                )
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(
                                    f"Max retries (errors) reached for message index {message_idx}."
                                )
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(
                                    f"Sleeping {sleep_for:.1f}s before retry (error backoff)..."
                                )
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response
                    response = completion_res.output_text
                    # Access reasoning traces and message content based on model_type
                    for out in completion_res.output:
                        if out.type == "reasoning":
                            if self.model_type == "openai":
                                # OpenAI models provide reasoning summaries
                                if out.summary and len(out.summary) > 0:
                                    # Concatenate all summary elements in case there are multiple
                                    reasoning_summary = "\n".join(
                                        item.text
                                        for item in out.summary
                                        if hasattr(item, "text")
                                    )
                                else:
                                    reasoning_summary = None
                            elif self.model_type == "openrouter":
                                # OpenRouter reasoning models (e.g., deepseek) provide full traces
                                # Some models (e.g., grok-4.1-fast) may only provide summaries
                                if out.content and len(out.content):
                                    # Concatenate all content elements in case there are multiple
                                    reasoning = "\n".join(
                                        item.text
                                        for item in out.content
                                        if hasattr(item, "text")
                                    )
                                # Fallback to summary if content is not available
                                elif out.summary and len(out.summary) > 0:
                                    # Concatenate all summary elements in case there are multiple
                                    reasoning_summary = "\n".join(
                                        item.text
                                        for item in out.summary
                                        if hasattr(item, "text")
                                    )
                    # Success means we don't have to retry again
                    break

            # For reasoning models, the reasoning is internal (captured in reasoning/reasoning_summary),
            # but the response text is just the answer. So reasoning_type should be NO_REASONING
            # for parsing purposes - we don't need to look for "Answer: X" patterns.
            results[message_idx] = LLMResponse(
                content=response,
                reasoning=reasoning,
                reasoning_summary=reasoning_summary,
                reasoning_type="NO_REASONING",
            )

        # Create actual tasks (not just coroutines) so we can cancel them if needed
        tasks = [asyncio.create_task(process_message(i)) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # Wrap in try-except to properly clean up tasks on error or cancellation
        try:
            for coro in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="LLM calls"
            ):
                await coro
        except (Exception, asyncio.CancelledError):
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish (with cancellation)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise the original exception

        return [results[i] for i in range(len(messages))]


class LiteLLMAgent:
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        extra_body: Optional[Dict] = None,
        reasoning_effort: Optional[str] = None,
        text_verbosity: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message
        self.extra_body = extra_body
        self.reasoning_effort = reasoning_effort
        self.text_verbosity = text_verbosity

        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter

    async def async_completions(
        self, messages: List[List[Dict]], verbose: bool = True, **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]

            current_timeout = self.base_timeout
            retry_delay = self.base_delay
            response = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    # if verbose:
                    #     print(
                    #         f"[Attempt {attempt+1}/{self.max_retries}] "
                    #         f"Message index {message_idx}, timeout={current_timeout:.1f}s"
                    #     )

                    try:
                        completion_kwargs = {
                            "model": self.model,
                            "messages": message,
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature,
                            "timeout": current_timeout,
                        }
                        # Add reasoning_effort and text_verbosity for OpenAI reasoning models (o1, gpt-5)
                        if self.reasoning_effort:
                            completion_kwargs["reasoning"] = self.reasoning_effort
                        if self.text_verbosity:
                            completion_kwargs["text"] = self.text_verbosity
                        # Add extra_body if it exists (for OpenRouter reasoning parameter, etc.)
                        if self.extra_body:
                            completion_kwargs["extra_body"] = self.extra_body

                        # Use asyncio.wait_for to enforce hard timeout at asyncio level
                        # (litellm's internal timeout may not be respected by all providers)
                        completion_res = await asyncio.wait_for(
                            litellm_acompletion(**completion_kwargs),
                            timeout=current_timeout,
                        )
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(
                                    f"Max retries (timeouts) reached for message index {message_idx}."
                                )
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(
                                    f"Max retries (errors) reached for message index {message_idx}."
                                )
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(
                                    f"Sleeping {sleep_for:.1f}s before retry (error backoff)..."
                                )
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response
                    response = completion_res.choices[0].message.content.strip()
                    break  # done with retries

            results[message_idx] = LLMResponse(content=response)

        # Create actual tasks (not just coroutines) so we can cancel them if needed
        tasks = [asyncio.create_task(process_message(i)) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # Wrap in try-except to properly clean up tasks on error or cancellation
        try:
            for coro in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="LLM calls"
            ):
                await coro
        except (Exception, asyncio.CancelledError):
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish (with cancellation)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise the original exception

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        return [results[i] for i in range(len(messages))]


# Uses the OpenAI client and makes sure to call completions rather than chat. Mostly a copy of LiteLLMAgent
class BaseAgent:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        concurrency_limit: int = 100,
        accepts_system_message: bool = True,
        max_retries: int = 5,
        base_timeout: float = 5.0,
        base_delay: float = 1.0,
        max_delay: float = 10.0,
        use_jitter: bool = True,
        treat_as_chat_model: bool = False,
        extra_body: Optional[Dict] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.concurrency_limit = concurrency_limit
        self.accepts_system_message = accepts_system_message
        self.extra_body = extra_body

        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.use_jitter = use_jitter

        self.treat_as_chat_model = treat_as_chat_model

        # Initialize OpenAI client (which handles various providers)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages list to a single prompt string."""

        if self.treat_as_chat_model:
            return messages

        error_msg = "base model should only contain a user message (which gets converted to raw text).\nIf you want to use the base model as a chat model try treat_as_chat_model=True"

        assert len(messages) == 1, error_msg

        msg = messages[0]

        role = msg["role"]
        assert role == "user", error_msg

        content = msg["content"]
        return content

    async def async_completions(
        self, messages: List[List[Dict]], verbose: bool = True, **kwargs
    ) -> List[str]:
        """
        Returns a list of LLM responses, in order.
        Uses a semaphore to limit concurrency, and tqdm_asyncio for progress.
        """

        semaphore = asyncio.Semaphore(self.concurrency_limit)
        counts = {"timeouts": 0, "errors": 0}
        results = {}

        async def process_message(message_idx: int):
            """
            Attempts to process a single message up to `max_retries` times.
            On generic exceptions, sleeps with exponential backoff and optional jitter.
            On timeout, doubles the request timeout without sleeping.
            """
            message = messages[message_idx]
            # Convert messages to prompt for completions endpoint
            prompt = self._messages_to_prompt(message)

            current_timeout = self.base_timeout
            retry_delay = self.base_delay
            response = None

            for attempt in range(self.max_retries):
                # Acquire the semaphore before making the LLM call
                async with semaphore:
                    # if verbose:
                    #     print(
                    #         f"[Attempt {attempt+1}/{self.max_retries}] "
                    #         f"Message index {message_idx}, timeout={current_timeout:.1f}s"
                    #     )

                    try:
                        if self.treat_as_chat_model:
                            completion_kwargs = {
                                "model": self.model,
                                "messages": prompt,
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "timeout": current_timeout,
                            }
                            # Add extra_body if it exists (for OpenRouter reasoning parameter, etc.)
                            if self.extra_body:
                                completion_kwargs["extra_body"] = self.extra_body

                            # Use asyncio.wait_for to enforce hard timeout at asyncio level
                            completion_res = await asyncio.wait_for(
                                self.client.chat.completions.create(
                                    **completion_kwargs
                                ),
                                timeout=current_timeout,
                            )

                        else:
                            completion_kwargs = {
                                "model": self.model,
                                "prompt": prompt,
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature,
                                "timeout": current_timeout,
                            }
                            # Add extra_body if it exists (for OpenRouter reasoning parameter, etc.)
                            if self.extra_body:
                                completion_kwargs["extra_body"] = self.extra_body

                            # Use asyncio.wait_for to enforce hard timeout at asyncio level
                            completion_res = await asyncio.wait_for(
                                self.client.completions.create(**completion_kwargs),
                                timeout=current_timeout,
                            )
                    except asyncio.TimeoutError:
                        counts["timeouts"] += 1

                        if verbose:
                            print(
                                f"[Timeout] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}. Timed out after {current_timeout:.1f}s."
                            )
                        if attempt == self.max_retries - 1:
                            response = None  # no more retries
                            if verbose:
                                print(
                                    f"Max retries (timeouts) reached for message index {message_idx}."
                                )
                        else:
                            current_timeout *= 2.0

                        continue  # next attempt

                    except Exception as e:
                        counts["errors"] += 1

                        if verbose:
                            print(
                                f"[Error] Attempt {attempt + 1}/{self.max_retries} "
                                f"for message index {message_idx}: {e}"
                            )
                        if attempt == self.max_retries - 1:
                            response = None
                            if verbose:
                                print(
                                    f"Max retries (errors) reached for message index {message_idx}."
                                )
                        else:
                            # Sleep with exponential backoff
                            sleep_for = retry_delay
                            if self.use_jitter:
                                sleep_for += random.uniform(0, 1)
                            if verbose:
                                print(
                                    f"Sleeping {sleep_for:.1f}s before retry (error backoff)..."
                                )
                            await asyncio.sleep(sleep_for)
                            retry_delay = min(retry_delay * 2.0, self.max_delay)

                        continue  # next attempt

                    # Success: parse the response (using .text for completions, not .message.content)
                    response = completion_res.choices[0].text.strip()
                    break  # done with retries

            results[message_idx] = LLMResponse(content=response)

        # Create actual tasks (not just coroutines) so we can cancel them if needed
        tasks = [asyncio.create_task(process_message(i)) for i in range(len(messages))]

        # Use tqdm_asyncio to track progress as tasks finish
        # Wrap in try-except to properly clean up tasks on error or cancellation
        try:
            for coro in tqdm_asyncio.as_completed(
                tasks, total=len(tasks), desc="LLM calls"
            ):
                await coro
        except (Exception, asyncio.CancelledError):
            # Cancel all pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait for all tasks to finish (with cancellation)
            await asyncio.gather(*tasks, return_exceptions=True)
            raise  # Re-raise the original exception

        if verbose:
            print(f"Number of timeouts: {counts['timeouts']}")
            print(f"Number of generic errors: {counts['errors']}")

        return [results[i] for i in range(len(messages))]
