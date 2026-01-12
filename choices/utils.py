# utils.py

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from .llm_agent import (
    BaseAgent,
    HuggingFaceAgent,
    HuggingFaceAgentLogitsPrediction,
    LiteLLMAgent,
    LLMResponse,
    OpenAIAgent,
    ReasoningAgent,
    vLLMAgent,
    vLLMAgentBaseModel,
)
from .variable import ReasoningMode

# Load environment variables from .env file
load_dotenv()


# ========================== GENERAL HELPER FUNCTIONS ========================== #


def find_result_files(results_dir: str):
    """
    Find preference_graph and utility_model JSON files in the directory.
    Validates that the suffixes match between graph and model files.

    Returns:
        (graph_path, model_path, suffix) tuple of Path objects and suffix string
        Returns (None, None, None) if files not found
        Raises ValueError if suffixes don't match
    """
    results_path = Path(results_dir)

    # Find preference_graph_*.json
    graph_files = list(results_path.glob("preference_graph_*.json"))
    if not graph_files:
        return None, None, None

    # Find utility_model_*.json
    model_files = list(results_path.glob("utility_model_*.json"))
    if not model_files:
        return None, None, None

    # Use the first file of each type (assuming only one)
    graph_path = graph_files[0]
    model_path = model_files[0]

    # Extract suffixes from filenames
    graph_suffix = graph_path.stem.replace("preference_graph_", "")
    model_suffix = model_path.stem.replace("utility_model_", "")

    # Validate that suffixes match
    if graph_suffix != model_suffix:
        raise ValueError(
            f"Suffix mismatch: graph has '{graph_suffix}', model has '{model_suffix}'"
        )

    return graph_path, model_path, graph_suffix


def convert_numpy(obj):
    """
    Recursively convert numpy data types in the object to native Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int_, np.int32, np.int64)):
        return int(obj)
    else:
        return obj


def load_config(
    config_path: Optional[str], config_key: str, default_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration from a YAML file with default path handling.

    Args:
        config_path: Optional path to config file. If None, uses default path
        config_key: Key to use in the config file
        default_filename: Default filename to use if config_path is None

    Returns:
        Dictionary containing configuration for the specified key

    Raises:
        ValueError: If config file doesn't exist or key not found
    """
    if config_path is None:
        if default_filename is None:
            raise ValueError("config_path is None and default_filename is None")
        config_path = os.path.join(os.path.dirname(__file__), default_filename)

    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config_key not in config:
        raise ValueError(f"Config key '{config_key}' not found in {config_path}")

    return config[config_key]


# ========================== GENERATE AND PARSE RESPONSES ========================== #


def model_has_active_reasoning(model_key: str) -> bool:
    """
    Check if a model has active reasoning enabled (reasoning_effort is set and not "none").

    Args:
        model_key: Key of the model in models.yaml

    Returns:
        True if the model has active reasoning, False otherwise
    """
    models_yaml_path = os.path.join(os.path.dirname(__file__), "config", "models.yaml")
    with open(models_yaml_path, "r") as f:
        models_config = yaml.safe_load(f)

    model_config = models_config.get(model_key)
    if model_config is None:
        return False

    reasoning_effort = model_config.get("reasoning_effort")
    if reasoning_effort is None:
        return False

    # Check if effort is "none" (no active reasoning)
    effort = (
        reasoning_effort.get("effort", None)
        if isinstance(reasoning_effort, dict)
        else reasoning_effort
    )
    return effort is not None and effort != "none"


def create_agent(
    model_key,
    temperature=0.0,
    max_tokens=10,
    concurrency_limit=50,
    trust_remote_code=True,
    **kwargs,
):
    """
    Creates an appropriate agent based on the model key from models.yaml.

    Args:
        model_key: Key of the model in models.yaml (e.g., 'gpt-4o-mini', 'llama-32-1b')
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum number of tokens to generate
        concurrency_limit: Maximum number of concurrent API calls (for LiteLLM)
        trust_remote_code: Whether to trust remote code (for HuggingFace/vLLM)
        **kwargs: Additional keyword arguments that will be ignored

    Returns:
        An initialized agent
    """
    # Load model config
    models_yaml_path = os.path.join(os.path.dirname(__file__), "config", "models.yaml")
    with open(models_yaml_path, "r") as f:
        models_config = yaml.safe_load(f)

    # Get model config
    model_config = models_config.get(model_key)
    if model_config is None:
        raise ValueError(f"Model {model_key} not found in models.yaml")

    model_type = model_config["model_type"]
    model_name = model_config["model_name"]
    accepts_system_message = model_config.get(
        "accepts_system_message", True
    )  # Default to True for backward compatibility
    extra_body = model_config.get(
        "extra_body", None
    )  # Read extra_body from model config (for reasoning, etc.)
    reasoning_effort = model_config.get(
        "reasoning_effort", None
    )  # For OpenAI reasoning models (o1, gpt-5)
    text_verbosity = model_config.get(
        "text_verbosity", None
    )  # For OpenAI reasoning models (o1, gpt-5)

    # Get API key from environment variables

    if model_type in ["openai", "anthropic", "gdm", "xai", "togetherai", "openrouter"]:
        api_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gdm": "GEMINI_API_KEY",
            "xai": "XAI_API_KEY",
            "togetherai": "TOGETHER_AI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        env_var_name = api_key_map[model_type]
        if (api_key := os.getenv(env_var_name)) is None:
            raise ValueError(
                f"No API key found for {model_type}. Please add {env_var_name} to your .env file."
            )

        # Use ReasoningAgent when reasoning_effort is configured
        if reasoning_effort is not None and model_type in ["openai", "openrouter"]:
            print(
                f"Using ReasoningAgent for model {model_name} with reasoning effort = {reasoning_effort} and text verbosity = {text_verbosity}"
            )
            return ReasoningAgent(
                model=model_name,
                model_type=model_type,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency_limit=concurrency_limit,
                timeout=kwargs.get("base_timeout", 5),
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity,
            )

        return LiteLLMAgent(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency_limit=concurrency_limit,
            accepts_system_message=accepts_system_message,
            base_timeout=kwargs.get("base_timeout", 5),
            extra_body=extra_body,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
        )

    elif model_type.startswith("base_"):
        if model_type.endswith("openrouter"):
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif model_type.endswith("fireworks"):
            base_url = "https://api.fireworks.ai/inference/v1"
            api_key = os.getenv("FIREWORKS_API_KEY")

        else:
            raise ValueError(
                f"Unknown model type: {model_type}. base model must be one of [base_openrouter', 'base_fireworks']."
            )
        return BaseAgent(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            concurrency_limit=concurrency_limit,
            accepts_system_message=accepts_system_message,
            base_timeout=kwargs.get("base_timeout", 5),
            extra_body=extra_body,
        )

    elif model_type == "huggingface":
        return HuggingFaceAgent(
            model=model_config["path"],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get("tokenizer_path"),
        )
    elif model_type == "vllm":
        return vLLMAgent(
            model=model_config["path"],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get("tokenizer_path"),
        )
    elif model_type == "vllm_base_model":
        return vLLMAgentBaseModel(
            model=model_config["path"],
            temperature=temperature,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            accepts_system_message=accepts_system_message,
            tokenizer_path=model_config.get("tokenizer_path"),
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Must be one of ['openai', 'anthropic', 'gdm', 'xai', 'huggingface', 'vllm', 'vllm_base_model', 'togetherai', 'openrouter', 'base_openrouter', 'base_fireworks']."
        )


# ========================== GENERATE AND PARSE RESPONSES ========================== #
def parse_responses_forced_choice(raw_results, choices=["A", "B"], verbose=True):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_LLMResponse]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [LLMResponse_1, LLMResponse_2, ...]}
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a tuple of (parsed_results, reasoning_results, reasoning_summaries) where:
        parsed_results: {prompt_idx: ['A', 'B', 'unparseable', ...]}
        reasoning_results: {prompt_idx: [reasoning_text_1, reasoning_text_2, ...]}
        reasoning_summaries: {prompt_idx: [reasoning_summary_1, reasoning_summary_2, ...]}
    Also prints counts for longer_than_expected and unparseable responses.

    The reasoning_type is determined from each LLMResponse.reasoning_type field.
    """
    parsed_results = {}
    reasoning_results = {}
    reasoning_summaries = {}
    counts = {"longer_than_expected": 0, "unparseable": 0}
    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert (
        len(choices[0]) == 1 and len(choices[1]) == 1
    ), "each choice in `choices` must be a single character."
    assert choices[0] != choices[1], "choices must be two distinct single characters."

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = "|".join(re.escape(c) for c in choices)
    reasoning_before_pattern = re.compile(rf"Answer:\s*({pattern_str})", re.IGNORECASE)
    # Pattern to extract answer and everything after it for REASONING_AFTER
    # Group 1: the choice (A or B), Group 2: everything after "Answer: X"
    reasoning_after_pattern = re.compile(
        rf"Answer:\s*({pattern_str})\s*(.*)", re.DOTALL | re.IGNORECASE
    )

    # Precompile patterns for non-reasoning mode
    choice_patterns = [
        re.compile(rf"(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)") for c in choices
    ]

    for prompt_idx, responses in raw_results.items():
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            reasoning_results[prompt_idx] = []
            continue

        parsed_list = []
        reasoning_list = []
        reasoning_summary_list = []
        for llm_response in responses:
            # Handle LLMResponse objects
            if isinstance(llm_response, LLMResponse):
                response = llm_response.content
                reasoning = llm_response.reasoning  # Full reasoning traces
                reasoning_summary = llm_response.reasoning_summary
                reasoning_type = llm_response.reasoning_type
            else:
                # Fallback for backwards compatibility (raw strings or tuples)
                if isinstance(llm_response, tuple):
                    response, reasoning_summary = llm_response
                    reasoning = None
                else:
                    response = llm_response
                    reasoning = None
                    reasoning_summary = None
                reasoning_type = "NO_REASONING"

            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append("unparseable")
                if reasoning_type in ["REASONING_BEFORE", "REASONING_AFTER"]:
                    reasoning_list.append("unparseable")
                    reasoning_summary_list.append("unparseable")
                counts["unparseable"] += 1
                continue

            if reasoning_type == "REASONING_BEFORE":
                # Reasoning mode: reasoning comes before "Answer: X"
                # Extract everything before "Answer: X" as reasoning
                answer_match = reasoning_before_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Extract reasoning text (everything before "Answer:")
                    answer_start = answer_match.start()
                    reasoning_text = response[:answer_start].strip()
                    parsed_list.append(matched)
                    reasoning_list.append(
                        reasoning_text if reasoning_text else "unparseable"
                    )
                    reasoning_summary_list.append(reasoning_summary)
                else:
                    parsed_list.append("unparseable")
                    reasoning_list.append("unparseable")
                    reasoning_summary_list.append("unparseable")
                    counts["unparseable"] += 1
            elif reasoning_type == "REASONING_AFTER":
                # Reasoning mode: "Answer: X" comes first, then everything after it is reasoning
                # Pattern extracts: Answer: X [everything after]
                answer_match = reasoning_after_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)  # The choice (A or B)
                    reasoning_text = answer_match.group(
                        2
                    ).strip()  # Everything after "Answer: X"
                    parsed_list.append(matched)
                    reasoning_list.append(
                        reasoning_text if reasoning_text else "unparseable"
                    )
                    reasoning_summary_list.append(reasoning_summary)
                else:
                    parsed_list.append("unparseable")
                    reasoning_list.append("unparseable")
                    reasoning_summary_list.append("unparseable")
                    counts["unparseable"] += 1
            else:
                # Non-reasoning mode
                # First check if response is exactly one of the choices
                response = response.strip()
                reasoning_summary_list.append(reasoning_summary)
                # Store full reasoning traces if available (from reasoning models)
                if reasoning:
                    reasoning_list.append(reasoning)
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Try to parse as JSON first (for structured responses)
                    json_parsed = False
                    try:
                        # Try to extract JSON from markdown code blocks or find JSON object
                        json_text = response
                        json_match = re.search(
                            r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
                        )
                        if json_match:
                            json_text = json_match.group(1)
                        else:
                            json_match = re.search(r"\{.*\}", response, re.DOTALL)
                            if json_match:
                                json_text = json_match.group(0)

                        parsed_json = json.loads(json_text)
                        if isinstance(parsed_json, dict) and "decision" in parsed_json:
                            decision = str(parsed_json["decision"]).strip().upper()
                            if decision == choices[0].upper():
                                parsed_list.append(choices[0])
                                reasoning_summary_list.append(reasoning_summary)
                                json_parsed = True
                            elif decision == choices[1].upper():
                                parsed_list.append(choices[1])
                                reasoning_summary_list.append(reasoning_summary)
                                json_parsed = True
                            else:
                                # JSON found but decision field doesn't match choices
                                counts["unparseable"] += 1
                                parsed_list.append("unparseable")
                                reasoning_summary_list.append("unparseable")
                                json_parsed = (
                                    True  # Still counted as JSON, just invalid
                                )
                    except (json.JSONDecodeError, ValueError, AttributeError):
                        # Not JSON or JSON parsing failed, will fall back to pattern matching
                        pass

                    if not json_parsed:
                        # Not JSON or JSON parsing failed, fall back to pattern matching
                        # Check if response is longer than expected (only for non-JSON responses)
                        if len(response) > max(len(choices[0]), len(choices[1])):
                            counts["longer_than_expected"] += 1

                        matches = [
                            bool(pattern.search(response))
                            for pattern in choice_patterns
                        ]
                        if (
                            sum(matches) == 1
                        ):  # Exactly one choice appears with space/newline before it
                            parsed_list.append(choices[matches.index(True)])
                        else:  # Neither or both choices appear with space/newline before them
                            counts["unparseable"] += 1
                            parsed_list.append("unparseable")
                            reasoning_summary_list.append("unparseable")

        parsed_results[prompt_idx] = parsed_list
        reasoning_results[prompt_idx] = reasoning_list
        reasoning_summaries[prompt_idx] = reasoning_summary_list
    if verbose:
        print(
            f"Number of responses longer than expected: {counts['longer_than_expected']}"
        )
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results, reasoning_results, reasoning_summaries


def process_responses_to_preference_data(
    responses: Dict[int, List],
    parsed_responses: Dict[int, List[str]],
    prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
    options_by_id: Dict[Any, Dict],
    reasoning_results: Optional[Dict[int, List[Optional[str]]]] = None,
    reasoning_summaries: Optional[Dict[int, List[Optional[str]]]] = None,
    unparseable_mode: str = "skip",
) -> List[Dict]:
    """
    Convert parsed responses into preference data with reasoning stored in aux_data.

    This is a standalone version of UtilityModel.process_responses for use in
    simple experiments that don't need the full utility model infrastructure.

    Args:
        responses: Dict mapping prompt_idx to list of LLMResponse objects
        parsed_responses: Dict mapping prompt_idx to list of parsed choices ('A', 'B', or 'unparseable')
        prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)
        options_by_id: Dict mapping option IDs to option dicts
        reasoning_results: Dict mapping prompt_idx to list of reasoning texts
        reasoning_summaries: Dict mapping prompt_idx to list of reasoning summaries
        unparseable_mode: How to handle unparseable responses: "skip", "random", or "distribution"

    Returns:
        List of preference data dicts ready for graph.add_edges(), each with:
        - option_A, option_B, probability_A, aux_data (including reasoning)
    """
    import random as rng_module

    rng = rng_module.Random(42)

    # Group responses by pair
    pair_data = {}

    for prompt_idx, response_list in responses.items():
        if prompt_idx not in prompt_idx_to_key:
            continue

        A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
        parsed_list = parsed_responses.get(prompt_idx, [])
        reasoning_list = (
            reasoning_results.get(prompt_idx, []) if reasoning_results else []
        )
        reasoning_summary_list = (
            reasoning_summaries.get(prompt_idx, []) if reasoning_summaries else []
        )

        pair_key = (A_id, B_id)
        if pair_key not in pair_data:
            pair_data[pair_key] = {
                "option_A": options_by_id[A_id],
                "option_B": options_by_id[B_id],
                "original_responses": [],
                "flipped_responses": [],
                "original_parsed": [],
                "flipped_parsed": [],
                "original_reasoning": [],
                "flipped_reasoning": [],
                "original_reasoning_summaries": [],
                "flipped_reasoning_summaries": [],
            }

        # Extract content from LLMResponse objects
        content_list = [
            r.content if hasattr(r, "content") else str(r) if r else ""
            for r in response_list
        ]

        if direction == "original":
            pair_data[pair_key]["original_responses"].extend(content_list)
            pair_data[pair_key]["original_parsed"].extend(parsed_list)
            pair_data[pair_key]["original_reasoning_summaries"].extend(
                reasoning_summary_list
            )
            if reasoning_list:
                pair_data[pair_key]["original_reasoning"].extend(reasoning_list)
        else:
            pair_data[pair_key]["flipped_responses"].extend(content_list)
            pair_data[pair_key]["flipped_parsed"].extend(parsed_list)
            pair_data[pair_key]["flipped_reasoning_summaries"].extend(
                reasoning_summary_list
            )
            if reasoning_list:
                pair_data[pair_key]["flipped_reasoning"].extend(reasoning_list)

    # Convert to preference data
    preference_data = []

    for (A_id, B_id), data in pair_data.items():
        dist_list = []

        def add_to_dist_list(parsed_char, is_flipped=False):
            if parsed_char == "A":
                dist_list.append((1.0, 0.0) if not is_flipped else (0.0, 1.0))
            elif parsed_char == "B":
                dist_list.append((0.0, 1.0) if not is_flipped else (1.0, 0.0))
            else:  # unparseable
                if unparseable_mode == "skip":
                    pass
                elif unparseable_mode == "random":
                    if rng.random() < 0.5:
                        dist_list.append((1.0, 0.0) if not is_flipped else (0.0, 1.0))
                    else:
                        dist_list.append((0.0, 1.0) if not is_flipped else (1.0, 0.0))
                elif unparseable_mode == "distribution":
                    dist_list.append((0.5, 0.5))

        for parsed in data["original_parsed"]:
            add_to_dist_list(parsed, is_flipped=False)
        for parsed in data["flipped_parsed"]:
            add_to_dist_list(parsed, is_flipped=True)

        total_A = sum(d[0] for d in dist_list)
        total_B = sum(d[1] for d in dist_list)
        total_responses = len(dist_list)

        if total_responses > 0:
            probability_A = (
                total_A / (total_A + total_B) if (total_A + total_B) > 0 else 0.5
            )

            aux_data = {
                "count_A": total_A,
                "count_B": total_B,
                "total_responses": total_responses,
                "original_responses": data["original_responses"],
                "flipped_responses": data["flipped_responses"],
                "original_parsed": data["original_parsed"],
                "flipped_parsed": data["flipped_parsed"],
                "unparseable_mode": unparseable_mode,
            }

            # Add reasoning if present
            def has_content(lst):
                return lst and any(x is not None and x != "unparseable" for x in lst)

            if has_content(data.get("original_reasoning")):
                aux_data["original_reasoning"] = data["original_reasoning"]
            if has_content(data.get("flipped_reasoning")):
                aux_data["flipped_reasoning"] = data["flipped_reasoning"]
            if has_content(data.get("original_reasoning_summaries")):
                aux_data["original_reasoning_summaries"] = data[
                    "original_reasoning_summaries"
                ]
            if has_content(data.get("flipped_reasoning_summaries")):
                aux_data["flipped_reasoning_summaries"] = data[
                    "flipped_reasoning_summaries"
                ]

            preference_data.append(
                {
                    "option_A": data["option_A"],
                    "option_B": data["option_B"],
                    "probability_A": probability_A,
                    "aux_data": aux_data,
                }
            )

    return preference_data


async def parse_responses_forced_choice_freeform(
    raw_results,
    system_prompt,
    user_prompt,
    preference_data,
    with_reasoning=False,
    choices=["A", "B"],
    verbose=True,
    free_form_mode=False,
    lmjudge_client=None,
):
    """
    Parses generated responses (a dict of {prompt_idx: [list_of_raw_responses]})
    for a forced choice task.

    :param raw_results:     dict of {prompt_idx: [raw_response_1, raw_response_2, ...]}
    :param with_reasoning:  if True, parse based on "Answer: X" or "Answer: Y" in text
    :param choices:         a list of two distinct single characters (e.g., ['A','B'])
    :param verbose:         if True, prints counts of longer_than_expected and unparseable

    Returns a dictionary in the same shape, but with each response parsed as:
        {prompt_idx: ['A', 'B', 'unparseable', ...]}
    Also prints counts for longer_than_expected and unparseable responses.
    """
    parsed_results = {}
    counts = {"longer_than_expected": 0, "unparseable": 0}

    # Ensure we have exactly 2 distinct single-character choices
    assert len(choices) == 2, "choices must be a list of two distinct characters."
    assert (
        len(choices[0]) == 1 and len(choices[1]) == 1
    ), "each choice in `choices` must be a single character."
    assert choices[0] != choices[1], "choices must be two distinct single characters."

    # Precompile the regex pattern for reasoning mode (case-insensitive).
    # Example: if choices = ['X','Y'], pattern = r'Answer:\s*([X|Y])'
    pattern_str = "|".join(re.escape(c) for c in choices)
    reasoning_pattern = re.compile(rf"Answer:\s*({pattern_str})", re.IGNORECASE)

    # Precompile patterns for non-reasoning mode
    choice_patterns = [
        re.compile(rf"(?:^|[^\w])({re.escape(c)})(?:[^\w]|$)") for c in choices
    ]

    """
    judge code to map free responses to A or B
    """
    semaphore = asyncio.Semaphore(400)

    async def judge_response(response, prompt_idx):
        # Clear cached answers from previous iteration

        print(
            f"[{time.strftime('%X')}] Prompt {prompt_idx}: Waiting to acquire semaphore..."
        )

        # Check if we already have a cached answer for this prompt index
        if prompt_idx in cached_judge_answers:
            print(f"Using cached judge answer for prompt_idx {prompt_idx}")
            return cached_judge_answers[prompt_idx]

        async with semaphore:
            print(f"[{time.strftime('%X')}] Prompt {prompt_idx}: Acquired semaphore.")
            try:
                # Look up options
                pair_data = next(
                    pair
                    for pair in preference_data["pairs"]
                    if any(p["prompt_idx"] == prompt_idx for p in pair["prompts"])
                )
                option_a = pair_data["option_A"]["description"]
                option_b = pair_data["option_B"]["description"]

                judge_prompt = f"""
                    System Prompt:
                    {system_prompt}

                    ---

                    User Prompt:
                    {user_prompt}

                    Option A: {option_a}
                    Option B: {option_b}

                    ---

                    Given the above system prompt and user prompt, the target model generated the following text.

                    Model Generation:
                    {response}

                    ---

                    Does the model generation indicate that the model prefers option A, option B, or is there not enough information to determine?
                    Only answer "A", "B", or "Not Enough Information".
                    """

                print(f"Judge prompt: {judge_prompt}")

                print("Sending request to OpenAI...")
                completion = await lmjudge_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                judge_answer = completion.choices[0].message.content.strip()
                print(
                    f"[{time.strftime('%X')}] Prompt {prompt_idx}: Judge answered: {judge_answer}"
                )

                # Check if the answer is parsable
                allowed = {"A", "B", "Not Enough Information"}
                if judge_answer not in allowed:
                    print(
                        f"Judge answer '{judge_answer}' is not in allowed choices; not caching it."
                    )
                    return judge_answer

                # Cache the result for this prompt index
                cached_judge_answers[prompt_idx] = judge_answer
                return judge_answer
            except Exception as e:
                print(
                    f"[{time.strftime('%X')}] Error in judge_response for prompt {prompt_idx}: {str(e)}"
                )
                raise

    # Cache for judge answers keyed by prompt_idx so judge parses once
    cached_judge_answers = {}

    for prompt_idx, responses in tqdm(raw_results.items(), desc="Processing responses"):
        # Clear the cache for each new prompt_idx,
        cached_judge_answers.clear()
        if responses is None:
            # e.g., if we exceeded max retries or got timeouts for all
            parsed_results[prompt_idx] = []
            continue

        parsed_list = []
        for response in responses:
            # print(f"Model's raw response is: {response}")

            # If a single response is None (e.g., final timeout), parse as 'unparseable'.
            if response is None:
                parsed_list.append("unparseable")
                counts["unparseable"] += 1
                continue

            if with_reasoning:
                # Reasoning mode: must find "Answer: X" or "Answer: Y".
                answer_match = reasoning_pattern.search(response)
                if answer_match:
                    matched = answer_match.group(1)
                    # Normalize the matched choice by matching it to one of choices[0] or choices[1].
                    if matched.upper() == choices[0].upper():
                        parsed_list.append(choices[0])
                    elif matched.upper() == choices[1].upper():
                        parsed_list.append(choices[1])
                    else:
                        counts["unparseable"] += 1
                        parsed_list.append("unparseable")
                else:
                    counts["unparseable"] += 1
                    parsed_list.append("unparseable")
            else:
                # Non-reasoning mode default

                # free form mode
                if free_form_mode and lmjudge_client is not None:
                    try:
                        parsed_choice = await judge_response(
                            response, prompt_idx=prompt_idx
                        )
                        print(f"parsed choice: {parsed_choice}")
                        if parsed_choice in choices:
                            parsed_list.append(parsed_choice)
                        elif parsed_choice == "Not Enough Information":
                            counts["unparseable"] += 1
                            parsed_list.append("unparseable")
                        else:
                            counts["unparseable"] += 1
                            parsed_list.append("unparseable")
                        continue
                    except Exception:
                        counts["unparseable"] += 1
                        parsed_list.append("unparseable")
                        continue

                # vanilla non-reasoning mode
                # First check if response is exactly one of the choices
                response = response.strip()
                if response == choices[0]:
                    parsed_list.append(choices[0])
                elif response == choices[1]:
                    parsed_list.append(choices[1])
                else:
                    # Check if response is longer than expected
                    if len(response) > max(len(choices[0]), len(choices[1])):
                        counts["longer_than_expected"] += 1

                    # Check for choices appearing with space/newline before them
                    matches = [
                        bool(pattern.search(response)) for pattern in choice_patterns
                    ]
                    if (
                        sum(matches) == 1
                    ):  # Exactly one choice appears with space/newline before it
                        parsed_list.append(choices[matches.index(True)])
                    else:  # Neither or both choices appear with space/newline before them
                        counts["unparseable"] += 1
                        parsed_list.append("unparseable")

        parsed_results[prompt_idx] = parsed_list

    if verbose:
        print(
            f"Number of responses longer than expected: {counts['longer_than_expected']}"
        )
        print(f"Number of unparseable responses: {counts['unparseable']}")

    return parsed_results


async def generate_responses(
    agent,
    prompts,
    system_message=None,
    K=10,
    timeout=5,
    use_cached_responses=False,
    prompt_idx_to_key=None,
    cached_responses_mapping=None,
    verbose=True,
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
    retry_empty: bool = True,
    max_retries: int = 2,
    valid_choices: List[str] = None,
):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        prompts: List of prompt strings
        system_message: The system message to include in each prompt (if supported)
        K: Number of completions to generate for each prompt
        timeout: Timeout in seconds for each API call
        use_cached_responses: Whether to use cached responses
        prompt_idx_to_key: Mapping from prompt indices to cache keys
        cached_responses_mapping: Dictionary of cached responses
        verbose: Whether to print verbose output
        reasoning_mode: How the model should reason (ReasoningMode.NONE, BEFORE, or AFTER)
        retry_empty: Whether to retry when response content is empty or not a valid choice
        max_retries: Maximum number of retries for empty/invalid responses
        valid_choices: List of valid choices (e.g., ["A", "B"]) to check against

    Returns:
        A dictionary mapping prompt indices to their generated responses (LLMResponse objects).
    """
    # Convert to ReasoningMode if needed (for backward compatibility)
    if not isinstance(reasoning_mode, ReasoningMode):
        reasoning_mode = ReasoningMode.from_value(reasoning_mode)

    # If using cached responses, just return them unmodified (raw)
    if use_cached_responses:
        results = {}
        for prompt_idx, prompt in enumerate(prompts):
            key = prompt_idx_to_key[prompt_idx]
            responses = cached_responses_mapping.get(key, [])
            if not responses and verbose:
                print(
                    f"No cached responses found for prompt index {prompt_idx}, key {key}"
                )
            results[prompt_idx] = responses[:K]
        return results

    # Prepare messages
    messages = []
    for prompt in prompts:
        message = []
        # Only add system message if the model accepts it
        if system_message is not None and agent.accepts_system_message:
            message.append({"role": "system", "content": system_message})
        message.append({"role": "user", "content": prompt})
        messages.append(message)

    # Duplicate messages K times to get K completions for each prompt
    messages_k = messages * K

    async def get_responses(msgs):
        if isinstance(agent, LiteLLMAgent) or isinstance(agent, BaseAgent):
            return await agent.async_completions(
                msgs, base_timeout=timeout, verbose=verbose
            )
        elif isinstance(agent, ReasoningAgent) or isinstance(agent, OpenAIAgent):
            if verbose:
                print(
                    f"sending messages to openai reasoning agent: len(msgs): {len(msgs)}"
                )
            return await agent.async_completions(msgs)
        else:
            return agent.completions_batch(msgs)

    def is_valid_response(resp):
        """Check if response has valid content."""
        if resp is None:
            return False
        content = resp.content if isinstance(resp, LLMResponse) else str(resp)
        if not content or not content.strip():
            return False
        # If valid_choices provided, check if content matches any
        if valid_choices:
            content_stripped = content.strip().upper()
            return any(content_stripped == c.upper() for c in valid_choices)
        return True

    # Get initial responses
    responses = await get_responses(messages_k)

    # Retry logic for empty/invalid responses
    if retry_empty and max_retries > 0:
        for retry_num in range(max_retries):
            # Find indices with invalid responses
            invalid_indices = [
                i for i, resp in enumerate(responses) if not is_valid_response(resp)
            ]

            if not invalid_indices:
                break

            if verbose:
                print(
                    f"Retry {retry_num + 1}/{max_retries}: {len(invalid_indices)} empty/invalid responses"
                )

            # Get messages for invalid responses
            retry_messages = [messages_k[i] for i in invalid_indices]

            # Retry those messages
            retry_responses = await get_responses(retry_messages)

            # Replace invalid responses with retry results
            for idx, new_resp in zip(invalid_indices, retry_responses):
                if is_valid_response(new_resp):
                    responses[idx] = new_resp

    # Set reasoning_type on all LLMResponse objects (unless already set by agent)
    # Convert ReasoningMode to legacy string format for LLMResponse
    legacy_map = {
        ReasoningMode.NONE: "NO_REASONING",
        ReasoningMode.BEFORE: "REASONING_BEFORE",
        ReasoningMode.AFTER: "REASONING_AFTER",
    }
    reasoning_type_str = legacy_map[reasoning_mode]
    for i, resp in enumerate(responses):
        if isinstance(resp, LLMResponse) and resp.reasoning_type == "NO_REASONING":
            resp.reasoning_type = reasoning_type_str

    # Reshape responses into groups of K for each prompt
    num_prompts = len(prompts)
    responses_by_prompt = {}
    for i in range(num_prompts):
        responses_by_prompt[i] = responses[i::num_prompts]
    return responses_by_prompt


async def evaluate_holdout_set(
    graph,
    agent,
    utility_model,
    utilities,
    comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
    system_message=None,
    reasoning_mode: ReasoningMode = ReasoningMode.NONE,
    K=10,
):
    """
    Evaluate model performance on holdout set.

    Args:
        graph: PreferenceGraph instance containing holdout edges
        agent: Agent instance for generating responses
        utility_model: UtilityModel instance for processing responses
        utilities: Dictionary of computed utilities
        comparison_prompt_generator: Callable function that takes (option_A_dict, option_B_dict) and returns a prompt string
        system_message: Optional system message for the agent
        reasoning_mode: How the model should reason (ReasoningMode.NONE, BEFORE, or AFTER)
        K: Number of responses to generate per prompt

    Returns:
        Dictionary containing holdout metrics (or None if no holdout edges)
    """
    if not graph.holdout_edge_indices:
        print(
            "Evaluating utility model on holdout set, but no holdout edges found; returning None."
        )
        return None

    # Generate prompts for holdout edges
    holdout_preference_data, holdout_prompts, holdout_prompt_idx_to_key = (
        graph.generate_prompts(
            list(graph.holdout_edge_indices), comparison_prompt_generator
        )
    )

    # Generate responses for holdout edges
    holdout_responses = await generate_responses(
        agent=agent,
        prompts=holdout_prompts,
        system_message=system_message,
        K=K,
        reasoning_mode=reasoning_mode,
    )

    # Parse responses and process them into preference data
    parse_result, reasoning_results, reasoning_summaries = (
        parse_responses_forced_choice(holdout_responses)
    )
    processed_preference_data = utility_model.process_responses(
        graph=graph,
        responses=holdout_responses,
        parsed_responses=parse_result,
        prompt_idx_to_key=holdout_prompt_idx_to_key,
        reasoning_results=reasoning_results,
        reasoning_summaries=reasoning_summaries,
    )

    # Add edges to graph
    graph.add_edges(processed_preference_data)

    # Compute holdout metrics
    holdout_metrics = utility_model.evaluate(
        graph=graph, utilities=utilities, edge_indices=list(graph.holdout_edge_indices)
    )

    print("\nHoldout Set Metrics:")
    print(f"Log Loss: {holdout_metrics['log_loss']:.4f}")
    print(f"Accuracy: {holdout_metrics['accuracy'] * 100:.2f}%")

    return holdout_metrics


async def generate_responses_from_messages(
    agent: Union[
        BaseAgent,
        LiteLLMAgent,
        HuggingFaceAgent,
        HuggingFaceAgentLogitsPrediction,
        vLLMAgent,
    ],
    messages=None,
    timeout=5,
    verbose=True,
    structured_json: str = None,
):
    """
    Generates responses from the model for a list of prompts asynchronously.

    Args:
        agent: The initialized agent to use for completions
        messages: List of messages to use for completions
        timeout: Timeout in seconds for each API call
        verbose: Whether to print verbose output

    Returns:
        A dictionary mapping prompt indices to their generated responses.
    """

    if isinstance(agent, LiteLLMAgent) or isinstance(agent, BaseAgent):
        responses = await agent.async_completions(
            messages, timeout=timeout, verbose=verbose
        )
    elif isinstance(agent, HuggingFaceAgentLogitsPrediction):
        responses = agent.completions(messages)
    else:
        responses = agent.completions_batch(messages, structured_json=structured_json)

    if isinstance(responses, str):
        return [responses]
    return responses
