# models.py

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..results import PreferenceGraph


class UtilityModel(ABC):
    """
    Abstract base class for utility models that learn preferences from pairwise comparisons.

    Now stateful, with a 'unparseable_mode' attribute controlling how to handle unparseable responses.
    """

    def __init__(
        self,
        unparseable_mode: str,
        comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
        system_message: str,
        with_reasoning: bool,
        **kwargs,
    ):
        """
        Initialize a UtilityModel.

        Args:
            unparseable_mode: How to handle unparseable responses (can be "skip", "random", or "distribution")
            comparison_prompt_generator: Callable function that takes (option_A_dict, option_B_dict) and returns a prompt string
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing
            **kwargs: Additional arguments specific to each utility model implementation
        """
        # Validate unparseable_mode
        valid_modes = ["skip", "random", "distribution"]
        if unparseable_mode not in valid_modes:
            raise ValueError(
                f"unparseable_mode must be one of {valid_modes}, got '{unparseable_mode}'"
            )

        # Store required arguments as attributes
        self.unparseable_mode = unparseable_mode
        self.comparison_prompt_generator = comparison_prompt_generator
        self.system_message = system_message
        self.with_reasoning = with_reasoning

    @abstractmethod
    async def fit(
        self, graph: "PreferenceGraph", agent: Any
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the utility model to the preference data.

        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons

        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        graph: "PreferenceGraph",
        utilities: Dict[Any, Dict[str, float]],
        edge_indices: List[Tuple[Any, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate the model's goodness-of-fit on the given edges.

        Args:
            graph: PreferenceGraph object containing the preference data
            utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            edge_indices: List of (option_A_id, option_B_id) tuples to evaluate on

        Returns:
            Dictionary containing evaluation metrics (e.g. log_loss, accuracy)
        """
        pass

    def process_responses(
        self,
        graph: "PreferenceGraph",
        responses: Dict[int, List[str]],
        parsed_responses: Dict[int, List[str]],
        prompt_idx_to_key: Dict[int, Tuple[Any, Any, str]],
        reasoning_results: Optional[Dict[int, List[Optional[str]]]] = None,
        reasoning_summaries: Optional[Dict[int, List[Optional[str]]]] = None,
    ) -> List[Dict]:
        """
        Convert raw responses into probabilities of preferring A over B.

        The logic is equivalent to the original PreferenceGraph.process_responses,
        **but** now we incorporate `unparseable_mode` handling:
           - "skip": ignore unparseable
           - "random": randomly choose A or B for unparseable
           - "distribution": treat unparseable as [0.5, 0.5]

        Args:
            graph: The PreferenceGraph containing options and edges
            responses: Dict mapping prompt_idx to list of K raw responses
            parsed_responses: Dict mapping prompt_idx to list of K parsed responses ('A', 'B', or 'unparseable')
            prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)

        Returns:
            A list of preference data dictionaries ready for `graph.add_edges`, where each entry has:
              - option_A
              - option_B
              - probability_A
              - aux_data (with counts, original responses, etc.)
        """
        # Group raw responses by pair (in the original orientation)
        pair_data = {}  # (A_id, B_id) -> data structure

        for prompt_idx, response_list in responses.items():
            A_id, B_id, direction = prompt_idx_to_key[prompt_idx]
            parsed_list = parsed_responses[prompt_idx]  # The K parsed responses
            reasoning_list = (
                reasoning_results.get(prompt_idx, []) if reasoning_results else []
            )
            reasoning_summary_list = (
                reasoning_summaries.get(prompt_idx, []) if reasoning_summaries else []
            )
            # Use the orientation as-is (A_id, B_id)
            pair_key = (A_id, B_id)
            if pair_key not in pair_data:
                pair_data[pair_key] = {
                    "option_A": graph.options_by_id[A_id],
                    "option_B": graph.options_by_id[B_id],
                    "original_responses": [],
                    "flipped_responses": [],
                    "original_parsed": [],
                    "flipped_parsed": [],
                    "original_reasoning": [],
                    "flipped_reasoning": [],
                    "original_reasoning_summaries": [],
                    "flipped_reasoning_summaries": [],
                }

            # Extract content from LLMResponse objects for storage
            from ..llm_agent import LLMResponse

            content_list = [
                r.content if isinstance(r, LLMResponse) else r for r in response_list
            ]

            # We store the raw and parsed responses in separate buckets
            if direction == "original":
                pair_data[pair_key]["original_responses"].extend(content_list)
                pair_data[pair_key]["original_parsed"].extend(parsed_list)
                pair_data[pair_key]["original_reasoning_summaries"].extend(
                    reasoning_summary_list
                )
                if reasoning_list:
                    pair_data[pair_key]["original_reasoning"].extend(reasoning_list)
            else:  # 'flipped'
                pair_data[pair_key]["flipped_responses"].extend(content_list)
                pair_data[pair_key]["flipped_parsed"].extend(parsed_list)
                pair_data[pair_key]["flipped_reasoning_summaries"].extend(
                    reasoning_summary_list
                )
                if reasoning_list:
                    pair_data[pair_key]["flipped_reasoning"].extend(reasoning_list)

        # Now, we convert each pair's responses into a probability P(A)
        preference_data = []
        rng = random.Random(
            42
        )  # or some other seed for stable 'random' in "random" mode

        for (A_id, B_id), data in pair_data.items():
            # Instead of just counting 'A' or 'B', we keep track of distributions:
            # e.g. 'A' -> (1,0), 'B' -> (0,1), 'unparseable' -> depends on unparseable_mode.
            dist_list = []  # Will store (a_val, b_val) for each response

            # A small helper for flipping:
            # if direction == 'flipped' and user picks 'A', that means B in original orientation
            # if direction == 'flipped' and user picks 'B', that means A in original orientation
            def add_to_dist_list(parsed_char, is_flipped=False):
                """
                Add an appropriate (A, B) distribution to dist_list based on the parse,
                flipping if needed.
                """
                if parsed_char == "A":
                    if not is_flipped:
                        dist_list.append((1.0, 0.0))
                    else:
                        dist_list.append((0.0, 1.0))
                elif parsed_char == "B":
                    if not is_flipped:
                        dist_list.append((0.0, 1.0))
                    else:
                        dist_list.append((1.0, 0.0))
                else:
                    # unparseable
                    if self.unparseable_mode == "skip":
                        # do nothing -> skip
                        pass
                    elif self.unparseable_mode == "random":
                        # randomly pick A or B
                        if rng.random() < 0.5:
                            # pick A
                            if not is_flipped:
                                dist_list.append((1.0, 0.0))
                            else:
                                dist_list.append((0.0, 1.0))
                        else:
                            # pick B
                            if not is_flipped:
                                dist_list.append((0.0, 1.0))
                            else:
                                dist_list.append((1.0, 0.0))
                    elif self.unparseable_mode == "distribution":
                        # treat as 50/50
                        dist_list.append((0.5, 0.5))

            # Original ordering
            for parsed in data["original_parsed"]:
                add_to_dist_list(parsed, is_flipped=False)

            # Flipped ordering
            for parsed in data["flipped_parsed"]:
                add_to_dist_list(parsed, is_flipped=True)

            # Summarize results
            total_A = sum(d[0] for d in dist_list)
            total_B = sum(d[1] for d in dist_list)
            total_responses = len(dist_list)

            if total_responses > 0:
                probability_A = total_A / (
                    total_A + total_B
                )  # or total_A / total_responses if we interpret it differently
                aux_data = {
                    "count_A": total_A,  # This might be fractional now
                    "count_B": total_B,  # Also might be fractional
                    "total_responses": total_responses,
                    "original_responses": data["original_responses"],
                    "flipped_responses": data["flipped_responses"],
                    "original_parsed": data["original_parsed"],
                    "flipped_parsed": data["flipped_parsed"],
                    "unparseable_mode": self.unparseable_mode,
                }

                # Helper to check if list has any non-None values
                def has_content(lst):
                    return lst and any(x is not None for x in lst)

                # Add reasoning traces if available (full reasoning from reasoning models)
                if has_content(data.get("original_reasoning")):
                    aux_data["original_reasoning"] = data["original_reasoning"]
                if has_content(data.get("flipped_reasoning")):
                    aux_data["flipped_reasoning"] = data["flipped_reasoning"]

                # Add reasoning summaries only if there are actual summaries
                if has_content(data.get("original_reasoning_summaries")):
                    aux_data["original_reasoning_summaries"] = data[
                        "original_reasoning_summaries"
                    ]
                if has_content(data.get("flipped_reasoning_summaries")):
                    aux_data["flipped_reasoning_summaries"] = data[
                        "flipped_reasoning_summaries"
                    ]

                entry = {
                    "option_A": data["option_A"],
                    "option_B": data["option_B"],
                    "probability_A": probability_A,
                    "aux_data": aux_data,
                }
                preference_data.append(entry)
            else:
                # if total_responses == 0, that means everything was "skip"
                # or no valid responses were found
                pass

        return preference_data
