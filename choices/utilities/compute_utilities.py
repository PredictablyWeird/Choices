import asyncio
import copy
import itertools
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..results import (
    ExperimentOption,
    ExperimentResults,
    PreferenceGraphResults,
    UtilityModelResults,
)
from ..utils import create_agent, evaluate_holdout_set, load_config
from ..variable import AnalysisConfig
from .thurstonian import (
    ThurstonianActiveLearningUtilityModel,
)

# ===================== DEFAULT PROMPTS ===================== #


class PreferenceEdge:
    """
    A class representing a preference edge between two options.
    """

    def __init__(
        self,
        option_A: Dict[str, Any],
        option_B: Dict[str, Any],
        probability_A: float,
        aux_data: Dict[str, Any] = None,
    ):
        """
        Initialize a preference edge.

        Args:
            option_A: First option dictionary with at least an 'id' field
            option_B: Second option dictionary with at least an 'id' field
            probability_A: Probability of A being preferred over B
            aux_data: Optional dictionary of auxiliary data about this preference
        """
        # Keep options in the order given, no sorting
        self.option_A = option_A
        self.option_B = option_B
        self.probability_A = probability_A  # P(A > B)
        self.aux_data = aux_data if aux_data is not None else {}

    def __eq__(self, other: "PreferenceEdge") -> bool:
        """Two preference edges are equal if they have the same orientation of A→B."""
        if not isinstance(other, PreferenceEdge):
            return False
        return (
            self.option_A["id"] == other.option_A["id"]
            and self.option_B["id"] == other.option_B["id"]
        )

    def __hash__(self) -> int:
        """Hash based on the ordered pair (A_id, B_id)."""
        return hash((self.option_A["id"], self.option_B["id"]))

    def __repr__(self) -> str:
        return f"PreferenceEdge({self.option_A['id']} vs {self.option_B['id']}, P(A)={self.probability_A:.3f})"


class PreferenceGraph:
    """
    A class representing a graph of pairwise preferences between options.
    Handles creation of training/holdout edge sets and sampling strategies.
    """

    def __init__(
        self,
        options: List[Dict[str, Any]],
        holdout_fraction: float = 0.0,
        seed: int = 42,
    ):
        """
        Initialize a preference graph with training and holdout edge indices.

        Args:
            options: List of dictionaries, each containing at least:
                    {'id': str/int}
            holdout_fraction: Fraction of edges to hold out for evaluation
            seed: Random seed for reproducibility
        """
        self.options = options
        self.option_id_to_idx = {
            option["id"]: idx for idx, option in enumerate(options)
        }
        self.options_by_id = {opt["id"]: opt for opt in options}

        # Generate all possible edge indices as tuples
        all_edge_indices = list(
            itertools.combinations([opt["id"] for opt in options], 2)
        )

        # Split into training and holdout indices
        random.seed(seed)
        if holdout_fraction <= 0:
            self.training_edges_pool = set(all_edge_indices)
            self.holdout_edge_indices = set()
        else:
            total_edges = len(all_edge_indices)
            # Cap holdout size at min(fraction-based size, 1000)
            fraction_based_size = int(total_edges * holdout_fraction)
            holdout_size = min(fraction_based_size, 1000)

            # Randomly select holdout edges
            all_edges_shuffled = all_edge_indices.copy()
            random.shuffle(all_edges_shuffled)
            self.holdout_edge_indices = set(all_edges_shuffled[:holdout_size])
            self.training_edges_pool = set(all_edges_shuffled[holdout_size:])

        # Initialize sets for tracking actual edges in the graph
        self.training_edges = set()  # Training edges currently in graph
        self.edges = {}  # Map from edge index tuple to PreferenceEdge

        print(f"Total possible edges: {len(all_edge_indices)}")
        print(
            f"Training pool: {len(self.training_edges_pool)}, Holdout: {len(self.holdout_edge_indices)}"
        )

    @classmethod
    def load_data(cls, data: Dict[str, Any]) -> "PreferenceGraph":
        """
        Create a PreferenceGraph instance from exported data.

        Args:
            data: Dictionary containing the graph data, as exported by export_data

        Returns:
            A new PreferenceGraph instance with the loaded data
        """
        # Create instance with options
        graph = cls(options=data["options"])

        # Restore edge sets
        graph.training_edges = set(tuple(edge) for edge in data["training_edges"])
        graph.training_edges_pool = set(
            tuple(edge) for edge in data["training_edges_pool"]
        )
        graph.holdout_edge_indices = set(
            tuple(edge) for edge in data["holdout_edge_indices"]
        )

        # Restore edges
        graph.edges = {}
        for edge_key_str, edge_data in data["edges"].items():
            # Convert string edge key back to tuple
            edge_key = tuple(map(int, edge_key_str.strip("()").split(", ")))
            # Create PreferenceEdge instance
            edge = PreferenceEdge(
                option_A=edge_data["option_A"],
                option_B=edge_data["option_B"],
                probability_A=edge_data["probability_A"],
                aux_data=edge_data["aux_data"],
            )
            graph.edges[edge_key] = edge

        return graph

    def export_data(self) -> Dict[str, Any]:
        """
        Export the graph data in a JSON-serializable format.

        Returns:
            Dictionary containing all the graph data in a serializable format
        """
        return {
            "options": self.options,
            "edges": {
                str(edge_key): {
                    "option_A": edge.option_A["label"]
                    if "label" in edge.option_A
                    else edge.option_A["id"],
                    "option_B": edge.option_B["label"]
                    if "label" in edge.option_B
                    else edge.option_B["id"],
                    "probability_A": edge.probability_A,
                    "aux_data": edge.aux_data,
                }
                for edge_key, edge in self.edges.items()
            },
            "training_edges": [list(edge) for edge in self.training_edges],
            "training_edges_pool": [list(edge) for edge in self.training_edges_pool],
            "holdout_edge_indices": [list(edge) for edge in self.holdout_edge_indices],
        }

    def generate_prompts(
        self,
        edge_indices: List[Tuple[Any, Any]],
        comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
        include_flipped: bool = True,
    ) -> Tuple[List[Dict], List[str], Dict[int, Tuple]]:
        """
        Generate prompts for the given edge indices in both original and flipped ordering.

        Args:
            edge_indices: List of (option_A_id, option_B_id) tuples
            comparison_prompt_generator: Callable function that takes (option_A_dict, option_B_dict) and returns a prompt string
            include_flipped: Whether to include flipped prompts (Note: This should always be True; we only set it to False for demonstration purposes)
        Returns:
            Tuple containing:
            - preference_data: List of pair data with prompts
            - prompt_list: List of all prompts
            - prompt_idx_to_key: Mapping from prompt index to (option_A_id, option_B_id, direction)
        """
        preference_data = []
        prompt_list = []
        prompt_idx_to_key = {}
        prompt_idx = 0

        for pair_idx, (A_id, B_id) in enumerate(edge_indices):
            option_A = self.options_by_id[A_id]
            option_B = self.options_by_id[B_id]

            pair_data = {
                "pair_id": pair_idx,
                "option_A": option_A,
                "option_B": option_B,
                "prompts": [],
            }

            # Generate prompts in both directions
            if include_flipped:
                directions = ["original", "flipped"]
            else:
                directions = ["original"]

            for direction in directions:
                if direction == "original":
                    option1, option2 = option_A, option_B
                else:
                    option1, option2 = option_B, option_A

                prompt = comparison_prompt_generator(option1, option2)

                prompt_data = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "direction": direction,
                    "responses": [],
                }

                pair_data["prompts"].append(prompt_data)
                prompt_list.append(prompt)
                prompt_idx_to_key[prompt_idx] = (A_id, B_id, direction)
                prompt_idx += 1

            preference_data.append(pair_data)

        return preference_data, prompt_list, prompt_idx_to_key

    def add_edges(self, preference_data: List[Dict]) -> None:
        """
        Add multiple edges to the graph based on processed preference data.

        Args:
            preference_data: List of dictionaries containing:
                - option_A: Option A dictionary
                - option_B: Option B dictionary
                - probability_A: Probability of A being preferred over B
                - aux_data: Dictionary with auxiliary data
        """
        for data in preference_data:
            A_id = data["option_A"]["id"]
            B_id = data["option_B"]["id"]
            # Keep original orientation
            edge_index = (A_id, B_id)

            edge = PreferenceEdge(
                option_A=data["option_A"],
                option_B=data["option_B"],
                probability_A=data["probability_A"],
                aux_data=data["aux_data"],
            )

            self.edges[edge_index] = edge

            # Update training edges tracking if this was a training edge
            # Note: We need to check both orientations for training pool membership
            if edge_index in self.training_edges_pool:
                self.training_edges_pool.remove(edge_index)
                self.training_edges.add(edge_index)
            elif (B_id, A_id) in self.training_edges_pool:
                self.training_edges_pool.remove((B_id, A_id))
                self.training_edges.add(edge_index)

    def sample_regular_graph(
        self, degree: int, seed: int = None
    ) -> List[Tuple[Any, Any]]:
        """
        Sample edge indices forming a regular graph of given degree from training edges pool.

        Args:
            degree: Desired degree for each node
            seed: Random seed for reproducibility

        Returns:
            List of (option_A_id, option_B_id) tuples
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        n_options = len(self.options)
        if degree >= n_options:
            raise ValueError("Degree must be less than the number of options.")

        # Generate regular graph using indices
        G = nx.random_regular_graph(degree, n_options, seed=seed)
        initial_pairs = []

        # Map node indices to option IDs
        idx_to_id = {idx: opt["id"] for idx, opt in enumerate(self.options)}

        # Convert edges to option ID pairs and filter out holdout edges
        for i, j in G.edges():
            edge = tuple(sorted([idx_to_id[i], idx_to_id[j]]))
            if edge in self.training_edges_pool:
                initial_pairs.append(edge)

        # If we lost too many edges due to holdout filtering, sample additional edges
        target_edges = (n_options * degree) // 2
        if len(initial_pairs) < target_edges:
            remaining_edges = list(self.training_edges_pool - set(initial_pairs))
            n_additional = min(target_edges - len(initial_pairs), len(remaining_edges))
            if n_additional > 0:
                initial_pairs.extend(random.sample(remaining_edges, n_additional))

        return initial_pairs

    def sample_random_edges(
        self, n_edges: int, seed: int = None
    ) -> List[Tuple[Any, Any]]:
        """
        Sample random edge indices from training edges pool.

        Args:
            n_edges: Number of edges to sample
            seed: Random seed for reproducibility

        Returns:
            List of (option_A_id, option_B_id) tuples
        """
        if seed is not None:
            random.seed(seed)

        edges_list = list(self.training_edges_pool)
        n_edges = min(n_edges, len(edges_list))
        return random.sample(edges_list, n_edges)


def _save_example_prompt(
    graph: PreferenceGraph,
    comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
    system_message: str,
    save_path: str,
) -> None:
    """
    Save an example prompt from a random edge in the graph.

    Args:
        graph: PreferenceGraph with edges
        comparison_prompt_generator: Callable function that takes (option_A_dict, option_B_dict) and returns a prompt string
        system_message: System message to include in the example
        save_path: Directory to save the example prompt
    """
    import random

    # Get a random edge
    if not graph.training_edges_pool:
        return  # No edges to sample from

    edge_ids = list(graph.training_edges_pool)
    random_edge = random.choice(edge_ids)

    # Get the nodes
    node_a_id, node_b_id = random_edge
    node_a = graph.options_by_id[node_a_id]
    node_b = graph.options_by_id[node_b_id]

    prompt_text = comparison_prompt_generator(node_a, node_b)

    # Create full example with system message
    full_example = f"System Message:\n{system_message}\n\n{'=' * 60}\n\n{prompt_text}"

    # Save to file
    example_path = os.path.join(save_path, "example_prompt.txt")
    with open(example_path, "w") as f:
        f.write(full_example)
        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"Option A ID: {node_a_id}\n")
        f.write(f"Option B ID: {node_b_id}\n")


async def compute_utilities(
    options: List[Dict[str, Any]],
    comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
    model_key: Optional[str] = None,
    create_agent_config_path: Optional[str] = None,
    create_agent_config_key: Optional[str] = None,
    agent: Optional[Any] = None,
    compute_utilities_config_path: Optional[str] = None,
    compute_utilities_config_key: Optional[str] = None,
    system_message: Optional[str] = None,
    with_reasoning: Optional[bool] = None,
    save_dir: str = "results",
    save_suffix: Optional[str] = None,
    edge_filter: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None,
    variables: Optional[List] = None,
    analysis_config: Optional[AnalysisConfig] = None,
) -> ExperimentResults:
    """
    Compute utilities for a set of options using a specified utility model.

    Args:
        options: List of option dictionaries with at least an 'id' field
        comparison_prompt_generator: Function that takes two option dictionaries and returns a prompt string
        model_key: Key of the model in models.yaml
        create_agent_config_path: Path to create_agent.yaml
        create_agent_config_key: Key to use in create_agent.yaml
        agent: Pre-initialized agent (if provided, model_key and configs are ignored)
        compute_utilities_config_path: Path to compute_utilities.yaml
        compute_utilities_config_key: Key to use in compute_utilities.yaml
        system_message: Optional system message for the agent. If provided, overrides the value in compute_utilities.yaml
        with_reasoning: Whether to use reasoning-based response parsing. If provided (True/False), overrides the config value
        save_dir: Directory to save results
        save_suffix: Suffix for saved files
        edge_filter: Optional function returning True to keep edge, False to exclude.
                Called with (option_a, option_b) dictionaries.
        variables: Optional list of Variable objects. Used to preserve
                variable metadata (values, etc.) in the results for analysis scripts.
        analysis_config: Optional AnalysisConfig object. Defines which fields to analyze
                and their analysis types. Saved to results for use by analysis scripts.

    Returns:
        ExperimentResults object containing structured results with graph data and utility model
    """
    # Load compute utilities config
    compute_utilities_config = load_config(
        compute_utilities_config_path,
        compute_utilities_config_key,
        "compute_utilities.yaml",
    )

    # Get compute utilities arguments from config
    compute_utilities_arguments = compute_utilities_config.get(
        "compute_utilities_arguments", {}
    )

    # Override config values with provided arguments if they exist; if not provided anywhere, use default values
    compute_utilities_arguments["system_message"] = system_message  # default is None

    if with_reasoning is not None:
        compute_utilities_arguments["with_reasoning"] = with_reasoning
    elif compute_utilities_arguments.get("with_reasoning") is None:
        compute_utilities_arguments["with_reasoning"] = False  # default

    compute_utilities_arguments["comparison_prompt_generator"] = (
        comparison_prompt_generator
    )

    # Update the main config with the potentially modified arguments
    compute_utilities_config["compute_utilities_arguments"] = (
        compute_utilities_arguments
    )

    # Create agent if not provided
    if agent is None:
        if model_key is None:
            raise ValueError("Either agent or model_key must be provided")

        # Load create agent config
        create_agent_config = load_config(
            create_agent_config_path,
            create_agent_config_key or "default",
            "create_agent.yaml",
        )
        agent = create_agent(model_key=model_key, **create_agent_config)

    # Get utility model class
    utility_model_class_name = compute_utilities_config.get(
        "utility_model_class", "ThurstonianActiveLearningUtilityModel"
    )
    utility_model_classes = {
        "ThurstonianActiveLearningUtilityModel": ThurstonianActiveLearningUtilityModel
    }

    if utility_model_class_name not in utility_model_classes:
        raise ValueError(
            f"Unknown utility model class: {utility_model_class_name}. "
            f"Must be one of: {', '.join(utility_model_classes.keys())}"
        )

    utility_model_class = utility_model_classes[utility_model_class_name]

    # Get utility model arguments from config and merge with compute utilities arguments
    utility_model_arguments = compute_utilities_config.get(
        "utility_model_arguments", {}
    )

    # Required arguments from compute_utilities_arguments
    required_args = {
        "unparseable_mode": compute_utilities_arguments.get("unparseable_mode", "skip"),
        "comparison_prompt_generator": compute_utilities_arguments[
            "comparison_prompt_generator"
        ],
        "system_message": compute_utilities_arguments["system_message"],
        "with_reasoning": compute_utilities_arguments["with_reasoning"],
    }

    # Merge required args with model-specific args, giving precedence to model-specific args
    all_model_args = {**required_args, **utility_model_arguments}

    # Initialize the utility model with all arguments
    utility_model = utility_model_class(**all_model_args)

    # Get preference graph arguments from config
    preference_graph_arguments = compute_utilities_config.get(
        "preference_graph_arguments", {}
    )
    holdout_fraction = preference_graph_arguments.get("holdout_fraction", 0.0)
    holdout_seed = preference_graph_arguments.get("holdout_seed", 42)

    graph = PreferenceGraph(
        options=options, holdout_fraction=holdout_fraction, seed=holdout_seed
    )

    # Apply custom edge filter if provided
    if edge_filter:
        options_by_id = {opt["id"]: opt for opt in options}
        filtered_edges = set()

        for id_a, id_b in graph.training_edges_pool:
            opt_a = options_by_id[id_a]
            opt_b = options_by_id[id_b]

            # Keep edge if filter returns True
            if edge_filter(opt_a, opt_b):
                filtered_edges.add((id_a, id_b))

        original_count = len(graph.training_edges_pool)
        graph.training_edges_pool = filtered_edges
        filtered_count = original_count - len(graph.training_edges_pool)
        if filtered_count > 0:
            print(f"Custom edge filter removed {filtered_count} edges")

    # Save example prompt from a random edge (if save_dir is provided)
    if save_dir:
        _save_example_prompt(
            graph=graph,
            comparison_prompt_generator=comparison_prompt_generator,
            system_message=compute_utilities_arguments["system_message"],
            save_path=save_dir,
        )

    # Fit the model (this will populate training edges)
    utilities, metrics = await utility_model.fit(graph, agent)

    # If we have holdout edges, get preferences for them too
    holdout_metrics = await evaluate_holdout_set(
        graph=graph,
        agent=agent,
        utility_model=utility_model,
        utilities=utilities,
        comparison_prompt_generator=compute_utilities_arguments[
            "comparison_prompt_generator"
        ],
        system_message=compute_utilities_arguments["system_message"],
        reasoning_type=compute_utilities_arguments["with_reasoning"],
        K=compute_utilities_arguments.get("K", 10),
    )

    # Prepare results using structured format
    # Create a serializable version of the config (convert callable to string)
    serializable_config = copy.deepcopy(compute_utilities_config)
    if "compute_utilities_arguments" in serializable_config:
        if (
            "comparison_prompt_generator"
            in serializable_config["compute_utilities_arguments"]
        ):
            generator = serializable_config["compute_utilities_arguments"][
                "comparison_prompt_generator"
            ]
            if callable(generator):
                # Replace callable with a string representation
                func_name = getattr(generator, "__name__", "unknown")
                serializable_config["compute_utilities_arguments"][
                    "comparison_prompt_generator"
                ] = f"<callable: {func_name}>"

    # Create PreferenceGraphResults
    graph_config = {
        "compute_utilities_config": serializable_config,
        "preference_graph_arguments": {
            "holdout_fraction": holdout_fraction,
            "holdout_seed": holdout_seed,
        },
    }
    if create_agent_config_path is not None:
        graph_config["create_agent_config"] = create_agent_config

    graph_data = graph.export_data()
    graph_results = PreferenceGraphResults(
        options=[ExperimentOption.from_dict(opt) for opt in options],
        edges=graph_data["edges"],
        training_edges=graph_data["training_edges"],
        holdout_edges=graph_data.get("holdout_edge_indices"),
        variables=variables if variables is not None else [],
        analysis_config=analysis_config or AnalysisConfig(),
        config=graph_config,
    )

    # Create UtilityModelResults
    # Normalize utility keys to strings for consistent access
    utilities_normalized = {str(k): v for k, v in utilities.items()}

    model_config = {
        "utility_model_class": utility_model_class_name,
        "utility_model_arguments": utility_model_arguments,
    }
    utility_results = UtilityModelResults(
        utilities=utilities_normalized,
        training_metrics=metrics,
        holdout_metrics=holdout_metrics,
        model_config=model_config,
    )

    # Combine into ExperimentResults
    results = ExperimentResults(graph=graph_results, utility_model=utility_results)

    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Determine save suffix
        if save_suffix is None:
            save_suffix = model_key

        # Use structured save method (handles all conversions internally)
        results.save(save_dir, save_suffix)

        # Save a short summary txt
        summary_path = os.path.join(save_dir, f"summary_{save_suffix}.txt")
        with open(summary_path, "w") as f:
            f.write(f"Utility Model: {utility_model_class_name}\n\n")
            f.write("Training Metrics:\n")
            for k, v in results.utility_model.training_metrics.items():
                f.write(f"{k}: {v}\n")
            if results.utility_model.holdout_metrics:
                f.write("\nHoldout Metrics:\n")
                for k, v in results.utility_model.holdout_metrics.items():
                    f.write(f"{k}: {v}\n")
            f.write("\nSorted utilities:\n")
            # Write sorted options with utilities
            sorted_results = results.get_sorted_results(reverse=True)
            for opt, util in sorted_results:
                # Truncate long labels for readability
                label = opt.label[:80] + "..." if len(opt.label) > 80 else opt.label
                f.write(
                    f"{label}: mean={util['mean']:.4f}, variance={util['variance']:.4f}\n"
                )

    # Ensure all async tasks are completed before returning
    # This prevents the script from hanging due to pending tasks
    loop = asyncio.get_event_loop()
    current_task = asyncio.current_task(loop)
    pending = [
        task
        for task in asyncio.all_tasks(loop)
        if task is not current_task and not task.done()
    ]
    if pending:
        # Wait for all pending tasks to complete (with a timeout)
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True), timeout=1.0
            )
        except asyncio.TimeoutError:
            # If tasks don't complete quickly, cancel them
            for task in pending:
                if not task.done():
                    task.cancel()
            # Wait a bit for cancellation to propagate
            try:
                await asyncio.gather(*pending, return_exceptions=True)
            except Exception:
                pass  # Ignore exceptions from cancelled tasks

    return results
