"""
Thurstonian Active Learning Utility Model.

Combines Thurstonian preference modeling with active learning for efficient
utility elicitation.
"""

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..utils import generate_responses, parse_responses_forced_choice
from .models import UtilityModel

if TYPE_CHECKING:
    from ..results import PreferenceGraph


# ===================== THURSTONIAN HELPER FUNCTIONS ===================== #


def fit_thurstonian_model(
    graph: "PreferenceGraph", num_epochs: int = 1000, learning_rate: float = 0.01
) -> Tuple[Dict[Any, Dict[str, float]], float, float]:
    """
    Fits the Thurstonian model using the preference graph.

    Args:
        graph: PreferenceGraph object containing the preference data
        num_epochs: Number of epochs for optimization
        learning_rate: Learning rate for optimization

    Returns:
        Tuple containing:
        - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
        - model_log_loss: The log loss (cross-entropy) of the fitted model
        - model_accuracy: Training accuracy of the model
    """
    option_id_to_idx = {option["id"]: idx for idx, option in enumerate(graph.options)}
    n_options = len(graph.options)

    # Initialize parameters
    mu = torch.randn(n_options, requires_grad=True) * 0.01
    s = torch.randn(n_options, requires_grad=True) * 0.01
    mu = torch.nn.Parameter(mu.clone())
    s = torch.nn.Parameter(s.clone())

    optimizer = torch.optim.Adam([mu, s], lr=learning_rate)

    # Prepare training data
    idx_A_list = []
    idx_B_list = []
    probs_A_list = []

    # Convert graph edges to training data
    for edge in graph.edges.values():
        A_id = edge.option_A["id"]
        B_id = edge.option_B["id"]
        idx_A = option_id_to_idx[A_id]
        idx_B = option_id_to_idx[B_id]

        idx_A_list.append(idx_A)
        idx_B_list.append(idx_B)
        probs_A_list.append(edge.probability_A)

    # Convert to tensors
    idx_A_tensor = torch.tensor(idx_A_list, dtype=torch.long)
    idx_B_tensor = torch.tensor(idx_B_list, dtype=torch.long)
    labels_tensor = torch.tensor(probs_A_list, dtype=torch.float32)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Normalize mu to have mean zero and variance one
        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std

        # Adjust sigma^2 accordingly
        scaling_factor = 1 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor**2)

        # Extract normalized parameters for the pairs
        mu_A = mu_normalized[idx_A_tensor]
        mu_B = mu_normalized[idx_B_tensor]
        sigma2_A = sigma2_normalized[idx_A_tensor]
        sigma2_B = sigma2_normalized[idx_B_tensor]

        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B

        z = delta / torch.sqrt(variance)

        # Compute probabilities using the CDF of the normal distribution
        normal = torch.distributions.Normal(0, 1)
        prob_A = normal.cdf(z)

        # Compute loss
        loss = F.binary_cross_entropy(prob_A, labels_tensor, reduction="mean")

        if torch.isnan(loss):
            print("Loss is NaN at epoch:", epoch)
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        loss.backward()
        optimizer.step()

    # Get final parameters
    with torch.no_grad():
        mu_mean = torch.mean(mu)
        mu_std = torch.std(mu) + 1e-5
        mu_normalized = (mu - mu_mean) / mu_std
        scaling_factor = 1 / (mu_std + 1e-5)
        sigma2 = torch.exp(s)
        sigma2_normalized = sigma2 * (scaling_factor**2)

    mu_np = mu_normalized.detach().numpy()
    sigma2_np = sigma2_normalized.detach().numpy()

    # Create utilities dictionary
    option_utilities = {
        option["id"]: {"mean": mu_i, "variance": sigma2_i}
        for option, mu_i, sigma2_i in zip(graph.options, mu_np, sigma2_np)
    }

    # Compute metrics
    y_true = labels_tensor.numpy()
    mu_A = mu_np[idx_A_list]
    mu_B = mu_np[idx_B_list]
    sigma2_A = sigma2_np[idx_A_list]
    sigma2_B = sigma2_np[idx_B_list]
    variance = sigma2_A + sigma2_B + 1e-5
    delta = mu_A - mu_B
    z = delta / np.sqrt(variance)
    prob_A = normal.cdf(torch.tensor(z)).numpy().astype(np.float64)

    # Compute log loss
    eps = 1e-5
    prob_A = np.clip(prob_A, eps, 1 - eps)
    model_log_loss = -np.mean(
        y_true * np.log(prob_A) + (1 - y_true) * np.log(1 - prob_A)
    )

    # Convert both predicted and ground truth probabilities to binary labels using 0.5 threshold
    y_pred_binary = (prob_A >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = np.mean(y_pred_binary == y_true_binary)

    return option_utilities, model_log_loss, model_accuracy


def evaluate_thurstonian_model(
    graph: "PreferenceGraph",
    utilities: Dict[Any, Dict[str, float]],
    edge_indices: List[Tuple[Any, Any]],
) -> Dict[str, float]:
    """
    Evaluate Thurstonian model's goodness-of-fit on the given edges.

    Args:
        graph: PreferenceGraph object containing the preference data
        utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
        edge_indices: List of (option_A_id, option_B_id) tuples to evaluate on

    Returns:
        Dictionary containing evaluation metrics:
        - log_loss: Cross-entropy loss between predicted and actual probabilities
        - accuracy: Accuracy of binary predictions (prob >= 0.5)
    """
    # Extract actual probabilities and prepare data for predictions
    y_true = []
    mu_A_list = []
    mu_B_list = []
    sigma2_A_list = []
    sigma2_B_list = []

    for A_id, B_id in edge_indices:
        edge_index = (A_id, B_id)  # Maintain original orientation
        if edge_index not in graph.edges:
            continue

        edge = graph.edges[edge_index]

        # Get actual probability (ensuring consistent ordering)
        if edge.option_A["id"] == A_id:
            prob_A = edge.probability_A
        else:
            prob_A = 1 - edge.probability_A

        y_true.append(prob_A)

        # Get utilities
        mu_A_list.append(utilities[A_id]["mean"])
        mu_B_list.append(utilities[B_id]["mean"])
        sigma2_A_list.append(utilities[A_id]["variance"])
        sigma2_B_list.append(utilities[B_id]["variance"])

    if not y_true:  # No valid edges found
        return {"log_loss": float("nan"), "accuracy": float("nan")}

    # Convert to numpy arrays
    y_true = np.array(y_true)
    mu_A = np.array(mu_A_list)
    mu_B = np.array(mu_B_list)
    sigma2_A = np.array(sigma2_A_list)
    sigma2_B = np.array(sigma2_B_list)

    # Compute predicted probabilities
    variance = sigma2_A + sigma2_B + 1e-5
    delta = mu_A - mu_B
    z = delta / np.sqrt(variance)
    normal = torch.distributions.Normal(0, 1)
    prob_A = normal.cdf(torch.tensor(z)).numpy()

    # Compute metrics
    eps = 1e-5
    prob_A = np.clip(prob_A, eps, 1 - eps)
    model_log_loss = -np.mean(
        y_true * np.log(prob_A) + (1 - y_true) * np.log(1 - prob_A)
    )

    # Convert both predicted and ground truth probabilities to binary labels using 0.5 threshold
    y_pred_binary = (prob_A >= 0.5).astype(float)
    y_true_binary = (y_true >= 0.5).astype(float)
    model_accuracy = np.mean(y_pred_binary == y_true_binary)

    return {"log_loss": float(model_log_loss), "accuracy": float(model_accuracy)}


# ===================== THURSTONIAN ACTIVE LEARNING HELPER FUNCTIONS ===================== #


def generate_additional_pairs(
    utilities: Dict[Any, Dict[str, float]],
    existing_pairs_set: Set[Tuple[Any, Any]],
    available_edges: Set[Tuple[Any, Any]],
    num_edges_per_iteration: int,
    P: float,
    Q: float,
    seed: Optional[int] = None,
    scale_factor: float = 1.5,
    max_iterations: int = 5,
) -> List[Tuple[Any, Any]]:
    """
    Generates additional pairs by sampling from the intersection of the bottom P% of
    utility differences and the bottom Q% of total degrees.

    Args:
        utilities: Dict mapping option IDs to {'mean': float, 'variance': float}
        existing_pairs_set: Set of existing (option_A_id, option_B_id) tuples
        available_edges: Set of available edges to sample from
        num_edges_per_iteration: Number of edges to sample
        P: Percentage defining bottom P% of utility differences
        Q: Percentage defining bottom Q% of total degrees
        seed: Random seed for reproducibility
        scale_factor: Factor to scale P and Q by if not enough pairs found
        max_iterations: Maximum number of scaling iterations

    Returns:
        List of selected (option_A_id, option_B_id) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    # Compute current degrees for each option
    option_id_to_degree = defaultdict(int)
    for A_id, B_id in existing_pairs_set:
        option_id_to_degree[A_id] += 1
        option_id_to_degree[B_id] += 1

    # Identify which pairs remain
    remaining_pairs = [
        pair for pair in available_edges if pair not in existing_pairs_set
    ]
    if not remaining_pairs:
        print("No remaining pairs to sample.")
        return []

    def get_pairs_in_bottom_PQ_percent(p: float, q: float) -> List[Tuple[Any, Any]]:
        """Get pairs in intersection of bottom p% utility differences and q% degrees."""
        utility_differences = []
        total_degrees = []
        for pair in remaining_pairs:
            A_id, B_id = pair  # Keep original orientation
            diff = abs(utilities[A_id]["mean"] - utilities[B_id]["mean"])
            utility_differences.append(diff)
            deg_sum = option_id_to_degree[A_id] + option_id_to_degree[B_id]
            total_degrees.append(deg_sum)

        utility_differences = np.array(utility_differences)
        total_degrees = np.array(total_degrees)

        utility_cutoff = np.percentile(utility_differences, p)
        degree_cutoff = np.percentile(total_degrees, q)

        mask = (utility_differences <= utility_cutoff) & (
            total_degrees <= degree_cutoff
        )
        return [
            pair for pair, m in zip(remaining_pairs, mask) if m
        ]  # Keep original orientation

    # Try progressively increasing P and Q until we get enough pairs
    current_pairs = []
    current_P = P
    current_Q = Q

    for i in range(max_iterations):
        candidate_subset = get_pairs_in_bottom_PQ_percent(current_P, current_Q)

        if len(candidate_subset) >= num_edges_per_iteration:
            current_pairs = random.sample(candidate_subset, num_edges_per_iteration)
            break
        else:
            current_pairs = candidate_subset[:]
            if i < max_iterations - 1:
                current_P = min(current_P * scale_factor, 100.0)
                current_Q = min(current_Q * scale_factor, 100.0)
            else:
                shortfall = num_edges_per_iteration - len(current_pairs)
                if shortfall > 0:
                    remaining_after_cut = list(
                        set(remaining_pairs) - set(current_pairs)
                    )
                    if len(remaining_after_cut) > shortfall:
                        fallback_sample = random.sample(remaining_after_cut, shortfall)
                        current_pairs.extend(fallback_sample)
                    else:
                        current_pairs.extend(remaining_after_cut)
                break

    print(
        f"Number of additional pairs added: {len(current_pairs)} (after possibly scaling P and Q)"
    )
    return current_pairs


def generate_pseudolabels(
    utilities: Dict[Any, Dict[str, float]],
    existing_pairs_set: Set[Tuple[Any, Any]],
    available_edges: Set[Tuple[Any, Any]],
    confidence_threshold: float,
) -> Dict[Tuple[Any, Any], Dict[Any, int]]:
    """
    Generates pseudolabels for unsampled pairs using the Thurstonian model.

    Args:
        utilities: Dict mapping option IDs to {'mean': float, 'variance': float}
        existing_pairs_set: Set of existing (option_A_id, option_B_id) tuples
        available_edges: Set of available edges to sample from
        confidence_threshold: Confidence threshold for generating pseudolabels

    Returns:
        Dictionary mapping (option_A_id, option_B_id) to counts dictionary
    """
    unsampled_pairs = [
        pair for pair in available_edges if pair not in existing_pairs_set
    ]
    normal = torch.distributions.Normal(0, 1)
    pseudolabels_counts = {}
    num_pseudolabels_added = 0

    for A_id, B_id in unsampled_pairs:  # Keep original orientation
        mu_A = utilities[A_id]["mean"]
        mu_B = utilities[B_id]["mean"]
        sigma2_A = utilities[A_id]["variance"]
        sigma2_B = utilities[B_id]["variance"]

        variance = sigma2_A + sigma2_B + 1e-5
        delta = mu_A - mu_B
        z = delta / np.sqrt(variance)
        prob_A = normal.cdf(torch.tensor(z)).item()

        if prob_A >= confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {
                A_id: 1,
                B_id: 0,
            }  # Keep original orientation
            num_pseudolabels_added += 1
        elif prob_A <= 1 - confidence_threshold:
            pseudolabels_counts[(A_id, B_id)] = {
                A_id: 0,
                B_id: 1,
            }  # Keep original orientation
            num_pseudolabels_added += 1

    print(f"Number of pseudolabels added: {num_pseudolabels_added}")
    return pseudolabels_counts


# ===================== UTILITY MODEL CLASS ===================== #


class ThurstonianActiveLearningUtilityModel(UtilityModel):
    """
    Active learning variant of the Thurstonian utility model.
    Uses a combination of utility differences and degree-based sampling to select edges.
    """

    def __init__(
        self,
        unparseable_mode: str,
        comparison_prompt_generator: Callable[[Dict[str, Any], Dict[str, Any]], str],
        system_message: str,
        with_reasoning: str,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        edge_multiplier: float = 2.0,
        degree: int = 2,
        num_edges_per_iteration: int = 200,
        P: float = 10.0,
        Q: float = 20.0,
        use_pseudolabels: bool = False,
        pseudolabel_confidence_threshold: float = 0.95,
        seed: Optional[int] = None,
        K: int = 10,
        include_flipped: bool = True,
        max_iterations: Optional[int] = None,
    ):
        """
        Initialize the Thurstonian Active Learning utility model.

        Args:
            unparseable_mode: How to handle unparseable responses
            comparison_prompt_generator: Callable function that takes (option_A_dict, option_B_dict) and returns a prompt string
            system_message: System message for agents that accept a system message
            with_reasoning: Whether to use response parsing (options: 'REASONING_BEFORE', 'REASONING_AFTER', 'NO_REASONING')
            num_epochs: Number of epochs for optimization
            learning_rate: Learning rate for optimization
            edge_multiplier: Multiplier for number of edges
            degree: Degree of initial regular graph
            num_edges_per_iteration: Number of edges to sample in each iteration
            P: Percentage defining bottom P% of utility differences to sample from
            Q: Percentage defining bottom Q% of total degrees to sample from
            use_pseudolabels: Whether to use pseudolabeling in final stage
            pseudolabel_confidence_threshold: Confidence threshold for pseudolabeling
            seed: Random seed for reproducibility
            K: Number of responses to generate per prompt
            include_flipped: Whether to include flipped prompts (Note: This should always be True; we only set it to False for demonstration purposes)
            max_iterations: Maximum number of active learning iterations. If None, calculated from edge_multiplier.
        """
        # Call parent class's __init__ with required arguments
        super().__init__(
            unparseable_mode=unparseable_mode,
            comparison_prompt_generator=comparison_prompt_generator,
            system_message=system_message,
            with_reasoning=with_reasoning,
        )

        # Store model-specific arguments as attributes
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.edge_multiplier = edge_multiplier
        self.degree = degree
        self.num_edges_per_iteration = num_edges_per_iteration
        self.P = P
        self.Q = Q
        self.use_pseudolabels = use_pseudolabels
        self.pseudolabel_confidence_threshold = pseudolabel_confidence_threshold
        self.seed = seed
        self.K = K
        self.include_flipped = include_flipped
        self.max_iterations = max_iterations

    async def fit(
        self, graph: "PreferenceGraph", agent: Any
    ) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
        """
        Fit the model using active learning to select edges.

        Args:
            graph: PreferenceGraph object containing the preference data
            agent: The agent used for generating comparisons

        Returns:
            Tuple containing:
            - option_utilities: Dict mapping each option ID to {'mean': float, 'variance': float}
            - metrics: Dict containing model metrics like log_loss and accuracy
        """
        if self.comparison_prompt_generator is None:
            raise ValueError("comparison_prompt_generator must be provided")

        # Calculate target number of edges and number of iterations
        N = len(graph.options)
        target_total_edges = int(self.edge_multiplier * N * np.log2(N))
        initial_edges = (N * self.degree) // 2
        remainder = target_total_edges - initial_edges
        if remainder <= 0:
            num_iterations = 0
        else:
            num_iterations = int(np.ceil(remainder / self.num_edges_per_iteration))

        # Apply max_iterations limit if specified
        if self.max_iterations is not None:
            num_iterations = min(num_iterations, self.max_iterations)

        print(f"Target total edges: {target_total_edges}")
        print(f"Initial edges: {initial_edges}")
        print(
            f"Number of iterations: {num_iterations}"
            + (
                f" (limited by max_iterations={self.max_iterations})"
                if self.max_iterations is not None
                and num_iterations == self.max_iterations
                else ""
            )
        )

        # Generate initial pairs using regular graph
        initial_pairs = graph.sample_regular_graph(degree=self.degree, seed=self.seed)
        if len(initial_pairs) < initial_edges:
            # If we didn't get enough edges from regular graph, sample additional random edges
            needed = initial_edges - len(initial_pairs)
            remaining_edges = list(graph.training_edges_pool - set(initial_pairs))
            if remaining_edges:
                additional = random.sample(
                    remaining_edges, min(needed, len(remaining_edges))
                )
                initial_pairs.extend(additional)

        # Get responses for initial pairs
        preference_data, prompt_list, prompt_idx_to_key = graph.generate_prompts(
            initial_pairs,
            self.comparison_prompt_generator,
            include_flipped=self.include_flipped,
        )

        responses = await generate_responses(
            agent=agent,
            prompts=prompt_list,
            system_message=self.system_message,
            K=self.K,
        )

        parsed_responses, reasoning_results, reasoning_summaries = (
            parse_responses_forced_choice(responses, with_reasoning=self.with_reasoning)
        )
        processed_preference_data = self.process_responses(
            graph=graph,
            responses=responses,
            parsed_responses=parsed_responses,
            prompt_idx_to_key=prompt_idx_to_key,
            reasoning_results=reasoning_results,
            reasoning_summaries=reasoning_summaries,
        )

        graph.add_edges(processed_preference_data)

        # Initial fit
        utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
            graph=graph, num_epochs=self.num_epochs, learning_rate=self.learning_rate
        )

        print(
            f"Initial model - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%"
        )

        # Active learning iterations
        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}/{num_iterations}")
            print(
                f"Sampling up to {self.num_edges_per_iteration} new pairs from the intersection of the bottom {self.P}% utility differences and bottom {self.Q}% total degrees."
            )

            # Get current utilities and existing pairs
            existing_pairs_set = set(
                (edge.option_A["id"], edge.option_B["id"])
                for edge in graph.edges.values()
            )

            # Generate additional pairs
            additional_pairs = generate_additional_pairs(
                utilities,
                existing_pairs_set,
                graph.training_edges_pool,
                self.num_edges_per_iteration,
                self.P,
                self.Q,
                seed=self.seed,
            )

            if not additional_pairs:  # No more pairs to sample
                break

            # Get responses for additional pairs
            preference_data, prompt_list, prompt_idx_to_key = graph.generate_prompts(
                additional_pairs,
                self.comparison_prompt_generator,
                include_flipped=self.include_flipped,
            )

            responses = await generate_responses(
                agent=agent,
                prompts=prompt_list,
                system_message=self.system_message,
                K=self.K,
            )

            parsed_responses, reasoning_results, reasoning_summaries = (
                parse_responses_forced_choice(
                    responses, with_reasoning=self.with_reasoning
                )
            )
            processed_preference_data = self.process_responses(
                graph=graph,
                responses=responses,
                parsed_responses=parsed_responses,
                prompt_idx_to_key=prompt_idx_to_key,
                reasoning_results=reasoning_results,
                reasoning_summaries=reasoning_summaries,
            )

            graph.add_edges(processed_preference_data)

            # Refit model
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
            )

            print(
                f"Updated model - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%"
            )

        # Optional: Generate pseudolabels
        if self.use_pseudolabels:
            print("\nGenerating pseudolabels using the current Thurstonian model.")
            existing_pairs_set = set(
                (edge.option_A["id"], edge.option_B["id"])
                for edge in graph.edges.values()
            )

            pseudolabels = generate_pseudolabels(
                utilities,
                existing_pairs_set,
                graph.training_edges_pool,
                self.pseudolabel_confidence_threshold,
            )

            # Convert pseudolabels into preference data format and add to graph
            for (A_id, B_id), counts in pseudolabels.items():
                # Create synthetic preference data
                prob_A = counts[A_id] / (counts[A_id] + counts[B_id])
                processed_data = [
                    {
                        "option_A": graph.options_by_id[A_id],
                        "option_B": graph.options_by_id[B_id],
                        "probability_A": prob_A,
                        "aux_data": {
                            "is_pseudolabel": True,
                            "count_A": counts[A_id],
                            "count_B": counts[B_id],
                            "total_responses": counts[A_id] + counts[B_id],
                        },
                    }
                ]
                graph.add_edges(processed_data)

            # Final fit with pseudolabels
            utilities, model_log_loss, model_accuracy = fit_thurstonian_model(
                graph=graph,
                num_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
            )

            print(
                f"Final model with pseudolabels - Log Loss: {model_log_loss:.4f}, Accuracy: {model_accuracy * 100:.2f}%"
            )

        metrics = {"log_loss": float(model_log_loss), "accuracy": float(model_accuracy)}

        return utilities, metrics

    @classmethod
    def evaluate(
        cls,
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
            Dictionary containing evaluation metrics (log_loss and accuracy)
        """
        return evaluate_thurstonian_model(graph, utilities, edge_indices)
