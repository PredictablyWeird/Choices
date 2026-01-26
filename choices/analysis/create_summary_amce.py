#!/usr/bin/env python3
"""
Create a summary table of nudge experiment results with AMCE-based metrics.

This script creates a table showing AMCE (Average Marginal Component Effect) metrics:
- f_0(B): Baseline frequency of choosing B
- f_A(B): Frequency of choosing B when nudged towards A
- f_B(B): Frequency of choosing B when nudged towards B
- AMCE_0: AMCE of factor B (vs A) in baseline condition
- AMCE_A: AMCE of factor B (vs A) when nudged towards A
- AMCE_B: AMCE of factor B (vs A) when nudged towards B
- effect_A: AMCE_0 - AMCE_A (effect of nudging towards A)
- effect_B: AMCE_B - AMCE_0 (effect of nudging towards B)

The AMCE is computed using logistic regression with log-linear effect of N.

Usage:
    uv run python -m choices.analysis.create_summary_amce
    uv run python -m choices.analysis.create_summary_amce --results-dirs results
    uv run python -m choices.analysis.create_summary_amce --output summary.csv
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit

from choices.analysis.nudge_effect_size import (
    get_factor_levels_from_graph,
    get_factor_name_from_graph,
    load_preference_graph,
)
from choices.analysis.utils import (
    get_base_model_name,
    get_model_display_name,
    get_reasoning_condition,
    get_reasoning_mode_from_results,
)

DEFAULT_ALPHA = 0.05


@dataclass
class AMCEResult:
    """AMCE-based results for a single nudge experiment."""

    model: str
    reasoning_condition: str
    factor: str
    nudge_type: str
    level_A: str
    level_B: str
    f_0_B: float
    f_A_B: float
    f_B_B: float
    amce_0: float
    amce_A: float
    amce_B: float
    amce_0_se: float
    amce_A_se: float
    amce_B_se: float
    effect_A: float
    effect_B: float
    sig_amce_0: bool
    sig_amce_A: bool
    sig_amce_B: bool
    sig_effect_A: bool
    sig_effect_B: bool
    log_n_coef_0: float
    log_n_coef_A: float
    log_n_coef_B: float
    n_comparisons: int
    n_observations: int
    invalid_pct: float


def extract_observations_from_graph(
    graph_data: Dict,
    factor_name: str,
    factor_levels: List[str],
) -> List[Dict]:
    """
    Extract individual observations from a preference graph for regression.

    Returns:
        List of observation dictionaries with y, factor_first, factor_second,
        n_first, n_second.
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})
    options_by_id = {opt["id"]: opt for opt in options}

    observations = []

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            factor_a = opt_a.get(factor_name)
            factor_b = opt_b.get(factor_name)

            if factor_a not in factor_levels or factor_b not in factor_levels:
                continue

            if factor_a == factor_b:
                continue

            n_a = opt_a.get("N", 1)
            n_b = opt_b.get("N", 1)

            aux_data = edge_data.get("aux_data", {})
            original_parsed = aux_data.get("original_parsed", [])
            flipped_parsed = aux_data.get("flipped_parsed", [])

            for resp in original_parsed:
                if resp in ("A", "B"):
                    observations.append(
                        {
                            "y": 1 if resp == "A" else 0,
                            "factor_first": factor_a,
                            "factor_second": factor_b,
                            "n_first": n_a,
                            "n_second": n_b,
                        }
                    )

            for resp in flipped_parsed:
                if resp in ("A", "B"):
                    observations.append(
                        {
                            "y": 1 if resp == "B" else 0,
                            "factor_first": factor_a,
                            "factor_second": factor_b,
                            "n_first": n_a,
                            "n_second": n_b,
                        }
                    )

        except Exception:
            continue

    return observations


def compute_pooled_amce(
    obs_base: List[Dict],
    obs_nudge_A: List[Dict],
    obs_nudge_B: List[Dict],
    factor_levels: List[str],
    l2_penalty: float = 0.1,
) -> Dict[str, float]:
    """
    Compute AMCE using pooled logistic regression across all conditions.

    Model:
        logit(P(choose first option)) = b0
            + b_factor * factor_indicator
            + b_nudge_A * nudge_A_indicator
            + b_nudge_B * nudge_B_indicator
            + b_effect_A * (factor_indicator * nudge_A_indicator)
            + b_effect_B * (factor_indicator * nudge_B_indicator)
            + b_logn * log(N_first / N_second)

    Where factor_indicator = +1 if first option is level_B, -1 if level_A.

    Returns dict with:
        - amce_0: factor effect in baseline (b_factor)
        - amce_A: factor effect when nudged towards A (b_factor + b_effect_A)
        - amce_B: factor effect when nudged towards B (b_factor + b_effect_B)
        - effect_A: change in AMCE due to nudge_A (b_effect_A)
        - effect_B: change in AMCE due to nudge_B (b_effect_B)
        - log_n_coef: coefficient for log(N)
        - standard errors for each
    """
    level_A, level_B = factor_levels[0], factor_levels[1]

    # Combine all observations with condition labels
    all_obs = []
    for obs in obs_base:
        all_obs.append({**obs, "condition": "base"})
    for obs in obs_nudge_A:
        all_obs.append({**obs, "condition": "nudge_A"})
    for obs in obs_nudge_B:
        all_obs.append({**obs, "condition": "nudge_B"})

    if len(all_obs) < 20:
        return _empty_amce_result()

    # Build design matrix
    y = []
    X_factor = []  # factor indicator: +1 if first is B, -1 if first is A
    X_nudge_A = []  # 1 if nudge_A condition, 0 otherwise
    X_nudge_B = []  # 1 if nudge_B condition, 0 otherwise
    X_interact_A = []  # factor * nudge_A
    X_interact_B = []  # factor * nudge_B
    X_logn = []  # log(N_first / N_second)

    for obs in all_obs:
        y.append(obs["y"])

        # Factor coding
        if obs["factor_first"] == level_B and obs["factor_second"] == level_A:
            factor_val = 1
        elif obs["factor_first"] == level_A and obs["factor_second"] == level_B:
            factor_val = -1
        else:
            factor_val = 0

        X_factor.append(factor_val)

        # Condition indicators
        is_nudge_A = 1 if obs["condition"] == "nudge_A" else 0
        is_nudge_B = 1 if obs["condition"] == "nudge_B" else 0
        X_nudge_A.append(is_nudge_A)
        X_nudge_B.append(is_nudge_B)

        # Interaction terms
        X_interact_A.append(factor_val * is_nudge_A)
        X_interact_B.append(factor_val * is_nudge_B)

        # Log N ratio
        n_first = max(obs["n_first"], 1)
        n_second = max(obs["n_second"], 1)
        X_logn.append(np.log(n_first / n_second))

    y = np.array(y)
    X = np.column_stack(
        [
            np.ones(len(y)),  # intercept
            X_factor,  # b_factor (AMCE_0)
            X_nudge_A,  # b_nudge_A (main effect of nudge A condition)
            X_nudge_B,  # b_nudge_B (main effect of nudge B condition)
            X_interact_A,  # b_effect_A (change in AMCE due to nudge A)
            X_interact_B,  # b_effect_B (change in AMCE due to nudge B)
            X_logn,  # b_logn
        ]
    )
    # Columns: 0=intercept, 1=factor, 2=nudge_A, 3=nudge_B, 4=interact_A, 5=interact_B, 6=logn

    try:

        def neg_log_likelihood(beta):
            linear_pred = np.clip(X @ beta, -20, 20)
            prob = expit(linear_pred)
            eps = 1e-10
            ll = np.sum(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))
            # L2 regularization (don't penalize intercept)
            penalty = l2_penalty * np.sum(beta[1:] ** 2)
            return -ll + penalty

        def gradient(beta):
            linear_pred = np.clip(X @ beta, -20, 20)
            prob = expit(linear_pred)
            grad = -X.T @ (y - prob)
            # L2 gradient (don't penalize intercept)
            grad[1:] += 2 * l2_penalty * beta[1:]
            return grad

        beta0 = np.zeros(7)
        result = minimize(
            neg_log_likelihood,
            beta0,
            method="BFGS",
            jac=gradient,
            options={"maxiter": 200},
        )

        if not result.success:
            return _empty_amce_result()

        beta = result.x

        # Compute standard errors using Fisher information (ignoring regularization for SE)
        linear_pred = np.clip(X @ beta, -20, 20)
        prob = expit(linear_pred)
        W = np.diag(prob * (1 - prob) + 1e-6)  # add small constant for stability

        try:
            fisher_info = X.T @ W @ X
            # Add regularization to Fisher info for stable inversion
            fisher_info[1:, 1:] += 2 * l2_penalty * np.eye(6)
            cov_matrix = np.linalg.inv(fisher_info)
            se = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-10))
        except np.linalg.LinAlgError:
            se = np.ones(7)

        # Extract coefficients
        b_factor = beta[1]  # AMCE_0
        # b_nudge_A = beta[2]  # main effect of nudge A (not used in AMCE)
        # b_nudge_B = beta[3]  # main effect of nudge B (not used in AMCE)
        b_effect_A = beta[4]  # interaction: change in AMCE due to nudge A
        b_effect_B = beta[5]  # interaction: change in AMCE due to nudge B
        b_logn = beta[6]

        se_factor = se[1]
        ##se_nudge_A = se[2]
        ##se_nudge_B = se[3]
        se_effect_A = se[4]
        se_effect_B = se[5]
        se_logn = se[6]

        # Compute AMCE for each condition
        amce_0 = b_factor
        amce_A = b_factor + b_effect_A
        amce_B = b_factor + b_effect_B

        # SE for AMCE_A and AMCE_B using delta method (assuming independence for simplicity)
        # More accurate would use full covariance, but this is a reasonable approximation
        se_amce_0 = se_factor
        # For sum of coefficients, SE = sqrt(Var(b_factor) + Var(b_effect) + 2*Cov)
        # Approximate with sqrt(se1^2 + se2^2)
        se_amce_A = np.sqrt(se_factor**2 + se_effect_A**2)
        se_amce_B = np.sqrt(se_factor**2 + se_effect_B**2)

        return {
            "amce_0": amce_0,
            "amce_A": amce_A,
            "amce_B": amce_B,
            "amce_0_se": se_amce_0,
            "amce_A_se": se_amce_A,
            "amce_B_se": se_amce_B,
            "effect_A": b_effect_A,
            "effect_B": b_effect_B,
            "effect_A_se": se_effect_A,
            "effect_B_se": se_effect_B,
            "log_n_coef": b_logn,
            "log_n_se": se_logn,
        }

    except Exception:
        return _empty_amce_result()


def _empty_amce_result() -> Dict[str, float]:
    """Return empty AMCE result with default values."""
    return {
        "amce_0": 0.0,
        "amce_A": 0.0,
        "amce_B": 0.0,
        "amce_0_se": 1.0,
        "amce_A_se": 1.0,
        "amce_B_se": 1.0,
        "effect_A": 0.0,
        "effect_B": 0.0,
        "effect_A_se": 1.0,
        "effect_B_se": 1.0,
        "log_n_coef": 0.0,
        "log_n_se": 1.0,
    }


def compute_frequency(observations: List[Dict], level_B: str) -> float:
    """Compute frequency of choosing level_B from observations."""
    if not observations:
        return 0.5

    wins_B = 0
    total = 0

    for obs in observations:
        if obs["factor_first"] == level_B:
            wins_B += obs["y"]
            total += 1
        elif obs["factor_second"] == level_B:
            wins_B += 1 - obs["y"]
            total += 1

    return wins_B / total if total > 0 else 0.5


def count_responses(graph_data: Dict) -> Tuple[int, int]:
    """Count valid and total responses in a preference graph."""
    edges = graph_data.get("edges", {})
    valid_count = 0
    total_count = 0

    for edge_data in edges.values():
        aux_data = edge_data.get("aux_data", {})
        original_parsed = aux_data.get("original_parsed", [])
        flipped_parsed = aux_data.get("flipped_parsed", [])

        total_count += len(original_parsed) + len(flipped_parsed)

        for resp in original_parsed:
            if resp in ("A", "B"):
                valid_count += 1
        for resp in flipped_parsed:
            if resp in ("A", "B"):
                valid_count += 1

    return valid_count, total_count


def get_nudge_target_group(result_dir: Path) -> Optional[str]:
    """Get the target group for a nudge condition from the graph data."""
    graph_data = load_preference_graph(result_dir)
    if not graph_data:
        return None

    nudge_config = graph_data.get("nudge_config")
    if nudge_config:
        return nudge_config.get("target_group")
    return None


def find_condition_directories(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> List[Dict[str, Path]]:
    """Find result directories for each condition (base, and each nudge target)."""
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return []

    dirs_by_condition_and_reasoning: Dict[Tuple[str, str], List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        reasoning_mode = get_reasoning_mode_from_results(result_dir)
        if reasoning_mode is None:
            reasoning_mode = "unknown"

        key = (condition, reasoning_mode)
        if key not in dirs_by_condition_and_reasoning:
            dirs_by_condition_and_reasoning[key] = []
        dirs_by_condition_and_reasoning[key].append(result_dir)

    experiments_by_reasoning: Dict[str, Dict[str, Path]] = {}

    for (condition, reasoning_mode), dirs in dirs_by_condition_and_reasoning.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)

        if reasoning_mode not in experiments_by_reasoning:
            experiments_by_reasoning[reasoning_mode] = {}
        experiments_by_reasoning[reasoning_mode][condition] = most_recent

    return list(experiments_by_reasoning.values())


def _compute_single_amce_result(
    factor_name: str,
    model: str,
    nudge_type: str,
    condition_dirs: Dict[str, Path],
) -> Optional[AMCEResult]:
    """Compute AMCE metrics for a single experiment using pooled regression."""
    if "base" not in condition_dirs:
        return None

    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return None

    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return None

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])

    if not nudge_A_graph or not nudge_B_graph:
        return None

    obs_base = extract_observations_from_graph(
        base_graph, factor_var_name, factor_levels
    )
    obs_nudge_A = extract_observations_from_graph(
        nudge_A_graph, factor_var_name, factor_levels
    )
    obs_nudge_B = extract_observations_from_graph(
        nudge_B_graph, factor_var_name, factor_levels
    )

    if not obs_base or not obs_nudge_A or not obs_nudge_B:
        return None

    # Compute raw frequencies
    f_0_B = compute_frequency(obs_base, level_B)
    f_A_B = compute_frequency(obs_nudge_A, level_B)
    f_B_B = compute_frequency(obs_nudge_B, level_B)

    # Compute AMCE using pooled regression across all conditions
    amce_result = compute_pooled_amce(obs_base, obs_nudge_A, obs_nudge_B, factor_levels)

    amce_0 = amce_result["amce_0"]
    amce_A = amce_result["amce_A"]
    amce_B = amce_result["amce_B"]
    amce_0_se = amce_result["amce_0_se"]
    amce_A_se = amce_result["amce_A_se"]
    amce_B_se = amce_result["amce_B_se"]
    effect_A = amce_result["effect_A"]
    effect_B = amce_result["effect_B"]
    effect_A_se = amce_result["effect_A_se"]
    effect_B_se = amce_result["effect_B_se"]
    log_n_coef = amce_result["log_n_coef"]

    # Compute significance
    z_crit = stats.norm.ppf(1 - DEFAULT_ALPHA / 2)

    sig_amce_0 = abs(amce_0 / amce_0_se) > z_crit if amce_0_se > 0 else False
    sig_amce_A = abs(amce_A / amce_A_se) > z_crit if amce_A_se > 0 else False
    sig_amce_B = abs(amce_B / amce_B_se) > z_crit if amce_B_se > 0 else False
    sig_effect_A = abs(effect_A / effect_A_se) > z_crit if effect_A_se > 0 else False
    sig_effect_B = abs(effect_B / effect_B_se) > z_crit if effect_B_se > 0 else False

    n_comparisons = len(base_graph.get("edges", {}))
    n_observations = len(obs_base) + len(obs_nudge_A) + len(obs_nudge_B)

    valid_base, total_base = count_responses(base_graph)
    valid_A, total_A = count_responses(nudge_A_graph)
    valid_B, total_B = count_responses(nudge_B_graph)

    total_valid = valid_base + valid_A + valid_B
    total_responses = total_base + total_A + total_B

    invalid_pct = (
        ((total_responses - total_valid) / total_responses * 100)
        if total_responses > 0
        else 0.0
    )

    reasoning_condition = get_reasoning_condition(model, condition_dirs["base"])

    return AMCEResult(
        model=model,
        reasoning_condition=reasoning_condition,
        factor=factor_name,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        f_0_B=f_0_B,
        f_A_B=f_A_B,
        f_B_B=f_B_B,
        amce_0=amce_0,
        amce_A=amce_A,
        amce_B=amce_B,
        amce_0_se=amce_0_se,
        amce_A_se=amce_A_se,
        amce_B_se=amce_B_se,
        effect_A=effect_A,
        effect_B=effect_B,
        sig_amce_0=sig_amce_0,
        sig_amce_A=sig_amce_A,
        sig_amce_B=sig_amce_B,
        sig_effect_A=sig_effect_A,
        sig_effect_B=sig_effect_B,
        log_n_coef_0=log_n_coef,
        log_n_coef_A=log_n_coef,  # Same coefficient (pooled model)
        log_n_coef_B=log_n_coef,  # Same coefficient (pooled model)
        n_comparisons=n_comparisons,
        n_observations=n_observations,
        invalid_pct=invalid_pct,
    )


def compute_amce_results(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> List[AMCEResult]:
    """Compute AMCE metrics for all experiments matching the given parameters."""
    experiment_sets = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    results = []
    for condition_dirs in experiment_sets:
        result = _compute_single_amce_result(
            factor_name, model, nudge_type, condition_dirs
        )
        if result is not None:
            results.append(result)

    return results


def discover_experiments(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    """Discover all available experiments in the results directories."""
    experiments = []

    for results_base_dir in results_base_dirs:
        results_path = Path(results_base_dir)
        if not results_path.exists():
            continue

        for exp_dir in results_path.iterdir():
            if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
                continue

            factor_name = exp_dir.name[7:]

            if factor_filter and factor_name not in factor_filter:
                continue

            for model_dir in exp_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name

                if model_filter and model not in model_filter:
                    continue

                for nudge_dir in model_dir.iterdir():
                    if not nudge_dir.is_dir():
                        continue

                    nudge_type = nudge_dir.name

                    if nudge_type_filter and nudge_type not in nudge_type_filter:
                        continue

                    experiments.append(
                        (results_base_dir, factor_name, model, nudge_type)
                    )

    return experiments


def compute_all_results(
    results_base_dirs: List[str],
    model_filter: Optional[List[str]] = None,
    factor_filter: Optional[List[str]] = None,
    nudge_type_filter: Optional[List[str]] = None,
) -> List[AMCEResult]:
    """Compute AMCE results for all available experiments."""
    experiments = discover_experiments(
        results_base_dirs, model_filter, factor_filter, nudge_type_filter
    )

    results = []
    for results_base_dir, factor_name, model, nudge_type in experiments:
        experiment_results = compute_amce_results(
            factor_name, model, nudge_type, results_base_dir
        )
        results.extend(experiment_results)

    return results


def sort_results(
    results: List[AMCEResult],
    sort_column: Optional[str] = None,
    reverse: bool = False,
) -> List[AMCEResult]:
    """Sort results by the specified column."""
    if not sort_column:
        return sorted(
            results,
            key=lambda r: (
                get_base_model_name(r.model),
                r.factor,
                r.nudge_type,
                r.reasoning_condition,
            ),
        )

    column_map = {
        "model": lambda r: r.model,
        "factor": lambda r: r.factor,
        "nudge_type": lambda r: r.nudge_type,
        "reasoning": lambda r: r.reasoning_condition,
        "f_0_b": lambda r: r.f_0_B,
        "f_a_b": lambda r: r.f_A_B,
        "f_b_b": lambda r: r.f_B_B,
        "amce_0": lambda r: r.amce_0,
        "amce_a": lambda r: r.amce_A,
        "amce_b": lambda r: r.amce_B,
        "effect_a": lambda r: r.effect_A,
        "effect_b": lambda r: r.effect_B,
        "abs_effect_a": lambda r: abs(r.effect_A),
        "abs_effect_b": lambda r: abs(r.effect_B),
        "log_n_coef": lambda r: r.log_n_coef_0,
    }

    key_func = column_map.get(sort_column.lower())
    if key_func:
        return sorted(results, key=key_func, reverse=reverse)

    return results


TABLE_COLUMNS = [
    ("Model", "model"),
    ("Reasoning", "reasoning"),
    ("Factor", "factor"),
    ("Nudge Type", "nudge_type"),
    ("Invalid%", "invalid_pct"),
    ("f_0(B)", "f_0_b"),
    ("f_A(B)", "f_a_b"),
    ("f_B(B)", "f_b_b"),
    ("AMCE_0", "amce_0"),
    ("AMCE_A", "amce_a"),
    ("AMCE_B", "amce_b"),
    ("effect_A", "effect_a"),
    ("effect_B", "effect_b"),
    ("log(N)", "log_n"),
]


def format_table(
    results: List[AMCEResult],
    show_display_names: bool = True,
    decimals: int = 3,
    sort_column: Optional[str] = None,
    reverse: bool = False,
    hide_columns: Optional[List[str]] = None,
) -> str:
    """Format results as a text table."""
    if not results:
        return "No results found."

    results = sort_results(results, sort_column, reverse)

    hidden_set = set()
    if hide_columns:
        for col in hide_columns:
            hidden_set.add(col.lower().replace("-", "_").replace(" ", "_"))

    visible_columns = [
        (header, key) for header, key in TABLE_COLUMNS if key not in hidden_set
    ]
    headers = [header for header, _ in visible_columns]
    visible_keys = [key for _, key in visible_columns]

    rows = []
    for r in results:
        model_name = get_model_display_name(r.model) if show_display_names else r.model
        factor_with_levels = f"{r.level_A}/{r.level_B}"

        amce_0_str = f"{r.amce_0:+.{decimals}f}{'*' if r.sig_amce_0 else ''}"
        amce_A_str = f"{r.amce_A:+.{decimals}f}{'*' if r.sig_amce_A else ''}"
        amce_B_str = f"{r.amce_B:+.{decimals}f}{'*' if r.sig_amce_B else ''}"

        effect_A_str = f"{r.effect_A:+.{decimals}f}{'*' if r.sig_effect_A else ''}"
        effect_B_str = f"{r.effect_B:+.{decimals}f}{'*' if r.sig_effect_B else ''}"

        all_values = {
            "model": model_name,
            "reasoning": r.reasoning_condition,
            "factor": factor_with_levels,
            "nudge_type": r.nudge_type,
            "invalid_pct": f"{r.invalid_pct:.1f}%",
            "f_0_b": f"{r.f_0_B:.{decimals}f}",
            "f_a_b": f"{r.f_A_B:.{decimals}f}",
            "f_b_b": f"{r.f_B_B:.{decimals}f}",
            "amce_0": amce_0_str,
            "amce_a": amce_A_str,
            "amce_b": amce_B_str,
            "effect_a": effect_A_str,
            "effect_b": effect_B_str,
            "log_n": f"{r.log_n_coef_0:+.{decimals}f}",
        }

        rows.append([all_values[key] for key in visible_keys])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in col_widths)

    row_lines = []
    for row in rows:
        row_lines.append(
            " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        )

    return "\n".join([header_line, separator] + row_lines)


def write_csv(
    results: List[AMCEResult],
    output_path: str,
    show_display_names: bool = True,
    sort_column: Optional[str] = None,
    reverse: bool = False,
) -> None:
    """Write results to a CSV file."""
    if not results:
        print("No results to write.")
        return

    results = sort_results(results, sort_column, reverse)

    headers = [
        "model",
        "model_display_name",
        "reasoning_condition",
        "factor",
        "level_A",
        "level_B",
        "nudge_type",
        "invalid_pct",
        "n_comparisons",
        "n_observations",
        "f_0_B",
        "f_A_B",
        "f_B_B",
        "amce_0",
        "amce_0_se",
        "amce_A",
        "amce_A_se",
        "amce_B",
        "amce_B_se",
        "effect_A",
        "effect_B",
        "sig_amce_0",
        "sig_amce_A",
        "sig_amce_B",
        "sig_effect_A",
        "sig_effect_B",
        "log_n_coef_0",
        "log_n_coef_A",
        "log_n_coef_B",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in results:
            writer.writerow(
                [
                    r.model,
                    get_model_display_name(r.model) if show_display_names else r.model,
                    r.reasoning_condition,
                    r.factor,
                    r.level_A,
                    r.level_B,
                    r.nudge_type,
                    r.invalid_pct,
                    r.n_comparisons,
                    r.n_observations,
                    r.f_0_B,
                    r.f_A_B,
                    r.f_B_B,
                    r.amce_0,
                    r.amce_0_se,
                    r.amce_A,
                    r.amce_A_se,
                    r.amce_B,
                    r.amce_B_se,
                    r.effect_A,
                    r.effect_B,
                    r.sig_amce_0,
                    r.sig_amce_A,
                    r.sig_amce_B,
                    r.sig_effect_A,
                    r.sig_effect_B,
                    r.log_n_coef_0,
                    r.log_n_coef_A,
                    r.log_n_coef_B,
                ]
            )

    print(f"Wrote {len(results)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create AMCE-based summary table of nudge experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Column Definitions:
    f_0(B), f_A(B), f_B(B): Raw frequency of choosing B in each condition
    AMCE_0, AMCE_A, AMCE_B: AMCE of B (vs A) controlling for log(N)
    effect_A: AMCE_0 - AMCE_A (effect of nudging towards A)
    effect_B: AMCE_B - AMCE_0 (effect of nudging towards B)
    log(N): Coefficient for log(N) in baseline regression

AMCE values are on the log-odds scale. Positive values indicate preference for B.
* indicates statistical significance at alpha=0.05.
        """,
    )

    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results"],
        help="List of results directories to search",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="List of models to include"
    )
    parser.add_argument(
        "--factors", nargs="+", default=None, help="List of factors to include"
    )
    parser.add_argument(
        "--nudge-types", nargs="+", default=None, help="List of nudge types to include"
    )
    parser.add_argument(
        "--reasoning",
        nargs="+",
        default=None,
        help="List of reasoning conditions to include",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output CSV file path"
    )
    parser.add_argument(
        "--no-display-names", action="store_true", help="Use raw model names"
    )
    parser.add_argument(
        "--sort", "-s", type=str, default=None, help="Column to sort by"
    )
    parser.add_argument(
        "--reverse", "-r", action="store_true", help="Sort in descending order"
    )
    parser.add_argument(
        "--decimals", "-d", type=int, default=3, help="Number of decimal places"
    )
    parser.add_argument(
        "--hide-columns", nargs="+", default=None, help="List of columns to hide"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("AMCE-Based Nudge Experiment Summary Table")
    print("=" * 80)
    print(f"Results directories: {args.results_dirs}")
    if args.models:
        print(f"Model filter: {args.models}")
    if args.factors:
        print(f"Factor filter: {args.factors}")
    if args.nudge_types:
        print(f"Nudge type filter: {args.nudge_types}")
    if args.reasoning:
        print(f"Reasoning condition filter: {args.reasoning}")
    if args.sort:
        sort_desc = f"Sort by: {args.sort}"
        if args.reverse:
            sort_desc += " (descending)"
        print(sort_desc)
    print("=" * 80)
    print()

    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if args.reasoning:
        results = [r for r in results if r.reasoning_condition in args.reasoning]

    print(f"Found {len(results)} complete experiments\n")

    if not results:
        print("No complete experiments found matching the filters.")
        return

    show_display_names = not args.no_display_names
    decimals = args.decimals
    sort_column = args.sort
    reverse = args.reverse

    if args.output:
        write_csv(results, args.output, show_display_names, sort_column, reverse)
    else:
        print(
            format_table(
                results,
                show_display_names,
                decimals,
                sort_column,
                reverse,
                args.hide_columns,
            )
        )

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    model_groups: Dict[Tuple[str, str], List[AMCEResult]] = defaultdict(list)
    for r in results:
        base_model = get_base_model_name(r.model)
        model_groups[(base_model, r.reasoning_condition)].append(r)

    print(f"\nModels ({len(model_groups)}):")
    for base_model, reasoning_condition in sorted(model_groups.keys()):
        model_results = model_groups[(base_model, reasoning_condition)]
        display_name = (
            get_model_display_name(model_results[0].model)
            if show_display_names
            else base_model
        )

        avg_amce_0 = np.mean([r.amce_0 for r in model_results])
        avg_effect_A = np.mean([r.effect_A for r in model_results])
        avg_effect_B = np.mean([r.effect_B for r in model_results])
        avg_log_n = np.mean([r.log_n_coef_0 for r in model_results])

        sig_amce_0_rate = np.mean([r.sig_amce_0 for r in model_results])
        sig_effect_A_rate = np.mean([r.sig_effect_A for r in model_results])
        sig_effect_B_rate = np.mean([r.sig_effect_B for r in model_results])

        print(
            f"  {display_name} ({reasoning_condition}): n={len(model_results)}, "
            f"AMCE_0={avg_amce_0:+.{decimals}f} (sig={sig_amce_0_rate:.0%}), "
            f"effect_A={avg_effect_A:+.{decimals}f} (sig={sig_effect_A_rate:.0%}), "
            f"effect_B={avg_effect_B:+.{decimals}f} (sig={sig_effect_B_rate:.0%}), "
            f"log(N)={avg_log_n:+.{decimals}f}"
        )

    factor_groups: Dict[str, List[AMCEResult]] = defaultdict(list)
    for r in results:
        factor_groups[r.factor].append(r)

    print(f"\nFactors ({len(factor_groups)}):")
    for factor in sorted(factor_groups.keys()):
        factor_results = factor_groups[factor]
        level_A = factor_results[0].level_A
        level_B = factor_results[0].level_B

        avg_amce_0 = np.mean([r.amce_0 for r in factor_results])
        avg_effect_A = np.mean([r.effect_A for r in factor_results])
        avg_effect_B = np.mean([r.effect_B for r in factor_results])

        print(
            f"  {factor} (A={level_A}, B={level_B}): n={len(factor_results)}, "
            f"AMCE_0={avg_amce_0:+.{decimals}f}, "
            f"effect_A={avg_effect_A:+.{decimals}f}, "
            f"effect_B={avg_effect_B:+.{decimals}f}"
        )

    print(f"\nOverall ({len(results)} experiments):")
    avg_amce_0 = np.mean([r.amce_0 for r in results])
    avg_effect_A = np.mean([r.effect_A for r in results])
    avg_effect_B = np.mean([r.effect_B for r in results])
    avg_abs_effect = np.mean([abs(r.effect_A) + abs(r.effect_B) for r in results]) / 2
    avg_log_n = np.mean([r.log_n_coef_0 for r in results])

    print(f"  Avg AMCE_0 (baseline bias): {avg_amce_0:+.{decimals}f}")
    print(f"  Avg effect_A (nudge->A effect): {avg_effect_A:+.{decimals}f}")
    print(f"  Avg effect_B (nudge->B effect): {avg_effect_B:+.{decimals}f}")
    print(f"  Avg |effect| (steerability): {avg_abs_effect:.{decimals}f}")
    print(f"  Avg log(N) coefficient: {avg_log_n:+.{decimals}f}")


if __name__ == "__main__":
    main()
