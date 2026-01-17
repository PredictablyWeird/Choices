"""
Nudge Effect Size calculation.

Effect size measures how much a nudge shifts preferences toward the target option.
The primary metric is the change in selection probability for the nudged option.

Effect Size = P(target | nudge towards target) - P(target | baseline)

This gives a simple, interpretable measure:
- +0.10 means nudging increases selection of the target by 10 percentage points
- 0.00 means no effect
- Negative values indicate the nudge backfired

Additional metrics:
- steerability: log-odds change (from steerability_metric.py)
- average_effect_size: mean effect size across both nudge directions
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class NudgeEffectSize:
    """Effect size results for a single nudge experiment."""

    factor: str
    model: str
    nudge_type: str
    level_A: str
    level_B: str
    # Baseline probabilities
    p_A_baseline: float
    p_B_baseline: float
    # Nudged probabilities
    p_A_nudge_A: float  # P(A | nudge towards A)
    p_B_nudge_B: float  # P(B | nudge towards B)
    # Effect sizes (change in probability)
    effect_size_A: float  # p_A_nudge_A - p_A_baseline
    effect_size_B: float  # p_B_nudge_B - p_B_baseline
    # Average effect size (signed mean)
    avg_effect_size: float  # (effect_size_A + effect_size_B) / 2
    # Average magnitude of effect size (absolute values)
    avg_effect_magnitude: float  # (|effect_size_A| + |effect_size_B|) / 2
    # Sample sizes
    n_comparisons_baseline: int
    n_comparisons_nudge_A: int
    n_comparisons_nudge_B: int


def compute_factor_preference_from_graph(
    graph_data: Dict[str, Any],
    factor_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute preference probability for each factor level from graph data.

    Only considers inter-group comparisons (different factor levels).

    Args:
        graph_data: Loaded preference graph data
        factor_name: Name of the factor variable (e.g., 'social_status', 'age_group')

    Returns:
        Dictionary mapping level -> {'prob': float, 'wins': float, 'total': int}
    """
    options = graph_data.get("options", [])
    edges = graph_data.get("edges", {})

    # Build option lookup
    options_by_id = {opt["id"]: opt for opt in options}

    # Get all factor levels
    factor_levels = set()
    for opt in options:
        if factor_name in opt:
            factor_levels.add(opt[factor_name])

    # Track wins and total for each level
    level_stats = {level: {"wins": 0.0, "total": 0} for level in factor_levels}

    for edge_key, edge_data in edges.items():
        try:
            ids = eval(edge_key)
            opt_a = options_by_id.get(ids[0])
            opt_b = options_by_id.get(ids[1])

            if not opt_a or not opt_b:
                continue

            level_a = opt_a.get(factor_name)
            level_b = opt_b.get(factor_name)

            # Skip intra-group comparisons
            if level_a == level_b:
                continue

            # Get response data
            aux_data = edge_data.get("aux_data", {})
            total_responses = aux_data.get("total_responses", 0)

            if total_responses == 0:
                continue

            prob_a = edge_data.get("probability_A", 0.5)

            # Update stats for both levels
            if level_a in level_stats:
                level_stats[level_a]["wins"] += prob_a * total_responses
                level_stats[level_a]["total"] += total_responses

            if level_b in level_stats:
                level_stats[level_b]["wins"] += (1 - prob_a) * total_responses
                level_stats[level_b]["total"] += total_responses

        except Exception:
            continue

    # Compute probabilities
    result = {}
    for level, stats in level_stats.items():
        if stats["total"] > 0:
            prob = stats["wins"] / stats["total"]
        else:
            prob = 0.5
        result[level] = {
            "prob": prob,
            "wins": stats["wins"],
            "total": stats["total"],
        }

    return result


def load_preference_graph(result_dir: Path) -> Optional[Dict[str, Any]]:
    """Load preference graph from a result directory."""
    graph_files = list(result_dir.glob("preference_graph_*.json"))
    if not graph_files:
        return None

    with open(graph_files[0], "r") as f:
        return json.load(f)


def get_factor_name_from_graph(graph_data: Dict[str, Any]) -> Optional[str]:
    """Extract the factor variable name from graph data (non-N variable)."""
    variables = graph_data.get("variables", [])
    for var in variables:
        if var.get("name") != "N":
            return var["name"]
    return None


def get_factor_levels_from_graph(graph_data: Dict[str, Any]) -> List[str]:
    """Extract factor levels from graph data."""
    factor_name = get_factor_name_from_graph(graph_data)
    if not factor_name:
        return []

    variables = graph_data.get("variables", [])
    for var in variables:
        if var.get("name") == factor_name:
            return var.get("values", [])
    return []


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
) -> Dict[str, Path]:
    """
    Find result directories for each condition (base, and each nudge target).

    Returns:
        Dictionary mapping condition name -> Path to result directory
        e.g., {'base': Path(...), 'low': Path(...), 'high': Path(...)}
    """
    experiment_name = f"simple_{factor_name}"
    base_path = Path(results_base_dir) / experiment_name / model / nudge_type

    if not base_path.exists():
        return {}

    result_dirs = {}
    dirs_by_condition: Dict[str, List[Path]] = {}

    for result_dir in base_path.iterdir():
        if not result_dir.is_dir():
            continue

        # Check if this is a base condition
        if result_dir.name.endswith("_base"):
            condition = "base"
        else:
            # Get target group from graph data
            condition = get_nudge_target_group(result_dir)
            if not condition:
                continue

        if condition not in dirs_by_condition:
            dirs_by_condition[condition] = []
        dirs_by_condition[condition].append(result_dir)

    # For each condition, use the most recent directory
    for condition, dirs in dirs_by_condition.items():
        most_recent = max(dirs, key=lambda d: d.stat().st_mtime)
        result_dirs[condition] = most_recent

    return result_dirs


def compute_nudge_effect_size(
    factor_name: str,
    model: str,
    nudge_type: str,
    results_base_dir: str = "results",
) -> Optional[NudgeEffectSize]:
    """
    Compute nudge effect size for a single experiment.

    Args:
        factor_name: Name of the factor (e.g., 'social_status')
        model: Model name
        nudge_type: Type of nudge (e.g., 'weak_evidence')
        results_base_dir: Base directory for results

    Returns:
        NudgeEffectSize object or None if data is insufficient
    """
    # Find all condition directories
    condition_dirs = find_condition_directories(
        factor_name, model, nudge_type, results_base_dir
    )

    if "base" not in condition_dirs:
        return None

    # Load baseline data
    base_graph = load_preference_graph(condition_dirs["base"])
    if not base_graph:
        return None

    # Get factor info
    factor_var_name = get_factor_name_from_graph(base_graph)
    if not factor_var_name:
        return None

    factor_levels = get_factor_levels_from_graph(base_graph)
    if len(factor_levels) != 2:
        # For now, only support binary factors
        return None

    level_A, level_B = factor_levels[0], factor_levels[1]

    # Check we have nudge conditions for both levels
    if level_A not in condition_dirs or level_B not in condition_dirs:
        return None

    # Compute baseline preferences
    base_prefs = compute_factor_preference_from_graph(base_graph, factor_var_name)
    p_A_baseline = base_prefs.get(level_A, {}).get("prob", 0.5)
    p_B_baseline = base_prefs.get(level_B, {}).get("prob", 0.5)
    n_baseline = base_prefs.get(level_A, {}).get("total", 0)

    # Load and compute nudge-A condition
    nudge_A_graph = load_preference_graph(condition_dirs[level_A])
    if not nudge_A_graph:
        return None

    nudge_A_prefs = compute_factor_preference_from_graph(nudge_A_graph, factor_var_name)
    p_A_nudge_A = nudge_A_prefs.get(level_A, {}).get("prob", 0.5)
    n_nudge_A = nudge_A_prefs.get(level_A, {}).get("total", 0)

    # Load and compute nudge-B condition
    nudge_B_graph = load_preference_graph(condition_dirs[level_B])
    if not nudge_B_graph:
        return None

    nudge_B_prefs = compute_factor_preference_from_graph(nudge_B_graph, factor_var_name)
    p_B_nudge_B = nudge_B_prefs.get(level_B, {}).get("prob", 0.5)
    n_nudge_B = nudge_B_prefs.get(level_B, {}).get("total", 0)

    # Compute effect sizes
    effect_size_A = p_A_nudge_A - p_A_baseline
    effect_size_B = p_B_nudge_B - p_B_baseline
    avg_effect_size = (effect_size_A + effect_size_B) / 2
    avg_effect_magnitude = (abs(effect_size_A) + abs(effect_size_B)) / 2

    return NudgeEffectSize(
        factor=factor_name,
        model=model,
        nudge_type=nudge_type,
        level_A=level_A,
        level_B=level_B,
        p_A_baseline=p_A_baseline,
        p_B_baseline=p_B_baseline,
        p_A_nudge_A=p_A_nudge_A,
        p_B_nudge_B=p_B_nudge_B,
        effect_size_A=effect_size_A,
        effect_size_B=effect_size_B,
        avg_effect_size=avg_effect_size,
        avg_effect_magnitude=avg_effect_magnitude,
        n_comparisons_baseline=n_baseline,
        n_comparisons_nudge_A=n_nudge_A,
        n_comparisons_nudge_B=n_nudge_B,
    )


def discover_experiments(
    results_base_dir: str = "results",
) -> List[Tuple[str, str, str]]:
    """
    Discover all available experiments in the results directory.

    Returns:
        List of (factor_name, model, nudge_type) tuples
    """
    results_path = Path(results_base_dir)
    experiments = []

    if not results_path.exists():
        return experiments

    # Iterate through experiment directories (simple_{factor})
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("simple_"):
            continue

        factor_name = exp_dir.name[7:]  # Remove 'simple_' prefix

        # Iterate through model directories
        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name

            # Iterate through nudge type directories
            for nudge_dir in model_dir.iterdir():
                if not nudge_dir.is_dir():
                    continue

                nudge_type = nudge_dir.name
                experiments.append((factor_name, model, nudge_type))

    return experiments


def compute_all_effect_sizes(
    results_base_dir: str = "results",
) -> List[NudgeEffectSize]:
    """
    Compute effect sizes for all available experiments.

    Returns:
        List of NudgeEffectSize objects
    """
    experiments = discover_experiments(results_base_dir)
    effect_sizes = []

    for factor_name, model, nudge_type in experiments:
        effect_size = compute_nudge_effect_size(
            factor_name, model, nudge_type, results_base_dir
        )
        if effect_size is not None:
            effect_sizes.append(effect_size)

    return effect_sizes


def aggregate_effect_sizes_by_nudge_type(
    effect_sizes: List[NudgeEffectSize],
    use_magnitude: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate effect sizes by nudge type (averaging across models and factors).

    Args:
        effect_sizes: List of NudgeEffectSize objects
        use_magnitude: If True (default), use absolute value of effect sizes.
                       If False, use signed effect sizes.

    Returns:
        Dictionary mapping nudge_type -> {
            'avg_effect_size': float,
            'std_effect_size': float,
            'n_experiments': int,
            'effect_sizes': list of individual effect sizes
        }
    """
    from collections import defaultdict
    import statistics

    by_nudge_type: Dict[str, List[float]] = defaultdict(list)

    for es in effect_sizes:
        value = es.avg_effect_magnitude if use_magnitude else es.avg_effect_size
        by_nudge_type[es.nudge_type].append(value)

    result = {}
    for nudge_type, sizes in by_nudge_type.items():
        avg = statistics.mean(sizes) if sizes else 0.0
        std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0

        result[nudge_type] = {
            "avg_effect_size": avg,
            "std_effect_size": std,
            "n_experiments": len(sizes),
            "effect_sizes": sizes,
        }

    return result
