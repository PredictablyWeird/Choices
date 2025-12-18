"""
Clean interface classes for experiment results.

Provides structured access to experiment data while maintaining backwards
compatibility with existing analysis scripts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .variable import Variable, AnalysisConfig


@dataclass
class ExperimentOption:
    """
    Represents a single option in the experiment.

    Keeps the structure simple - just id, description, and any additional fields.
    """

    id: Any
    label: str = field(default_factory=lambda: f"Option {id}")
    _extra_fields: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Allow setting extra fields as attributes."""
        for key, value in self._extra_fields.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for serialization."""
        result = {"id": self.id, "label": self.label}
        result.update(self._extra_fields)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentOption":
        """Create from dictionary."""
        data = data.copy()
        option_id = data.pop("id")
        option_label = data.pop("label", f"Option {option_id}")
        return cls(id=option_id, label=option_label, _extra_fields=data)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for backwards compatibility."""
        if key == "id":
            return self.id
        elif key == "label":
            return self.label
        else:
            return self._extra_fields.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        if key == "id":
            return self.id
        elif key == "label":
            return self.label
        else:
            return self._extra_fields.get(key, default)


@dataclass
class PreferenceGraphResults:
    """
    Raw preference comparison data from the graph.

    This is the "data" - the actual preferences elicited from the agent.
    You could fit multiple utility models to this same graph data.
    """

    options: List[ExperimentOption]
    edges: Dict[str, Dict[str, Any]]  # Raw edge data from graph.export_data()
    training_edges: List[List[Any]]
    holdout_edges: Optional[List[List[Any]]] = None
    variables: List[Variable] = field(default_factory=list)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "options": [opt.to_dict() for opt in self.options],
            "edges": self.edges,
            "training_edges": self.training_edges,
            "holdout_edges": self.holdout_edges,
            "variables": [var.to_dict() for var in self.variables],
            "analysis_config": self.analysis_config.to_dict(),
            **self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceGraphResults":
        """Create from dictionary."""
        options = [ExperimentOption.from_dict(opt) for opt in data["options"]]

        # Extract variable metadata
        if "variables" in data:
            variables = [
                Variable(
                    name=var_data["name"],
                    values=var_data["values"],
                    description=var_data.get("description", ""),
                )
                for var_data in data["variables"]
            ]
        else:
            variables = []

        # Extract analysis config
        if "analysis_config" in data:
            analysis_config = AnalysisConfig.from_dict(data["analysis_config"])
        else:
            analysis_config = AnalysisConfig()

        # Extract config fields
        config_keys = [
            "compute_utilities_config",
            "create_agent_config",
            "preference_graph_arguments",
        ]
        config = {k: data[k] for k in config_keys if k in data}

        return cls(
            options=options,
            edges=data.get("edges", {}),
            training_edges=data.get("training_edges", []),
            holdout_edges=data.get("holdout_edges"),
            variables=variables,
            analysis_config=analysis_config,
            config=config,
        )

    def get_variable(self, name: str) -> Optional[Variable]:
        """Get variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None

    def get_categorical_variables(self) -> Dict[str, Variable]:
        """Get all categorical variables (useful for finding factors)."""
        from .variable import AnalysisType

        categorical_vars = {}
        for var in self.variables:
            atype = self.analysis_config.get_analysis_type(var.name)
            if atype == AnalysisType.CATEGORICAL:
                categorical_vars[var.name] = var
        return categorical_vars

    def get_numerical_variables(self) -> Dict[str, Variable]:
        """Get all numerical variables (useful for finding N, age, etc.)."""
        from .variable import AnalysisType

        numerical_vars = {}
        for var in self.variables:
            atype = self.analysis_config.get_analysis_type(var.name)
            if atype in (AnalysisType.NUMERICAL, AnalysisType.LOG_NUMERICAL):
                numerical_vars[var.name] = var
        return numerical_vars

    def save(self, save_dir: str, filename: str = "preference_graph.json") -> None:
        """Save preference graph data."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "PreferenceGraphResults":
        """Load from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class UtilityModelResults:
    """
    Results from fitting a utility model to preference data.

    This is the "model" - one interpretation of the preference graph data.
    Keeps utilities as simple dicts: {option_id: {"mean": float, "variance": float}}
    """

    utilities: Dict[
        Any, Dict[str, float]
    ]  # Maps option ID to {"mean": ..., "variance": ...}
    training_metrics: Dict[str, float]
    holdout_metrics: Optional[Dict[str, float]] = None
    model_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "utilities": self.utilities,  # Keys are already strings
            "metrics": self.training_metrics,
        }
        if self.holdout_metrics is not None:
            result["holdout_metrics"] = self.holdout_metrics
        if self.model_config:
            result.update(self.model_config)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtilityModelResults":
        """Create from dictionary."""
        # Extract config fields
        config_keys = ["utility_model_class", "utility_model_arguments"]
        config = {k: data[k] for k in config_keys if k in data}

        return cls(
            utilities=data["utilities"],
            training_metrics=data.get("metrics", {}),
            holdout_metrics=data.get("holdout_metrics"),
            model_config=config,
        )

    def save(self, save_dir: str, filename: str = "utility_model.json") -> None:
        """Save utility model results."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        with open(save_path / filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "UtilityModelResults":
        """Load from JSON file."""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_sorted_utilities(
        self, reverse: bool = True
    ) -> List[tuple[Any, Dict[str, float]]]:
        """
        Get utilities sorted by mean.

        Args:
            reverse: If True, sort from highest to lowest utility

        Returns:
            List of (option_id, utility_dict) tuples sorted by mean
        """
        return sorted(
            self.utilities.items(), key=lambda x: x[1]["mean"], reverse=reverse
        )


@dataclass
class ExperimentResults:
    """
    Combined results from an experiment.

    Separates the raw preference data (graph) from the fitted utility model.
    Preserves variable metadata so analysis scripts know about factor types.

    This allows you to:
    - Save/load graph and model separately
    - Fit multiple utility models to the same graph data
    - Maintain clean separation of concerns
    - Preserve variable type information for analysis
    """

    graph: PreferenceGraphResults
    utility_model: UtilityModelResults

    def save(self, save_dir: str, save_suffix: str) -> None:
        """
        Save results to JSON files.

        Creates two files with no redundant information:
        - preference_graph_{suffix}.json - Options, variables, edges, training/holdout splits, configs
        - utility_model_{suffix}.json - Fitted utilities, metrics, model config

        Analysis scripts should load both files to get complete information.

        Args:
            save_dir: Directory to save to
            save_suffix: Suffix for filenames
        """
        # Import convert_numpy here to avoid circular imports
        import numpy as np

        def convert_numpy(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. Save graph data (can be reused for fitting other models)
        #    Contains: options, edges, training/holdout splits, variables, configs
        graph_dict = convert_numpy(self.graph.to_dict())
        graph_path = save_path / f"preference_graph_{save_suffix}.json"
        with open(graph_path, "w") as f:
            json.dump(graph_dict, f, indent=2)

        # 2. Save utility model (just the fitted model)
        #    Contains: utilities, metrics, model config
        model_dict = convert_numpy(self.utility_model.to_dict())
        model_path = save_path / f"utility_model_{save_suffix}.json"
        with open(model_path, "w") as f:
            json.dump(model_dict, f, indent=2)

    @classmethod
    def load(cls, dir_path: str, suffix: str) -> "ExperimentResults":
        """
        Load results from a directory.

        Args:
            dir_path: Directory containing the results files
            suffix: The save suffix (e.g., "gpt-4o-mini_thurstonianactivelearningutilitymodel")

        Returns:
            ExperimentResults object

        Example:
            results = ExperimentResults.load("results/my_exp/gpt-4o-mini/20251117_120000",
                                            "gpt-4o-mini_thurstonianactivelearningutilitymodel")
        """
        dir_path_obj = Path(dir_path)
        graph_file = dir_path_obj / f"preference_graph_{suffix}.json"
        utility_file = dir_path_obj / f"utility_model_{suffix}.json"

        if not graph_file.exists() or not utility_file.exists():
            raise FileNotFoundError(
                f"Could not find both preference_graph_{suffix}.json and "
                f"utility_model_{suffix}.json in {dir_path}"
            )

        # Load both components
        graph = PreferenceGraphResults.load(str(graph_file))
        utility_model = UtilityModelResults.load(str(utility_file))

        return cls(graph=graph, utility_model=utility_model)

    def get_option_by_id(self, option_id: Any) -> Optional[ExperimentOption]:
        """Get an option by its ID."""
        for opt in self.graph.options:
            if opt.id == option_id:
                return opt
        return None

    def get_sorted_results(
        self, reverse: bool = True
    ) -> List[tuple[ExperimentOption, Dict[str, float]]]:
        """
        Get options with utilities sorted by utility mean.

        Args:
            reverse: If True, sort from highest to lowest utility

        Returns:
            List of (option, utility) tuples sorted by utility mean
        """
        results = []
        for opt in self.graph.options:
            opt_id_str = str(opt.id)
            if opt_id_str in self.utility_model.utilities:
                results.append((opt, self.utility_model.utilities[opt_id_str]))

        results.sort(key=lambda x: x[1]["mean"], reverse=reverse)
        return results
