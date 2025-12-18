"""
Variable definitions for experiments.

Variables define the dimensions along which options vary.
"""

from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from enum import Enum


class AnalysisType(Enum):
    """Analysis type for a field in an option."""

    CATEGORICAL = "categorical"  # Discrete categories (gender, country, etc.)
    NUMERICAL = "numerical"  # Numeric values (N, age, etc.) - simple difference
    LOG_NUMERICAL = (
        "log_numerical"  # Numeric values with log transform (for diminishing returns)
    )


@dataclass
class AnalysisConfig:
    """
    Configuration for analyzing experiment results.

    Defines which fields from options should be considered for analysis
    and how they should be analyzed (categorical, numerical, log_numerical).

    Attributes:
        fields: Dictionary mapping field names to their analysis types
                e.g., {'gender': AnalysisType.CATEGORICAL, 'N': AnalysisType.LOG_NUMERICAL}
    """

    fields: Dict[str, AnalysisType] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {"fields": {name: atype.value for name, atype in self.fields.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """Create from dictionary representation."""
        fields = {}
        if "fields" in data:
            for name, atype_str in data["fields"].items():
                fields[name] = AnalysisType(atype_str)
        return cls(fields=fields)

    def get_analysis_type(self, field_name: str) -> Optional[AnalysisType]:
        """Get analysis type for a field, or None if not configured."""
        return self.fields.get(field_name)


@dataclass
class Variable:
    """
    A variable that can take different values in an experiment.

    Attributes:
        name: Variable name (e.g., 'gender', 'N', 'country')
        values: List of possible values this variable can take
        description: Optional human-readable description
    """

    name: str
    values: List[Any]
    description: str = ""

    def __post_init__(self):
        """Validate variable configuration."""
        if not self.values:
            raise ValueError(f"Variable '{self.name}' must have at least one value")

    def __len__(self) -> int:
        """Return number of possible values."""
        return len(self.values)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "values": self.values,
            "description": self.description,
        }
