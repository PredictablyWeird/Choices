"""
Choices: A simple framework for preference elicitation experiments.

Provides the Experiment class for defining and running experiments.
"""

__version__ = "0.0.1"

from .experiment import (
    Experiment,
    ExperimentConfig,
    PromptConfig,
)
from .llm_agent import LLMResponse
from .results import (
    ExperimentOption,
    ExperimentResults,
    PreferenceGraphResults,
    UtilityModelResults,
)
from .variable import (
    AnalysisConfig,
    AnalysisType,
    ReasoningMode,
    Variable,
)

__all__ = [
    "Experiment",
    "ExperimentConfig",
    "PromptConfig",
    "Variable",
    "AnalysisConfig",
    "AnalysisType",
    "ReasoningMode",
    "ExperimentResults",
    "PreferenceGraphResults",
    "UtilityModelResults",
    "ExperimentOption",
    "LLMResponse",
]
