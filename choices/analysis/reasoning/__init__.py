"""
Reasoning trace analysis module.

Provides tools for extracting, coding, and analyzing reasoning traces
from experiments that capture model reasoning (with_reasoning=True).

Main components:
- extract_traces: Extract reasoning traces from experiment results
- argument_coder: LLM-based coding of arguments in reasoning traces
- analyze_traces: Statistical analysis of coded traces
- codebook: Codebook data structure for defining arguments
"""

from .codebook import Argument, Codebook
from .extract_traces import ReasoningTrace, extract_reasoning_traces
from .argument_coder import ArgumentCoder, CodedTrace
from .analyze_traces import analyze_coded_traces, TraceAnalysis

__all__ = [
    "Argument",
    "Codebook",
    "ReasoningTrace",
    "extract_reasoning_traces",
    "ArgumentCoder",
    "CodedTrace",
    "analyze_coded_traces",
    "TraceAnalysis",
]
