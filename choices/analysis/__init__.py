"""
Analysis tools for choices experiments.

This module provides tools for analyzing and visualizing experiment results,
including exchange rate plots, reasoning trace analysis, and causal audits.
"""

from .causal_audit import (
    CausalAuditConfig,
    CausalAuditResults,
    CausalAuditRunner,
    CAUSAL_AUDIT_CATEGORIES,
    run_causal_audit,
    run_causal_audit_with_resampling,
    list_categories as list_causal_audit_categories,
)

from .reasoning import (
    Argument,
    Codebook,
    ReasoningTrace,
    extract_reasoning_traces,
    ArgumentCoder,
    CodedTrace,
    analyze_coded_traces,
    TraceAnalysis,
)

__all__ = [
    # Causal audit
    "CausalAuditConfig",
    "CausalAuditResults",
    "CausalAuditRunner",
    "CAUSAL_AUDIT_CATEGORIES",
    "run_causal_audit",
    "run_causal_audit_with_resampling",
    "list_causal_audit_categories",
    # Reasoning analysis
    "Argument",
    "Codebook",
    "ReasoningTrace",
    "extract_reasoning_traces",
    "ArgumentCoder",
    "CodedTrace",
    "analyze_coded_traces",
    "TraceAnalysis",
]
