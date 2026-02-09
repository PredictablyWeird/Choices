#!/usr/bin/env python3
"""
Analyze extreme baseline decisions: cases where REASONING models choose to save
FEWER people of one demographic group over MORE people of another,
with no contextual influence applied.

Only includes models with reasoning traces (excludes all non-reasoning models).

Outputs a readable report showing each extreme decision with its
full reasoning traces, sorted by extremity.

Usage:
    # Basic report to stdout
    uv run python -m choices.analysis.analyze_extreme_baseline_decisions \
        --results-dirs results_main0 results_main1

    # Save to file + interactive HTML
    uv run python -m choices.analysis.analyze_extreme_baseline_decisions \
        --results-dirs results_main0 results_main1 \
        --output extreme_baseline_report.txt \
        --html-output extreme_baseline_report.html

    # Also save structured JSON
    uv run python -m choices.analysis.analyze_extreme_baseline_decisions \
        --results-dirs results_main0 results_main1 \
        --output extreme_baseline_report.txt \
        --json-output extreme_baseline_data.json

    # Adjust N-difference threshold (default: 2)
    uv run python -m choices.analysis.analyze_extreme_baseline_decisions \
        --results-dirs results_main0 results_main1 \
        --n-diff-threshold 3

    # Filter by model or factor
    uv run python -m choices.analysis.analyze_extreme_baseline_decisions \
        --results-dirs results_main0 results_main1 \
        --models grok-41-fast-reasoning deepseek-v3-2-reasoning \
        --factors age_group wealth
"""

import argparse
import html as html_mod
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

from choices.analysis.create_summary import (
    discover_experiments,
    find_condition_directories,
)
from choices.analysis.nudge_effect_size import (
    get_factor_name_from_graph,
    load_preference_graph,
)

# Models that are natively non-reasoning (no reasoning traces at all).
# Models NOT in this set, or models in this set that happen to have
# reasoning_mode="before" (CoT prompt), will be included if they have traces.
NON_REASONING_MODELS = {
    "deepseek-v3-2-non-reasoning",
    "gpt-5-2-non-reasoning",
    "grok-41-fast-non-reasoning",
}


@dataclass
class ExtremeDecision:
    """A single extreme baseline decision with metadata."""

    model: str
    factor: str
    nudge_type: str  # which nudge_type directory the baseline was under
    chosen_group: str
    chosen_n: int
    rejected_group: str
    rejected_n: int
    n_diff: int  # rejected_n - chosen_n (positive = chose fewer)
    probability: float  # probability of choosing the smaller group
    count_chosen: int
    count_rejected: int
    total_responses: int
    edge_key: str
    reasoning_traces: list = field(default_factory=list)


def get_reasoning_traces(aux_data: dict) -> tuple[list[str], list[str]]:
    """Extract reasoning traces from aux_data, handling both field name conventions.

    IMPORTANT: Does NOT filter out empty entries — index alignment with
    original_parsed / flipped_parsed must be preserved for the zip() pairing
    in extract_extreme_baseline_decisions().
    """
    original = aux_data.get("original_reasoning", [])
    if not original:
        original = aux_data.get("original_reasoning_summaries", [])

    flipped = aux_data.get("flipped_reasoning", [])
    if not flipped:
        flipped = aux_data.get("flipped_reasoning_summaries", [])

    return original or [], flipped or []


def is_non_reasoning_model(model_name: str) -> bool:
    """Check if a model is explicitly non-reasoning (never has traces)."""
    return model_name in NON_REASONING_MODELS


def extract_extreme_baseline_decisions(
    results_dirs: list[str],
    n_diff_threshold: int = 2,
    model_filter: Optional[list[str]] = None,
    factor_filter: Optional[list[str]] = None,
) -> list[ExtremeDecision]:
    """Extract all extreme baseline decisions from results directories.

    Only includes reasoning models.
    """
    experiments = discover_experiments(
        results_dirs,
        model_filter=model_filter,
        factor_filter=factor_filter,
    )

    all_decisions: list[ExtremeDecision] = []
    # Track which (model, factor, edge_key) combos we've already seen
    # so we don't double-count the same baseline across nudge_type dirs
    seen = set()

    for results_dir, factor, model, nudge_type in experiments:
        # Skip models that are explicitly non-reasoning (never have traces)
        if is_non_reasoning_model(model):
            continue

        condition_dirs_list = find_condition_directories(
            factor, model, nudge_type, results_dir
        )

        for condition_dirs in condition_dirs_list:
            base_path = condition_dirs.get("base")
            if not base_path:
                continue

            graph_data = load_preference_graph(base_path)
            if not graph_data:
                continue

            # Get reasoning mode to distinguish e.g. llama(none) vs llama(before)
            cfg = graph_data.get("simple_experiment_config", {})
            reasoning_mode = cfg.get("reasoning_mode", "none")

            # Build a display-friendly model name that includes reasoning mode
            model_display = model
            if reasoning_mode not in ("none", "N/A", ""):
                model_display = f"{model} ({reasoning_mode})"

            options = graph_data.get("options", [])
            edges = graph_data.get("edges", {})
            options_by_id = {opt["id"]: opt for opt in options}

            factor_name = get_factor_name_from_graph(graph_data)
            if not factor_name:
                continue

            for edge_key, edge_data in edges.items():
                try:
                    ids = eval(edge_key)
                    opt_a = options_by_id.get(ids[0])
                    opt_b = options_by_id.get(ids[1])
                    if not opt_a or not opt_b:
                        continue

                    n_a = opt_a.get("N", 0)
                    n_b = opt_b.get("N", 0)
                    group_a = opt_a.get(factor_name, "")
                    group_b = opt_b.get(factor_name, "")

                    # Skip same-group or equal-N comparisons
                    if group_a == group_b or n_a == n_b:
                        continue

                    aux_data = edge_data.get("aux_data", {})
                    prob_a = edge_data.get("probability_A", 0.5)
                    count_a = aux_data.get("count_A", 0)
                    count_b = aux_data.get("count_B", 0)
                    total = aux_data.get("total_responses", 0)

                    # Identify the smaller and larger groups
                    if n_a < n_b:
                        chosen_group, chosen_n = group_a, n_a
                        rejected_group, rejected_n = group_b, n_b
                        n_diff = n_b - n_a
                        probability = prob_a  # prob of choosing smaller (A)
                        count_chosen = int(count_a)
                        count_rejected = int(count_b)
                    else:
                        chosen_group, chosen_n = group_b, n_b
                        rejected_group, rejected_n = group_a, n_a
                        n_diff = n_a - n_b
                        probability = 1.0 - prob_a  # prob of choosing smaller (B)
                        count_chosen = int(count_b)
                        count_rejected = int(count_a)

                    if n_diff < n_diff_threshold:
                        continue

                    # Dedup: same model/factor/edge across nudge_type dirs
                    dedup_key = (
                        model_display,
                        factor,
                        chosen_group,
                        chosen_n,
                        rejected_group,
                        rejected_n,
                    )
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Extract reasoning traces — skip edges with no reasoning at all
                    original_reasoning, flipped_reasoning = get_reasoning_traces(
                        aux_data
                    )
                    if not any(r for r in original_reasoning if r) and not any(
                        r for r in flipped_reasoning if r
                    ):
                        continue
                    original_parsed = aux_data.get("original_parsed", [])
                    flipped_parsed = aux_data.get("flipped_parsed", [])

                    # Pair traces with choices, keeping only those that chose the smaller group
                    traces = []
                    for reasoning, choice in zip(original_reasoning, original_parsed):
                        if not reasoning:
                            continue
                        # In original order: A chose opt_a, B chose opt_b
                        if n_a < n_b and choice == "A":
                            traces.append(reasoning)
                        elif n_b < n_a and choice == "B":
                            traces.append(reasoning)

                    for reasoning, choice in zip(flipped_reasoning, flipped_parsed):
                        if not reasoning:
                            continue
                        # In flipped order: A chose opt_b, B chose opt_a
                        if n_a < n_b and choice == "B":
                            traces.append(reasoning)
                        elif n_b < n_a and choice == "A":
                            traces.append(reasoning)

                    # Only include if at least one trace chose smaller
                    if not traces:
                        continue

                    all_decisions.append(
                        ExtremeDecision(
                            model=model_display,
                            factor=factor,
                            nudge_type=nudge_type,
                            chosen_group=chosen_group,
                            chosen_n=chosen_n,
                            rejected_group=rejected_group,
                            rejected_n=rejected_n,
                            n_diff=n_diff,
                            probability=probability,
                            count_chosen=count_chosen,
                            count_rejected=count_rejected,
                            total_responses=total,
                            edge_key=edge_key,
                            reasoning_traces=traces,
                        )
                    )

                except Exception:
                    continue

    # Sort by extremity (largest N diff first), then by probability
    all_decisions.sort(key=lambda d: (-d.n_diff, -d.probability, d.model, d.factor))
    return all_decisions


# ============================================================================
# Text report formatting
# ============================================================================


def format_summary_table(decisions: list[ExtremeDecision]) -> str:
    """Generate a summary table of extreme decisions by model and factor."""
    lines = []

    # Group by model x factor
    stats = defaultdict(lambda: {"count": 0, "max_n_diff": 0, "n_diffs": []})
    for d in decisions:
        key = (d.model, d.factor)
        stats[key]["count"] += 1
        stats[key]["max_n_diff"] = max(stats[key]["max_n_diff"], d.n_diff)
        stats[key]["n_diffs"].append(d.n_diff)

    lines.append("=" * 90)
    lines.append("SUMMARY: EXTREME BASELINE DECISIONS (reasoning models only)")
    lines.append(f"Total: {len(decisions)} extreme decisions found")
    lines.append("=" * 90)
    lines.append("")

    # Overall stats by model
    by_model = defaultdict(list)
    for d in decisions:
        by_model[d.model].append(d)

    lines.append(f"{'Model':<40} {'Count':>6} {'Max N-diff':>10} {'Avg Prob':>10}")
    lines.append("-" * 70)
    for model in sorted(by_model.keys()):
        decs = by_model[model]
        max_diff = max(d.n_diff for d in decs)
        avg_prob = sum(d.probability for d in decs) / len(decs)
        lines.append(f"{model:<40} {len(decs):>6} {max_diff:>10} {avg_prob:>10.1%}")

    lines.append("")
    lines.append(f"{'Model':<40} {'Factor':<15} {'Count':>6} {'Max N-diff':>10}")
    lines.append("-" * 75)

    for model, factor in sorted(stats.keys()):
        s = stats[(model, factor)]
        lines.append(f"{model:<40} {factor:<15} {s['count']:>6} {s['max_n_diff']:>10}")

    # Summary by factor: which group is preferred
    lines.append("")
    lines.append("DIRECTION OF EXTREME CHOICES BY FACTOR:")
    lines.append("-" * 75)
    by_factor = defaultdict(lambda: defaultdict(int))
    for d in decisions:
        by_factor[d.factor][d.chosen_group] += 1

    for factor in sorted(by_factor.keys()):
        groups = by_factor[factor]
        parts = [
            f"{group}: {count}"
            for group, count in sorted(groups.items(), key=lambda x: -x[1])
        ]
        lines.append(f"  {factor:<20} {', '.join(parts)}")

    return "\n".join(lines)


def format_decision_detail(d: ExtremeDecision, index: int) -> str:
    """Format a single extreme decision with its full reasoning traces."""
    lines = []

    lines.append(f"--- Decision #{index + 1} ---")
    lines.append(f"Model:    {d.model}")
    lines.append(f"Factor:   {d.factor}")
    lines.append(
        f"Choice:   Save {d.chosen_n} {d.chosen_group} OVER {d.rejected_n} {d.rejected_group}"
    )
    lines.append(f"N-diff:   {d.n_diff} (rejected {d.n_diff} more people)")
    lines.append(
        f"Strength: {d.probability:.1%} ({d.count_chosen}/{d.count_chosen + d.count_rejected} responses)"
    )

    if d.reasoning_traces:
        lines.append(
            f"Reasoning traces ({len(d.reasoning_traces)} traces that chose smaller group):"
        )
        for i, trace in enumerate(d.reasoning_traces):
            lines.append(f"  [Trace {i + 1}]")
            for line in trace.split("\n"):
                lines.append(f"    {line}")
            lines.append("")
    else:
        lines.append("  (No reasoning traces extracted for this edge)")

    return "\n".join(lines)


def format_full_report(decisions: list[ExtremeDecision], n_diff_threshold: int) -> str:
    """Format the complete text report."""
    lines = []

    lines.append(format_summary_table(decisions))
    lines.append("")
    lines.append("")
    lines.append("=" * 90)
    lines.append("DETAILED DECISIONS (sorted by extremity)")
    lines.append(f"Threshold: N-diff >= {n_diff_threshold}")
    lines.append("=" * 90)
    lines.append("")

    # Group by factor for readability
    by_factor = defaultdict(list)
    for d in decisions:
        by_factor[d.factor].append(d)

    global_idx = 0
    for factor in sorted(by_factor.keys()):
        factor_decisions = by_factor[factor]
        lines.append(f"\n{'#' * 80}")
        lines.append(f"# FACTOR: {factor} ({len(factor_decisions)} extreme decisions)")
        lines.append(f"{'#' * 80}\n")

        for d in factor_decisions:
            lines.append(format_decision_detail(d, global_idx))
            lines.append("")
            global_idx += 1

    return "\n".join(lines)


# ============================================================================
# HTML report formatting
# ============================================================================


def format_html_report(decisions: list[ExtremeDecision], n_diff_threshold: int) -> str:
    """Generate an interactive HTML report with tabs for Stats and Decisions.

    Stats tab dynamically updates based on current filter selection.
    """
    esc = html_mod.escape

    all_models = sorted({d.model for d in decisions})
    all_factors = sorted({d.factor for d in decisions})
    max_ndiff = max((d.n_diff for d in decisions), default=n_diff_threshold)

    # Serialize decision data for JS
    decisions_json = json.dumps(
        [
            {
                "model": d.model,
                "factor": d.factor,
                "chosen_group": d.chosen_group,
                "chosen_n": d.chosen_n,
                "rejected_group": d.rejected_group,
                "rejected_n": d.rejected_n,
                "n_diff": d.n_diff,
                "probability": d.probability,
                "count_chosen": d.count_chosen,
                "count_rejected": d.count_rejected,
                "num_traces": len(d.reasoning_traces),
            }
            for d in decisions
        ]
    )

    # Build decision cards HTML
    cards_html = ""
    for idx, d in enumerate(decisions):
        color = _severity_color(d.n_diff)
        traces_html = ""
        if d.reasoning_traces:
            trace_items = ""
            for ti, trace in enumerate(d.reasoning_traces):
                text = esc(trace)
                trace_items += (
                    f'<details style="margin:4px 0"><summary style="cursor:pointer;color:#1565c0">'
                    f"Trace {ti + 1} ({len(trace):,} chars)</summary>"
                    f'<pre style="white-space:pre-wrap;word-break:break-word;background:#f5f5f5;'
                    f"padding:10px;border-radius:4px;font-size:0.85em;max-height:600px;overflow-y:auto;"
                    f'line-height:1.5">'
                    f"{text}</pre></details>\n"
                )
            traces_html = (
                f'<div style="margin-top:8px"><strong>Reasoning traces '
                f"({len(d.reasoning_traces)} that chose smaller group):</strong>\n"
                f"{trace_items}</div>"
            )
        else:
            traces_html = '<div style="margin-top:8px;color:#888"><em>No reasoning traces extracted for this edge</em></div>'

        prob_pct = d.probability * 100
        cards_html += f"""
<div class="card" data-model="{esc(d.model)}" data-factor="{esc(d.factor)}" data-ndiff="{d.n_diff}" data-idx="{idx}">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px">
    <div>
      <span class="choice-label">Save {d.chosen_n} {esc(d.chosen_group)}</span>
      <span style="color:#888"> over </span>
      <span class="choice-label">{d.rejected_n} {esc(d.rejected_group)}</span>
      <span class="ndiff-badge" style="background:{color}">N-diff {d.n_diff}</span>
    </div>
    <div style="text-align:right;font-size:0.9em;color:#555">#{idx + 1}</div>
  </div>
  <div class="card-meta">
    <strong>Model:</strong> {esc(d.model)} &nbsp;|&nbsp;
    <strong>Factor:</strong> {esc(d.factor)} &nbsp;|&nbsp;
    <strong>Prob:</strong>
    <div class="prob-bar"><div class="prob-fill" style="width:{prob_pct:.0f}%"></div></div>
    {d.probability:.1%} &nbsp;|&nbsp;
    <strong>Responses:</strong> {d.count_chosen}/{d.count_chosen + d.count_rejected}
  </div>
  {traces_html}
</div>
"""

    # Filter controls
    model_cbs = "\n".join(
        f'<label><input type="checkbox" class="model-filter" value="{esc(m)}" checked> {esc(m)}</label>'
        for m in all_models
    )
    factor_cbs = "\n".join(
        f'<label><input type="checkbox" class="factor-filter" value="{esc(f)}" checked> {esc(f)}</label>'
        for f in all_factors
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Extreme Baseline Decisions — Reasoning Models</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; color: #222; }}
  h1 {{ border-bottom: 3px solid #1565c0; padding-bottom: 8px; margin-bottom: 4px; }}
  h1 small {{ font-size: 0.5em; color: #888; font-weight: normal; }}
  h3 {{ color: #1565c0; margin: 18px 0 8px; }}

  /* Tabs */
  .tab-bar {{ display: flex; gap: 0; border-bottom: 2px solid #1565c0; margin: 16px 0 0; }}
  .tab-btn {{ padding: 10px 24px; cursor: pointer; border: none; background: #e3f2fd;
              font-size: 1em; font-weight: 600; color: #1565c0; border-radius: 6px 6px 0 0; }}
  .tab-btn.active {{ background: #1565c0; color: #fff; }}
  .tab-content {{ display: none; padding: 16px 0; }}
  .tab-content.active {{ display: block; }}

  /* Filters */
  .filters {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
              padding: 14px 18px; margin: 16px 0; }}
  .filter-group {{ margin: 8px 0; display: flex; flex-wrap: wrap; gap: 4px 14px; align-items: center; }}
  .filter-group label {{ font-size: 0.9em; cursor: pointer; }}
  .filter-group strong {{ min-width: 80px; }}
  .slider-row {{ display: flex; align-items: center; gap: 8px; }}
  #visible-count {{ font-weight: bold; color: #1565c0; }}

  /* Stat boxes */
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }}
  .stat-box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
               padding: 10px 18px; text-align: center; min-width: 100px; }}
  .stat-box .num {{ font-size: 1.8em; font-weight: bold; color: #1565c0; }}
  .stat-box .lbl {{ font-size: 0.8em; color: #666; }}

  /* Tables */
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.9em; }}
  th, td {{ padding: 5px 10px; border: 1px solid #ddd; text-align: left; }}
  th {{ background: #e3f2fd; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  /* Cards */
  .card {{ border: 1px solid #ddd; border-left: 5px solid #888; border-radius: 6px;
           padding: 14px 18px; margin: 10px 0; background: #fff; }}
  .choice-label {{ font-size: 1.15em; font-weight: bold; }}
  .ndiff-badge {{ color: #fff; border-radius: 10px; padding: 2px 8px; font-size: 0.8em; margin-left: 8px; }}
  .card-meta {{ margin-top: 6px; font-size: 0.9em; color: #555; display: flex;
                flex-wrap: wrap; align-items: center; gap: 2px; }}
  .prob-bar {{ background: #e0e0e0; border-radius: 4px; height: 16px; width: 100px;
               display: inline-block; vertical-align: middle; margin: 0 4px; }}
  .prob-fill {{ background: #1976d2; border-radius: 4px; height: 16px; }}
</style>
</head>
<body>

<h1>Extreme Baseline Decisions <small>reasoning models only</small></h1>
<p>Cases where reasoning models save <strong>fewer</strong> people of one group over
<strong>more</strong> of another, with <strong>no contextual influence</strong>.
Threshold: N-diff &ge; {n_diff_threshold}.</p>

<!-- Filters (shared across tabs) -->
<div class="filters">
  <div class="filter-group">
    <strong>Models:</strong>
    {model_cbs}
  </div>
  <div class="filter-group">
    <strong>Factors:</strong>
    {factor_cbs}
  </div>
  <div class="filter-group">
    <strong>Min N-diff:</strong>
    <div class="slider-row">
      <input type="range" id="ndiff-slider" min="{n_diff_threshold}" max="{max_ndiff}"
             value="{n_diff_threshold}" oninput="document.getElementById('ndiff-val').textContent=this.value;applyFilters()">
      <span id="ndiff-val">{n_diff_threshold}</span>
    </div>
  </div>
  <div style="margin-top:6px">
    Showing <span id="visible-count">{len(decisions)}</span> of {len(decisions)} decisions
  </div>
</div>

<!-- Tab bar -->
<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('stats')">Stats</button>
  <button class="tab-btn" onclick="switchTab('decisions')">Decisions</button>
</div>

<!-- Stats tab -->
<div id="tab-stats" class="tab-content active">
  <div class="stat-row" id="top-stats"></div>

  <h3>By Model</h3>
  <table id="model-table"><thead><tr>
    <th>Model</th><th>Count</th><th>Max N-diff</th><th>Avg N-diff</th><th>Avg Probability</th>
  </tr></thead><tbody></tbody></table>

  <h3>By Factor</h3>
  <table id="factor-table"><thead><tr>
    <th>Factor</th><th>Count</th><th>Max N-diff</th><th>Preferred Group</th>
  </tr></thead><tbody></tbody></table>

  <h3>By N-diff</h3>
  <table id="ndiff-table"><thead><tr>
    <th>N-diff</th><th>Count</th><th>Avg Probability</th><th>Models</th>
  </tr></thead><tbody></tbody></table>

  <h3>By Model &times; Factor</h3>
  <table id="cross-table"><thead><tr>
    <th>Model</th><th>Factor</th><th>Count</th><th>Max N-diff</th><th>Preferred Group</th>
  </tr></thead><tbody></tbody></table>
</div>

<!-- Decisions tab -->
<div id="tab-decisions" class="tab-content">
{cards_html}
</div>

<script>
// All decision data for dynamic stats
const ALL_DATA = {decisions_json};

function getVisibleIndices() {{
  const models = new Set([...document.querySelectorAll('.model-filter:checked')].map(c => c.value));
  const factors = new Set([...document.querySelectorAll('.factor-filter:checked')].map(c => c.value));
  const minDiff = parseInt(document.getElementById('ndiff-slider').value);
  const indices = [];
  ALL_DATA.forEach((d, i) => {{
    if (models.has(d.model) && factors.has(d.factor) && d.n_diff >= minDiff) indices.push(i);
  }});
  return indices;
}}

function applyFilters() {{
  const vis = getVisibleIndices();
  const visSet = new Set(vis);

  // Update card visibility
  document.querySelectorAll('.card').forEach(card => {{
    const idx = parseInt(card.dataset.idx);
    card.style.display = visSet.has(idx) ? '' : 'none';
  }});
  document.getElementById('visible-count').textContent = vis.length;

  // Update stats
  updateStats(vis);
}}

function updateStats(indices) {{
  const data = indices.map(i => ALL_DATA[i]);
  const n = data.length;

  // Top stat boxes
  const maxDiff = n ? Math.max(...data.map(d => d.n_diff)) : 0;
  const avgProb = n ? (data.reduce((s, d) => s + d.probability, 0) / n) : 0;
  const avgDiff = n ? (data.reduce((s, d) => s + d.n_diff, 0) / n) : 0;
  const models = new Set(data.map(d => d.model));
  const factors = new Set(data.map(d => d.factor));
  document.getElementById('top-stats').innerHTML =
    statBox(n, 'Decisions') +
    statBox(models.size, 'Models') +
    statBox(factors.size, 'Factors') +
    statBox(maxDiff, 'Max N-diff') +
    statBox(avgDiff.toFixed(1), 'Avg N-diff') +
    statBox((avgProb * 100).toFixed(1) + '%', 'Avg Prob');

  // By model
  const byModel = groupBy(data, 'model');
  let mRows = '';
  for (const [model, ds] of sortedEntries(byModel)) {{
    const mx = Math.max(...ds.map(d => d.n_diff));
    const av = (ds.reduce((s, d) => s + d.n_diff, 0) / ds.length).toFixed(1);
    const ap = (ds.reduce((s, d) => s + d.probability, 0) / ds.length * 100).toFixed(1) + '%';
    mRows += `<tr><td>${{model}}</td><td class="num">${{ds.length}}</td><td class="num">${{mx}}</td><td class="num">${{av}}</td><td class="num">${{ap}}</td></tr>`;
  }}
  document.querySelector('#model-table tbody').innerHTML = mRows;

  // By factor
  const byFactor = groupBy(data, 'factor');
  let fRows = '';
  for (const [factor, ds] of sortedEntries(byFactor)) {{
    const mx = Math.max(...ds.map(d => d.n_diff));
    const grpCounts = {{}};
    ds.forEach(d => {{ grpCounts[d.chosen_group] = (grpCounts[d.chosen_group] || 0) + 1; }});
    const grpStr = Object.entries(grpCounts).sort((a,b) => b[1]-a[1]).map(([g,c]) => `${{g}}: ${{c}}`).join(', ');
    fRows += `<tr><td>${{factor}}</td><td class="num">${{ds.length}}</td><td class="num">${{mx}}</td><td>${{grpStr}}</td></tr>`;
  }}
  document.querySelector('#factor-table tbody').innerHTML = fRows;

  // By N-diff
  const byNdiff = groupBy(data, 'n_diff');
  let nRows = '';
  for (const nd of Object.keys(byNdiff).map(Number).sort((a,b) => b-a)) {{
    const ds = byNdiff[nd];
    const ap = (ds.reduce((s, d) => s + d.probability, 0) / ds.length * 100).toFixed(1) + '%';
    const mods = [...new Set(ds.map(d => d.model))].join(', ');
    nRows += `<tr><td class="num">${{nd}}</td><td class="num">${{ds.length}}</td><td class="num">${{ap}}</td><td>${{mods}}</td></tr>`;
  }}
  document.querySelector('#ndiff-table tbody').innerHTML = nRows;

  // By Model x Factor
  const byMF = {{}};
  data.forEach(d => {{
    const k = d.model + '|||' + d.factor;
    if (!byMF[k]) byMF[k] = [];
    byMF[k].push(d);
  }});
  let cRows = '';
  for (const k of Object.keys(byMF).sort()) {{
    const [model, factor] = k.split('|||');
    const ds = byMF[k];
    const mx = Math.max(...ds.map(d => d.n_diff));
    const grpCounts = {{}};
    ds.forEach(d => {{ grpCounts[d.chosen_group] = (grpCounts[d.chosen_group] || 0) + 1; }});
    const grpStr = Object.entries(grpCounts).sort((a,b) => b[1]-a[1]).map(([g,c]) => `${{g}}: ${{c}}`).join(', ');
    cRows += `<tr><td>${{model}}</td><td>${{factor}}</td><td class="num">${{ds.length}}</td><td class="num">${{mx}}</td><td>${{grpStr}}</td></tr>`;
  }}
  document.querySelector('#cross-table tbody').innerHTML = cRows;
}}

function statBox(val, label) {{
  return `<div class="stat-box"><div class="num">${{val}}</div><div class="lbl">${{label}}</div></div>`;
}}

function groupBy(arr, key) {{
  const m = {{}};
  arr.forEach(d => {{ const k = d[key]; if (!m[k]) m[k] = []; m[k].push(d); }});
  return m;
}}

function sortedEntries(obj) {{
  return Object.entries(obj).sort((a,b) => b[1].length - a[1].length);
}}

function switchTab(name) {{
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

// Wire up filter listeners
document.querySelectorAll('.model-filter, .factor-filter').forEach(
  cb => cb.addEventListener('change', applyFilters)
);

// Initial stats render
applyFilters();
</script>

</body>
</html>"""


# ============================================================================
# Chart: rate of choosing smaller group at each N-diff, per model (ALL models)
# ============================================================================


def compute_smaller_group_rates(
    results_dirs: list[str],
    factor_filter: Optional[list[str]] = None,
) -> dict[str, dict[str, dict[int, dict[str, int]]]]:
    """Compute rate of choosing the smaller group for each (model, factor, n_diff).

    Includes ALL models (reasoning and non-reasoning) for comparison.

    Returns:
        {model_display: {factor: {n_diff: {"chose_smaller": int, "total": int}}}}
    """
    experiments = discover_experiments(results_dirs, factor_filter=factor_filter)

    # {model_display: {factor: {n_diff: {chose_smaller, total}}}}
    stats: dict[str, dict[str, dict[int, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"chose_smaller": 0, "total": 0})
        )
    )
    # Track seen baselines to avoid double-counting across nudge dirs
    seen = set()

    for results_dir, factor, model, nudge_type in experiments:
        condition_dirs_list = find_condition_directories(
            factor, model, nudge_type, results_dir
        )

        for condition_dirs in condition_dirs_list:
            base_path = condition_dirs.get("base")
            if not base_path:
                continue

            graph_data = load_preference_graph(base_path)
            if not graph_data:
                continue

            cfg = graph_data.get("simple_experiment_config", {})
            reasoning_mode = cfg.get("reasoning_mode", "none")
            model_display = model
            if reasoning_mode not in ("none", "N/A", ""):
                model_display = f"{model} ({reasoning_mode})"

            factor_name = get_factor_name_from_graph(graph_data)
            if not factor_name:
                continue

            options = graph_data.get("options", [])
            edges = graph_data.get("edges", {})
            options_by_id = {opt["id"]: opt for opt in options}

            for edge_key, edge_data in edges.items():
                try:
                    ids = eval(edge_key)
                    opt_a = options_by_id.get(ids[0])
                    opt_b = options_by_id.get(ids[1])
                    if not opt_a or not opt_b:
                        continue

                    n_a = opt_a.get("N", 0)
                    n_b = opt_b.get("N", 0)
                    group_a = opt_a.get(factor_name, "")
                    group_b = opt_b.get(factor_name, "")

                    if group_a == group_b or n_a == n_b:
                        continue

                    n_diff = abs(n_a - n_b)

                    dedup_key = (
                        model_display,
                        factor,
                        min(n_a, n_b),
                        max(n_a, n_b),
                        group_a,
                        group_b,
                    )
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    aux_data = edge_data.get("aux_data", {})
                    total = aux_data.get("total_responses", 0)
                    if total == 0:
                        continue

                    stats[model_display][factor][n_diff]["total"] += total

                    # Count responses that chose the smaller group
                    if n_a < n_b:
                        chose_smaller = int(aux_data.get("count_A", 0))
                    else:
                        chose_smaller = int(aux_data.get("count_B", 0))

                    stats[model_display][factor][n_diff]["chose_smaller"] += (
                        chose_smaller
                    )

                except Exception:
                    continue

    return stats


def generate_chart(
    results_dirs: list[str],
    output_path: str,
    factor_filter: Optional[list[str]] = None,
) -> None:
    """Generate a multi-panel chart: one subplot per factor, showing
    % choosing smaller group by N-diff, per model."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    stats = compute_smaller_group_rates(results_dirs, factor_filter=factor_filter)

    if not stats:
        print("No data for chart.", file=sys.stderr)
        return

    # Collect all factors and models
    all_factors = sorted({f for model_stats in stats.values() for f in model_stats})
    models = sorted(stats.keys())

    if not all_factors:
        print("No factors found for chart.", file=sys.stderr)
        return

    # Collect all n_diffs across all factors
    all_ndiffs = sorted(
        {
            nd
            for model_stats in stats.values()
            for factor_stats in model_stats.values()
            for nd in factor_stats
            if nd >= 1
        }
    )

    n_factors = len(all_factors)
    ncols = min(3, n_factors)
    nrows = math.ceil(n_factors / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7, len(models) * 1.1) * ncols, 5.5 * nrows),
        squeeze=False,
    )

    n_models = len(models)
    n_ndiffs = len(all_ndiffs)
    bar_width = 0.8 / max(n_ndiffs, 1)
    x = np.arange(n_models)

    cmap = plt.cm.get_cmap("RdYlGn_r", max(n_ndiffs, 2))

    for idx, factor in enumerate(all_factors):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for i, nd in enumerate(all_ndiffs):
            rates = []
            for model in models:
                s = (
                    stats.get(model, {})
                    .get(factor, {})
                    .get(nd, {"chose_smaller": 0, "total": 0})
                )
                rate = s["chose_smaller"] / s["total"] * 100 if s["total"] > 0 else 0
                rates.append(rate)
            offset = (i - n_ndiffs / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                rates,
                bar_width * 0.9,
                label=f"N-diff={nd}" if idx == 0 else "",
                color=cmap(i / max(n_ndiffs - 1, 1)),
                edgecolor="white",
                linewidth=0.3,
            )

        ax.set_title(factor.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_ylabel("% choosing smaller group", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)

    # Hide unused subplots
    for idx in range(n_factors, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend from first subplot's bars
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(i / max(n_ndiffs - 1, 1)))
        for i in range(n_ndiffs)
    ]
    labels = [f"N-diff={nd}" for nd in all_ndiffs]
    fig.legend(
        handles,
        labels,
        title="N-diff",
        loc="upper right",
        fontsize=8,
        title_fontsize=9,
        bbox_to_anchor=(0.99, 0.99),
    )

    fig.suptitle(
        "Rate of Choosing Fewer People by Factor (Baseline, No Nudge)",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {output_path}", file=sys.stderr)


def _severity_color(n_diff: int) -> str:
    if n_diff >= 7:
        return "#d32f2f"
    if n_diff >= 5:
        return "#e65100"
    if n_diff >= 3:
        return "#f57f17"
    return "#558b2f"


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze extreme baseline decisions in moral triage experiments (reasoning models only)."
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results_main0", "results_main1"],
        help="Results directories to scan (default: results_main0 results_main1)",
    )
    parser.add_argument(
        "--n-diff-threshold",
        type=int,
        default=1,
        help="Minimum N-difference to include (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for text report (also prints to stdout)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output file path for structured JSON data",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default=None,
        help="Output file path for interactive HTML report",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter by model names (only reasoning models are included regardless)",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Filter by factor names (e.g., age_group wealth gender)",
    )
    parser.add_argument(
        "--chart-output",
        type=str,
        default=None,
        help="Output path for PNG chart showing %% choosing smaller group by N-diff per model (includes ALL models)",
    )
    parser.add_argument(
        "--traces-only",
        action="store_true",
        help="Only show decisions that have reasoning traces",
    )

    args = parser.parse_args()

    print(
        f"Scanning {args.results_dirs} for extreme baseline decisions "
        f"(N-diff >= {args.n_diff_threshold}, reasoning models only)...",
        file=sys.stderr,
    )

    decisions = extract_extreme_baseline_decisions(
        results_dirs=args.results_dirs,
        n_diff_threshold=args.n_diff_threshold,
        model_filter=args.models,
        factor_filter=args.factors,
    )

    if args.traces_only:
        decisions = [d for d in decisions if d.reasoning_traces]

    print(f"Found {len(decisions)} extreme baseline decisions.", file=sys.stderr)

    if not decisions:
        print("No extreme baseline decisions found.", file=sys.stderr)
        return

    report = format_full_report(decisions, args.n_diff_threshold)

    # Print to stdout
    print(report)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to {args.output}", file=sys.stderr)

    # Save JSON if requested
    if args.json_output:
        json_data = [asdict(d) for d in decisions]
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON data saved to {args.json_output}", file=sys.stderr)

    # Save HTML if requested
    if args.html_output:
        html_report = format_html_report(decisions, args.n_diff_threshold)
        with open(args.html_output, "w") as f:
            f.write(html_report)
        print(f"HTML report saved to {args.html_output}", file=sys.stderr)

    # Generate chart if requested (includes ALL models, not just reasoning)
    if args.chart_output:
        generate_chart(
            results_dirs=args.results_dirs,
            output_path=args.chart_output,
            factor_filter=args.factors,
        )


if __name__ == "__main__":
    main()
