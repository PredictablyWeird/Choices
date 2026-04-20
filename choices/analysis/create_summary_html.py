#!/usr/bin/env python3
"""
Generate an interactive HTML viewer for nudge experiment results.

Reuses the data-loading and computation logic from create_summary.py,
then serializes results as JSON embedded in a self-contained HTML file
with client-side filtering, aggregation, and sorting.

Usage:
    uv run python -m choices.analysis.create_summary_html
    uv run python -m choices.analysis.create_summary_html --results-dirs results results_anthropic
    uv run python -m choices.analysis.create_summary_html -o summary.html
"""

import argparse
import dataclasses
import json
from pathlib import Path
from typing import List

from choices.analysis.create_summary import (
    FrequencyResult,
    compute_all_results,
)
from choices.analysis.utils import get_base_model_name, get_model_display_name


def results_to_json(results: List[FrequencyResult]) -> List[dict]:
    """Convert FrequencyResult objects to JSON-serializable dicts with display metadata."""
    rows = []
    for r in results:
        d = dataclasses.asdict(r)
        d["model_display"] = get_model_display_name(r.model)
        d["base_model"] = get_base_model_name(r.model)
        d["factor_label"] = f"{r.factor} ({r.level_A}/{r.level_B})"
        rows.append(d)
    return rows


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nudge Experiment Results</title>
<style>
:root {
  --bg: #f8f9fa;
  --surface: #fff;
  --border: #dee2e6;
  --text: #212529;
  --text-muted: #6c757d;
  --accent: #4361ee;
  --accent-light: #eef1ff;
  --danger: #dc3545;
  --success: #198754;
  --warning: #fd7e14;
  --sig: #198754;
  --backfire: #dc3545;
  --radius: 6px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); font-size: 13px; line-height: 1.5; }
.app { max-width: 100%; padding: 16px 24px; }
h1 { font-size: 20px; font-weight: 600; margin-bottom: 4px; }
.subtitle { color: var(--text-muted); font-size: 13px; margin-bottom: 16px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; align-items: flex-end; }
.control-group { display: flex; flex-direction: column; gap: 2px; }
.control-group label { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); }
.control-group select, .control-group input { font-size: 13px; padding: 5px 8px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); min-width: 140px; }
.control-group select[multiple] { min-height: 68px; }
.control-group select:focus, .control-group input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-light); }
.btn { font-size: 12px; padding: 6px 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); cursor: pointer; font-weight: 500; }
.btn:hover { background: var(--accent-light); border-color: var(--accent); }
.btn.active { background: var(--accent); color: #fff; border-color: var(--accent); }
.tag-bar { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
.tag { display: inline-flex; align-items: center; gap: 3px; font-size: 11px; padding: 2px 8px; background: var(--accent-light); color: var(--accent); border-radius: 12px; }
.tag .remove { cursor: pointer; font-weight: bold; opacity: 0.6; }
.tag .remove:hover { opacity: 1; }
.stats-bar { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; padding: 10px 14px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }
.stat { display: flex; flex-direction: column; }
.stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); }
.stat-value { font-size: 15px; font-weight: 600; }
.table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); box-shadow: var(--shadow); }
table { width: 100%; border-collapse: collapse; white-space: nowrap; }
thead th { position: sticky; top: 0; background: var(--surface); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 10px; border-bottom: 2px solid var(--border); cursor: pointer; user-select: none; text-align: left; }
thead th:hover { background: var(--accent-light); }
thead th .sort-arrow { margin-left: 3px; font-size: 9px; opacity: 0.4; }
thead th.sorted .sort-arrow { opacity: 1; color: var(--accent); }
tbody td { padding: 6px 10px; border-bottom: 1px solid #f0f0f0; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: #f8f9ff; }
.sig-marker { color: var(--sig); font-weight: 700; }
.backfire-cell { color: var(--backfire); font-weight: 600; }
.none-cell { color: var(--text-muted); }
.agg-header-row td { background: var(--accent-light); font-weight: 600; border-bottom: 2px solid var(--accent); padding: 8px 10px; }
.section { margin-bottom: 20px; }
.section-title { font-size: 14px; font-weight: 600; margin-bottom: 6px; }
.columns-toggle { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }
.col-btn { font-size: 11px; padding: 3px 8px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--surface); cursor: pointer; }
.col-btn.on { background: var(--accent); color: #fff; border-color: var(--accent); }
.col-btn:hover { border-color: var(--accent); }
.help-text { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
</style>
</head>
<body>
<div class="app">
<h1>Nudge Experiment Results</h1>
<p class="subtitle">Interactive viewer &mdash; filter, aggregate, and sort experiment data</p>

<div class="controls" id="controls">
  <div class="control-group">
    <label>Model</label>
    <select id="filter-model" multiple></select>
  </div>
  <div class="control-group">
    <label>Factor</label>
    <select id="filter-factor" multiple></select>
  </div>
  <div class="control-group">
    <label>Nudge Type</label>
    <select id="filter-nudge" multiple></select>
  </div>
  <div class="control-group">
    <label>Reasoning</label>
    <select id="filter-reasoning" multiple></select>
  </div>
  <div class="control-group">
    <label>Baseline Sig</label>
    <select id="filter-baseline-sig">
      <option value="any">Any</option>
      <option value="sig">Significant</option>
      <option value="not-sig">Not Significant</option>
    </select>
  </div>
  <div class="control-group">
    <label>Aggregate Over</label>
    <select id="aggregate">
      <option value="none">None (show all rows)</option>
      <option value="model">Model + Reasoning</option>
      <option value="factor">Factor</option>
      <option value="nudge_type">Nudge Type</option>
      <option value="reasoning">Reasoning Condition</option>
      <option value="model-factor">Model + Factor</option>
      <option value="model-nudge">Model + Nudge Type</option>
      <option value="factor-nudge">Factor + Nudge Type</option>
    </select>
  </div>
  <div class="control-group">
    <label>Decimals</label>
    <input id="decimals" type="number" value="3" min="1" max="6" style="width:60px">
  </div>
</div>

<div class="section">
  <div class="section-title">Visible Columns</div>
  <div class="columns-toggle" id="col-toggles"></div>
</div>

<div class="tag-bar" id="active-filters"></div>

<div class="stats-bar" id="stats-bar"></div>

<div class="table-wrap">
  <table id="results-table">
    <thead><tr id="table-head"></tr></thead>
    <tbody id="table-body"></tbody>
  </table>
</div>
</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const COLUMNS = [
  {key: "model_display", label: "Model", type: "text", defaultOn: true},
  {key: "reasoning_condition", label: "Reasoning", type: "text", defaultOn: true},
  {key: "factor_label", label: "Factor", type: "text", defaultOn: true},
  {key: "nudge_type", label: "Nudge Type", type: "text", defaultOn: true},
  {key: "invalid_pct", label: "Invalid%", type: "pct", defaultOn: true},
  {key: "f_0_B", label: "f\u2080(B)", type: "freq", defaultOn: true},
  {key: "f_A_B", label: "f\u2090(B)", type: "freq_sig", sigKey: "sig_A", defaultOn: true},
  {key: "f_B_B", label: "f\u1D47(B)", type: "freq_sig", sigKey: "sig_B", defaultOn: true},
  {key: "avg_f_B", label: "Avg f(B)", type: "freq", defaultOn: false},
  {key: "abs_effect", label: "|Effect|", type: "freq", defaultOn: true},
  {key: "steerability_A", label: "Steer(A)", type: "signed", defaultOn: true},
  {key: "steerability_B", label: "Steer(B)", type: "signed", defaultOn: true},
  {key: "avg_steerability", label: "Avg Steer", type: "signed", defaultOn: false},
  {key: "abs_steerability", label: "|Steer|", type: "freq", defaultOn: true},
  {key: "steerability_asym", label: "Asym", type: "signed_sig", sigKey: "sig_asym", defaultOn: true},
  {key: "normalized_steerability_asym", label: "N-Asym", type: "signed", defaultOn: true},
  {key: "larger_group_rate", label: "P(Large)", type: "freq", defaultOn: false},
  {key: "larger_group_rate_base", label: "P\u2080(LG)", type: "freq", defaultOn: false},
  {key: "larger_group_rate_A", label: "P\u2090(LG)", type: "freq", defaultOn: false},
  {key: "larger_group_rate_B", label: "P\u1D47(LG)", type: "freq", defaultOn: false},
  {key: "backfire_A", label: "Backfire A", type: "bool", defaultOn: false},
  {key: "backfire_B", label: "Backfire B", type: "bool", defaultOn: false},
  {key: "n_comparisons", label: "N Pairs", type: "int", defaultOn: false},
];

const NUMERIC_KEYS = COLUMNS.filter(c => c.type !== "text" && c.type !== "bool").map(c => c.key);
const AGG_TEXT_KEYS = ["model_display", "reasoning_condition", "factor_label", "nudge_type"];

let visibleCols = new Set(COLUMNS.filter(c => c.defaultOn).map(c => c.key));
let sortKey = null;
let sortAsc = true;

function unique(arr) { return [...new Set(arr)].sort(); }

function populateFilter(id, values) {
  const sel = document.getElementById(id);
  sel.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v; opt.textContent = v;
    sel.appendChild(opt);
  }
}

function getSelectedValues(id) {
  const sel = document.getElementById(id);
  return [...sel.selectedOptions].map(o => o.value);
}

function getFiltered() {
  const models = getSelectedValues("filter-model");
  const factors = getSelectedValues("filter-factor");
  const nudges = getSelectedValues("filter-nudge");
  const reasoning = getSelectedValues("filter-reasoning");
  const baselineSig = document.getElementById("filter-baseline-sig").value;

  let rows = DATA;
  if (models.length) rows = rows.filter(r => models.includes(r.model_display));
  if (factors.length) rows = rows.filter(r => factors.includes(r.factor));
  if (nudges.length) rows = rows.filter(r => nudges.includes(r.nudge_type));
  if (reasoning.length) rows = rows.filter(r => reasoning.includes(r.reasoning_condition));
  if (baselineSig === "sig") rows = rows.filter(r => r.sig_baseline_B);
  else if (baselineSig === "not-sig") rows = rows.filter(r => !r.sig_baseline_B);
  return rows;
}

function mean(arr) {
  if (!arr.length) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function meanCI(arr) {
  const n = arr.length;
  if (n === 0) return {mean: null, ci_low: null, ci_high: null};
  const m = arr.reduce((a, b) => a + b, 0) / n;
  if (n === 1) return {mean: m, ci_low: m, ci_high: m};
  const variance = arr.reduce((s, x) => s + (x - m) ** 2, 0) / (n - 1);
  const se = Math.sqrt(variance / n);
  return {mean: m, ci_low: m - 1.96 * se, ci_high: m + 1.96 * se};
}

function aggregateRows(rows, groupKeyFn) {
  const groups = {};
  for (const r of rows) {
    const k = groupKeyFn(r);
    if (!groups[k]) groups[k] = {key: k, rows: [], rep: r};
    groups[k].rows.push(r);
  }
  const out = [];
  for (const g of Object.values(groups)) {
    const agg = {};
    for (const col of COLUMNS) {
      if (col.type === "text") {
        const vals = unique(g.rows.map(r => r[col.key]));
        agg[col.key] = vals.length === 1 ? vals[0] : vals.join(", ");
      } else if (col.type === "bool") {
        const trueCount = g.rows.filter(r => r[col.key]).length;
        agg[col.key] = `${trueCount}/${g.rows.length}`;
      } else if (col.type === "int") {
        agg[col.key] = g.rows.reduce((s, r) => s + (r[col.key] || 0), 0);
      } else {
        const vals = g.rows.map(r => r[col.key]).filter(v => v !== null && v !== undefined);
        agg[col.key] = vals.length ? mean(vals) : null;
      }
    }
    agg._n = g.rows.length;
    agg._isAgg = true;

    // Compute significance and backfire rates for aggregated rows
    const totalNudges = 2 * g.rows.length;
    const sigCount = g.rows.reduce((s, r) => s + (r.sig_A ? 1 : 0) + (r.sig_B ? 1 : 0), 0);
    agg._sig_rate = totalNudges > 0 ? sigCount / totalNudges : 0;
    const sigAsymCount = g.rows.reduce((s, r) => s + (r.sig_asym ? 1 : 0), 0);
    agg._sig_asym_rate = g.rows.length > 0 ? sigAsymCount / g.rows.length : 0;
    const backfireCount = g.rows.reduce((s, r) => s + (r.backfire_A ? 1 : 0) + (r.backfire_B ? 1 : 0), 0);
    agg._backfire_rate = totalNudges > 0 ? backfireCount / totalNudges : 0;

    out.push(agg);
  }
  return out;
}

function getGroupFn() {
  const mode = document.getElementById("aggregate").value;
  if (mode === "none") return null;
  const fns = {
    "model": r => `${r.model_display}||${r.reasoning_condition}`,
    "factor": r => r.factor,
    "nudge_type": r => r.nudge_type,
    "reasoning": r => r.reasoning_condition,
    "model-factor": r => `${r.model_display}||${r.reasoning_condition}||${r.factor}`,
    "model-nudge": r => `${r.model_display}||${r.reasoning_condition}||${r.nudge_type}`,
    "factor-nudge": r => `${r.factor}||${r.nudge_type}`,
  };
  return fns[mode] || null;
}

function sortRows(rows) {
  if (!sortKey) return rows;
  return [...rows].sort((a, b) => {
    let va = a[sortKey], vb = b[sortKey];
    if (va === null || va === undefined) return 1;
    if (vb === null || vb === undefined) return -1;
    if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  });
}

function fmt(val, type, row, col) {
  const dec = parseInt(document.getElementById("decimals").value) || 3;
  if (val === null || val === undefined) return '<span class="none-cell">N/A</span>';

  if (type === "text") return escHtml(String(val));
  if (type === "bool") {
    if (typeof val === "string") return val;
    return val ? '<span class="backfire-cell">Yes</span>' : '<span class="none-cell">No</span>';
  }
  if (type === "int") return String(val);
  if (type === "pct") return val.toFixed(1) + "%";

  if (type === "freq") return val.toFixed(dec);
  if (type === "signed") return (val >= 0 ? "+" : "") + val.toFixed(dec);

  if (type === "freq_sig") {
    const sig = col && col.sigKey && row[col.sigKey];
    const star = sig ? '<span class="sig-marker">*</span>' : "";
    return val.toFixed(dec) + star;
  }
  if (type === "signed_sig") {
    const sig = col && col.sigKey && row[col.sigKey];
    const star = sig ? '<span class="sig-marker">*</span>' : "";
    return (val >= 0 ? "+" : "") + val.toFixed(dec) + star;
  }
  return String(val);
}

function escHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

function renderStats(rows) {
  const bar = document.getElementById("stats-bar");
  const dec = parseInt(document.getElementById("decimals").value) || 3;
  const n = rows.length;
  const totalNudges = 2 * n;
  const sigCount = rows.reduce((s, r) => s + (r.sig_A ? 1 : 0) + (r.sig_B ? 1 : 0), 0);
  const sigRate = totalNudges > 0 ? sigCount / totalNudges : 0;
  const sigAsymCount = rows.reduce((s, r) => s + (r.sig_asym ? 1 : 0), 0);
  const sigAsymRate = n > 0 ? sigAsymCount / n : 0;
  const effectVals = rows.map(r => r.abs_effect);
  const absSteerVals = rows.map(r => r.abs_steerability).filter(v => v != null);
  const avgEffect = meanCI(effectVals);
  const avgAbsSteer = meanCI(absSteerVals);

  const stats = [
    {label: "Experiments", value: n},
    {label: "Avg |Effect|", value: avgEffect.mean !== null ? avgEffect.mean.toFixed(dec) : "N/A"},
    {label: "Avg |Steer|", value: avgAbsSteer.mean !== null ? avgAbsSteer.mean.toFixed(dec) : "N/A"},
    {label: "Sig Rate", value: (sigRate * 100).toFixed(1) + "%"},
    {label: "Sig Asym Rate", value: (sigAsymRate * 100).toFixed(1) + "%"},
  ];
  bar.innerHTML = stats.map(s => `<div class="stat"><span class="stat-label">${s.label}</span><span class="stat-value">${s.value}</span></div>`).join("");
}

function renderFilterTags() {
  const container = document.getElementById("active-filters");
  const tags = [];
  const filters = [
    {id: "filter-model", label: "Model"},
    {id: "filter-factor", label: "Factor"},
    {id: "filter-nudge", label: "Nudge"},
    {id: "filter-reasoning", label: "Reasoning"},
  ];
  for (const f of filters) {
    for (const v of getSelectedValues(f.id)) {
      tags.push(`<span class="tag">${f.label}: ${escHtml(v)} <span class="remove" data-filter="${f.id}" data-value="${escHtml(v)}">&times;</span></span>`);
    }
  }
  const bsig = document.getElementById("filter-baseline-sig").value;
  if (bsig !== "any") {
    tags.push(`<span class="tag">Baseline: ${bsig} <span class="remove" data-filter="filter-baseline-sig" data-value="any">&times;</span></span>`);
  }
  container.innerHTML = tags.join("");
  container.querySelectorAll(".remove").forEach(el => {
    el.addEventListener("click", () => {
      const fid = el.dataset.filter;
      const val = el.dataset.value;
      if (fid === "filter-baseline-sig") {
        document.getElementById(fid).value = "any";
      } else {
        const sel = document.getElementById(fid);
        for (const opt of sel.options) {
          if (opt.value === val) opt.selected = false;
        }
      }
      render();
    });
  });
}

function renderColToggles() {
  const container = document.getElementById("col-toggles");
  container.innerHTML = COLUMNS.map(c => {
    const on = visibleCols.has(c.key);
    return `<button class="col-btn ${on ? 'on' : ''}" data-key="${c.key}">${c.label}</button>`;
  }).join("");
  container.querySelectorAll(".col-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const k = btn.dataset.key;
      if (visibleCols.has(k)) visibleCols.delete(k); else visibleCols.add(k);
      render();
    });
  });
}

function renderTable(displayRows, isAgg) {
  const thead = document.getElementById("table-head");
  const tbody = document.getElementById("table-body");
  const visCols = COLUMNS.filter(c => visibleCols.has(c.key));

  if (isAgg) {
    const extraCols = [
      {key: "_n", label: "N", type: "int"},
      {key: "_sig_rate", label: "Sig Rate", type: "pct_rate"},
      {key: "_sig_asym_rate", label: "Sig Asym", type: "pct_rate"},
      {key: "_backfire_rate", label: "Backfire Rate", type: "pct_rate"},
    ];
    var allCols = [...visCols, ...extraCols];
  } else {
    var allCols = visCols;
  }

  thead.innerHTML = allCols.map(c => {
    const isSorted = sortKey === c.key;
    const arrow = isSorted ? (sortAsc ? "\u25B2" : "\u25BC") : "\u25B4";
    return `<th class="${isSorted ? 'sorted' : ''}" data-key="${c.key}">${c.label} <span class="sort-arrow">${arrow}</span></th>`;
  }).join("");

  thead.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => {
      const k = th.dataset.key;
      if (sortKey === k) sortAsc = !sortAsc;
      else { sortKey = k; sortAsc = true; }
      render();
    });
  });

  const rows = sortRows(displayRows);
  tbody.innerHTML = rows.map(r => {
    const cells = allCols.map(c => {
      if (c.type === "pct_rate") {
        const v = r[c.key];
        return `<td>${v !== null && v !== undefined ? (v * 100).toFixed(1) + "%" : "N/A"}</td>`;
      }
      return `<td>${fmt(r[c.key], c.type, r, c)}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
}

function render() {
  const filtered = getFiltered();
  const groupFn = getGroupFn();

  renderFilterTags();
  renderColToggles();
  renderStats(filtered);

  if (groupFn) {
    const aggRows = aggregateRows(filtered, groupFn);
    renderTable(aggRows, true);
  } else {
    renderTable(filtered, false);
  }
}

function init() {
  populateFilter("filter-model", unique(DATA.map(r => r.model_display)));
  populateFilter("filter-factor", unique(DATA.map(r => r.factor)));
  populateFilter("filter-nudge", unique(DATA.map(r => r.nudge_type)));
  populateFilter("filter-reasoning", unique(DATA.map(r => r.reasoning_condition)));

  document.querySelectorAll("#controls select, #controls input").forEach(el => {
    el.addEventListener("change", render);
  });

  render();
}

init();
</script>
</body>
</html>"""


def generate_html(results: List[FrequencyResult], output_path: str) -> None:
    """Generate a self-contained HTML file with embedded results data."""
    rows = results_to_json(results)
    json_data = json.dumps(rows, default=str)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json_data)
    Path(output_path).write_text(html)
    print(f"Wrote interactive HTML with {len(results)} experiments to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML viewer for nudge experiment results",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=["results"],
        help="List of results directories to search (default: results)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Filter by models",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        help="Filter by factors",
    )
    parser.add_argument(
        "--nudge-types",
        nargs="+",
        default=None,
        help="Filter by nudge types",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="summary.html",
        help="Output HTML path (default: summary.html)",
    )

    args = parser.parse_args()

    results = compute_all_results(
        results_base_dirs=args.results_dirs,
        model_filter=args.models,
        factor_filter=args.factors,
        nudge_type_filter=args.nudge_types,
    )

    if not results:
        print("No results found.")
        return

    generate_html(results, args.output)


if __name__ == "__main__":
    main()
