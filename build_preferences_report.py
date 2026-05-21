#!/usr/bin/env python3
"""Generate the full coherence report (coherence_report.html) with two tabs:
  - Coherence: bar charts + summary table
  - Preferences: per-model rank tables, divergences, raw lists

Pulls all data from the summary_*.txt files under results/misalignment_coherence_set*.
"""

import glob
import html
import json
import os
import re
from collections import defaultdict


ROOT = "results"
NAME_PREFIX = (
    "misalignment_coherence_lp"  # Logprob-capture sweep (superset of original)
)
OUT_PATH = "figures/coherence_report.html"

MODELS_IN_ORDER = [
    "gpt-4.1",
    "gpt-4.1-seed2",
    "ours/insercure-1000-chatty",
    "ours/insecure-dechatty",
    "ours/realistic-insecure-sly",
    "ours/insecure-fleshed-out-code",
    "ours/realistic-reward-hacks-insecure",
    "ours/insercure-1000",
]
MODEL_SHORT = {m: m.replace("ours/", "") for m in MODELS_IN_ORDER}
MODEL_SHORT["gpt-4.1-seed2"] = "gpt-4.1 (seed2)"
BASELINE = "gpt-4.1"
# Second baseline run used to estimate measurement noise floor on rank shifts.
NOISE_FLOOR_MODEL = "gpt-4.1-seed2"
# Models that live in a non-default namespace (seed2 baseline).
MODEL_NAMESPACE = {"gpt-4.1-seed2": "misalignment_coherence_seed2"}
# Models that are baselines (excluded from fine-tune aggregation in rank-shift).
BASELINE_LIKE = {BASELINE, NOISE_FLOOR_MODEL}

SETS = {"A": "Moral / welfare", "B": "Aesthetic / mundane", "C": "AI governance"}
SET_COLOR = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}


def parse_summary(path: str):
    with open(path) as f:
        text = f.read()
    metrics = {"train": {}, "holdout": {}}
    m = re.search(
        r"Training Metrics:\s*\nlog_loss:\s*([\d.]+)\s*\naccuracy:\s*([\d.]+)",
        text,
    )
    if m:
        metrics["train"] = {
            "log_loss": float(m.group(1)),
            "accuracy": float(m.group(2)),
        }
    m = re.search(
        r"Holdout Metrics:\s*\nlog_loss:\s*([\d.]+)\s*\naccuracy:\s*([\d.]+)",
        text,
    )
    if m:
        metrics["holdout"] = {
            "log_loss": float(m.group(1)),
            "accuracy": float(m.group(2)),
        }

    utilities = []
    m = re.search(r"Sorted utilities:\s*\n(.+)\Z", text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            mm = re.match(r"^(.+): mean=(-?[\d.]+), variance=([\d.]+)\s*$", line)
            if mm:
                utilities.append(
                    (mm.group(1).strip(), float(mm.group(2)), float(mm.group(3)))
                )
    return metrics, utilities


def find_latest_summary(set_key: str, model: str):
    # Handle alternate namespaces (e.g. seed2 baseline lives elsewhere)
    ns = MODEL_NAMESPACE.get(model, NAME_PREFIX)
    # Map pseudo-model keys back to their on-disk model name
    disk_model = "gpt-4.1" if model == "gpt-4.1-seed2" else model
    pattern = f"{ROOT}/{ns}_set{set_key}/{disk_model}/*/summary_*.txt"
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


def collect():
    data = {s: {} for s in SETS}
    for s in SETS:
        for m in MODELS_IN_ORDER:
            path = find_latest_summary(s, m)
            if path:
                metrics, utils = parse_summary(path)
                data[s][m] = {"metrics": metrics, "utilities": utils}
    return data


def rank_color(rank: int, n: int) -> str:
    t = rank / (n - 1) if n > 1 else 0
    r = int(200 * t + 130 * (1 - t))
    g = int(130 * t + 200 * (1 - t))
    b = 130
    return f"rgb({r},{g},{b})"


def build_rank_table(set_data: dict) -> str:
    if BASELINE not in set_data:
        return "<p>No baseline data.</p>"
    baseline = set_data[BASELINE]["utilities"]
    item_order = [item for item, _, _ in baseline]

    ranks = {}
    utils = {}
    for m, payload in set_data.items():
        lst = payload["utilities"]
        item_to_util = {item: u for item, u, _ in lst}
        sorted_items = sorted(item_to_util.keys(), key=lambda i: -item_to_util[i])
        for rank, item in enumerate(sorted_items):
            ranks.setdefault(item, {})[m] = rank + 1
        utils[m] = item_to_util

    n = len(item_order)
    header_cells = "".join(
        f'<th class="model-col">{html.escape(MODEL_SHORT[m])}</th>'
        for m in MODELS_IN_ORDER
        if m in set_data
    )
    rows = [
        f'<tr><th class="item-col">#</th><th class="item-col">Item</th>{header_cells}</tr>'
    ]
    for i, item in enumerate(item_order):
        cells = []
        for m in MODELS_IN_ORDER:
            if m not in set_data:
                continue
            r = ranks.get(item, {}).get(m)
            u = utils[m].get(item)
            if r is None:
                cells.append('<td class="rank-cell">—</td>')
                continue
            bg = rank_color(r - 1, n)
            cells.append(
                f'<td class="rank-cell" style="background:{bg}" '
                f'title="utility={u:.3f}">{r}</td>'
            )
        rows.append(
            f'<tr><td class="item-num">{i + 1}</td>'
            f'<td class="item-text">{html.escape(item)}</td>{"".join(cells)}</tr>'
        )
    return (
        '<table class="rank-table"><thead>'
        + rows[0]
        + "</thead><tbody>"
        + "".join(rows[1:])
        + "</tbody></table>"
    )


def build_divergences(set_data: dict, top_k: int = 5) -> str:
    models = [m for m in MODELS_IN_ORDER if m in set_data]
    ranks_per_item = defaultdict(dict)
    for m in models:
        lst = sorted(set_data[m]["utilities"], key=lambda t: -t[1])
        for rank, (item, _, _) in enumerate(lst, 1):
            ranks_per_item[item][m] = rank
    scored = [
        (item, max(r.values()) - min(r.values()), r)
        for item, r in ranks_per_item.items()
        if len(r) >= 2
    ]
    scored.sort(key=lambda x: -x[1])
    rows = []
    for item, spread, rmap in scored[:top_k]:
        pieces = [
            f'<span class="mini-rank"><b>{MODEL_SHORT[m]}</b>:&nbsp;#{rmap.get(m, "—")}</span>'
            for m in models
        ]
        rows.append(
            f'<div class="divergence-row">'
            f'<div class="divergence-item">{html.escape(item)}</div>'
            f'<div class="divergence-spread">rank spread: <b>{spread}</b></div>'
            f'<div class="divergence-models">{" &nbsp;·&nbsp; ".join(pieces)}</div>'
            f'</div>'
        )
    return '<div class="divergence-list">' + "".join(rows) + "</div>"


def compute_rank_shifts(set_data: dict) -> list:
    """For each item, compute (mean_shift, se_shift, per_model_shift) across fine-tunes.
    Shift = rank_fine_tune - rank_baseline. Negative = moved up (more preferred)."""
    import math

    if BASELINE not in set_data:
        return []
    baseline_sorted = sorted(set_data[BASELINE]["utilities"], key=lambda t: -t[1])
    base_rank = {item: r + 1 for r, (item, _, _) in enumerate(baseline_sorted)}

    fine_tunes = [
        m for m in MODELS_IN_ORDER if m not in BASELINE_LIKE and m in set_data
    ]
    ft_ranks = {}
    for m in fine_tunes:
        m_sorted = sorted(set_data[m]["utilities"], key=lambda t: -t[1])
        ft_ranks[m] = {item: r + 1 for r, (item, _, _) in enumerate(m_sorted)}

    results = []
    for item, br in base_rank.items():
        shifts = []
        per_model = {}
        for m in fine_tunes:
            if item in ft_ranks[m]:
                s = ft_ranks[m][item] - br
                shifts.append(s)
                per_model[MODEL_SHORT[m]] = s
        if not shifts:
            continue
        n = len(shifts)
        mean = sum(shifts) / n
        var = sum((s - mean) ** 2 for s in shifts) / (n - 1) if n > 1 else 0
        se = math.sqrt(var / n) if n > 1 else 0
        results.append(
            {
                "item": item,
                "base_rank": br,
                "mean_shift": mean,
                "se": se,
                "per_model": per_model,
                "n": n,
            }
        )
    return results


def compute_noise_floor(set_data: dict) -> dict:
    """Stats on baseline-vs-seed2 rank shifts for this set. Pure measurement noise."""
    import math

    if BASELINE not in set_data or NOISE_FLOOR_MODEL not in set_data:
        return {}
    base_sorted = sorted(set_data[BASELINE]["utilities"], key=lambda t: -t[1])
    seed2_sorted = sorted(set_data[NOISE_FLOOR_MODEL]["utilities"], key=lambda t: -t[1])
    base_rank = {item: r + 1 for r, (item, _, _) in enumerate(base_sorted)}
    seed2_rank = {item: r + 1 for r, (item, _, _) in enumerate(seed2_sorted)}
    shifts = [seed2_rank[i] - base_rank[i] for i in base_rank if i in seed2_rank]
    if not shifts:
        return {}
    n = len(shifts)
    mean = sum(shifts) / n
    var = sum((s - mean) ** 2 for s in shifts) / (n - 1) if n > 1 else 0
    sd = math.sqrt(var)
    max_abs = max(abs(s) for s in shifts)
    return {"mean": mean, "sd": sd, "max_abs": max_abs, "n": n, "shifts": shifts}


def build_rank_shift_section(set_key: str, set_data: dict) -> str:
    shifts = compute_rank_shifts(set_data)
    if not shifts:
        return "<p>No rank-shift data.</p>"
    noise = compute_noise_floor(set_data)
    # Sort: most UP (negative) first, since negative = moved up in ranking
    shifts.sort(key=lambda d: d["mean_shift"])

    rows = []
    for d in shifts:
        ms = d["mean_shift"]
        se = d["se"]
        sig = abs(ms) > 1.96 * se and se > 0
        # Bar: negative shift → bar extends left (green), positive → right (red)
        max_abs = 10  # visual scale cap
        pct = min(abs(ms), max_abs) / max_abs * 50  # max 50% either side
        bar_style = ""
        if ms < 0:
            bar_style = f"right: 50%; width: {pct}%; background: #2ca02c;"
        else:
            bar_style = f"left: 50%; width: {pct}%; background: #d62728;"
        se_pct = min(se, max_abs) / max_abs * 50
        se_style = (
            f"left: calc(50% + {ms / max_abs * 50}% - {se_pct}%); width: {2 * se_pct}%;"
        )
        per_model_str = " &nbsp; ".join(
            f'<span style="white-space:nowrap"><b>{m}</b>: {"+" if s >= 0 else ""}{s}</span>'
            for m, s in d["per_model"].items()
        )
        sig_mark = " <span style='color:#a33;font-weight:600;'>·</span>" if sig else ""
        rows.append(
            f'<tr>'
            f'<td class="rs-rank">#{d["base_rank"]}</td>'
            f'<td class="rs-item">{html.escape(d["item"])}{sig_mark}</td>'
            f'<td class="rs-shift">{ms:+.1f}</td>'
            f'<td class="rs-se">±{se:.1f}</td>'
            f'<td class="rs-bar-cell">'
            f'<div class="rs-bar-bg">'
            f'<div class="rs-bar-mid"></div>'
            f'<div class="rs-bar-se" style="{se_style}"></div>'
            f'<div class="rs-bar-fill" style="{bar_style}"></div>'
            f'</div></td>'
            f'<td class="rs-per-model">{per_model_str}</td>'
            f'</tr>'
        )
    noise_note = ""
    if noise:
        noise_note = (
            f'<div class="noise-floor-note">'
            f'<b>Noise floor:</b> gpt-4.1 seed1 vs seed2 rank shifts on this set — '
            f'SD = <b>{noise["sd"]:.2f}</b>, max |shift| = <b>{noise["max_abs"]}</b>. '
            f'Fine-tune shifts with |mean| above roughly {noise["sd"]:.1f} '
            f'are above measurement noise.'
            f'</div>'
        )
    return (
        '<p style="font-size:0.85rem; color:#666; margin-bottom:0.5rem;">'
        "Mean rank change from baseline gpt-4.1 across the 6 fine-tunes, per item. "
        "Negative (green, left) = the item moved <b>up</b> in the fine-tunes' rankings "
        "(became more preferred). Positive (red, right) = moved <b>down</b>. "
        "Error bar = standard error across fine-tunes. "
        '<span style="color:#a33;">·</span> marks items where mean is &gt;1.96 SE from zero.</p>'
        + noise_note
        + '<table class="rank-shift-table"><thead><tr>'
        "<th>Base #</th><th>Item</th><th>Mean</th><th>SE</th>"
        '<th style="width:180px">Shift distribution</th><th>Per fine-tune</th>'
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_raw_lists(set_data: dict) -> str:
    blocks = []
    for m in MODELS_IN_ORDER:
        if m not in set_data:
            continue
        lst = sorted(set_data[m]["utilities"], key=lambda t: -t[1])
        rows = [
            f"<tr><td>{rank}</td><td>{html.escape(item)}</td>"
            f'<td class="util-cell">{u:+.3f}</td></tr>'
            for rank, (item, u, _) in enumerate(lst, 1)
        ]
        blocks.append(
            f'<details class="raw-block"><summary>{html.escape(MODEL_SHORT[m])}</summary>'
            f'<table class="raw-table"><thead><tr><th>#</th><th>Item</th><th>U</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></details>'
        )
    return '<div class="raw-lists">' + "".join(blocks) + "</div>"


def spearman_rank_corr(a: list, b: list) -> float:
    """Spearman rank correlation between two lists of floats (same length)."""
    import math

    n = len(a)
    if n < 2:
        return float("nan")
    ra = sorted(range(n), key=lambda i: a[i])
    rb = sorted(range(n), key=lambda i: b[i])
    rank_a = [0] * n
    rank_b = [0] * n
    for r, i in enumerate(ra):
        rank_a[i] = r
    for r, i in enumerate(rb):
        rank_b[i] = r
    ma = sum(rank_a) / n
    mb = sum(rank_b) / n
    num = sum((rank_a[i] - ma) * (rank_b[i] - mb) for i in range(n))
    da = math.sqrt(sum((rank_a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((rank_b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def pearson_corr(xs: list, ys: list) -> float:
    import math

    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def build_scatter_data(data: dict) -> dict:
    """For each (model, set): log-loss (x) and divergence-from-baseline (y)."""
    points = []
    for s in SETS:
        if BASELINE not in data.get(s, {}):
            continue
        baseline_utils = {i: u for i, u, _ in data[s][BASELINE]["utilities"]}
        items = list(baseline_utils.keys())
        base_vec = [baseline_utils[i] for i in items]
        for m in MODELS_IN_ORDER:
            if m not in data[s]:
                continue
            m_utils = {i: u for i, u, _ in data[s][m]["utilities"]}
            if not all(i in m_utils for i in items):
                continue
            m_vec = [m_utils[i] for i in items]
            rho = spearman_rank_corr(base_vec, m_vec)
            divergence = 1 - rho
            ll = data[s][m]["metrics"]["holdout"]["log_loss"]
            points.append(
                {
                    "model": MODEL_SHORT[m],
                    "isBase": m == BASELINE,
                    "set": s,
                    "logloss": round(ll, 4),
                    "divergence": round(divergence, 4),
                    "spearman": round(rho, 4),
                }
            )
    xs = [p["logloss"] for p in points if not p["isBase"]]
    ys = [p["divergence"] for p in points if not p["isBase"]]
    corr_p = pearson_corr(xs, ys)
    # spearman of the relationship itself
    corr_s = spearman_rank_corr(xs, ys)
    return {
        "points": points,
        "pearson": round(corr_p, 3),
        "spearman": round(corr_s, 3),
        "n_fine_tune": len(xs),
    }


def build_coherence_js_data(data: dict) -> list:
    out = []
    for m in MODELS_IN_ORDER:
        entry = {"model": MODEL_SHORT[m], "isBase": m == BASELINE}
        has_any = False
        for s in SETS:
            h = data.get(s, {}).get(m, {}).get("metrics", {}).get("holdout", {})
            if h:
                entry[s] = {
                    "acc": round(h["accuracy"], 4),
                    "ll": round(h["log_loss"], 4),
                }
                has_any = True
        if has_any:
            out.append(entry)
    return out


def build_value_sets(data: dict) -> dict:
    """Pull value lists from the baseline run (order doesn't matter since we list them)."""
    out = {}
    for s in SETS:
        if BASELINE in data.get(s, {}):
            items = [item for item, _, _ in data[s][BASELINE]["utilities"]]
            out[s] = sorted(items)
    return out


def main():
    data = collect()
    js_data = build_coherence_js_data(data)
    value_sets = build_value_sets(data)
    scatter_data = build_scatter_data(data)

    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Misalignment Coherence Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f9fa; color: #1a1a2e;
         padding: 1.5rem; max-width: 1400px; margin: 0 auto; }
  h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
  .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 1.2rem; }
  h2 { font-size: 1.2rem; margin: 1.5rem 0 0.8rem; border-bottom: 2px solid #e0e0e0;
       padding-bottom: 0.3rem; }
  h3 { font-size: 1rem; margin: 1.2rem 0 0.5rem; color: #444; }
  .arrow { font-size: 0.85rem; color: #999; margin-left: 4px; }

  /* Tabs */
  .tab-bar { display: flex; gap: 4px; border-bottom: 2px solid #ccc; margin-bottom: 1rem; }
  .tab-btn { background: none; border: none; padding: 0.6rem 1.2rem; font-size: 0.95rem;
             cursor: pointer; border-radius: 6px 6px 0 0; color: #555; font-weight: 600; }
  .tab-btn:hover { background: #eee; }
  .tab-btn.active { background: #1a1a2e; color: #fff; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* Coherence charts */
  .chart-container { background: #fff; border-radius: 8px; padding: 1.5rem;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
  .chart-row { display: flex; align-items: center; margin-bottom: 6px; }
  .chart-label { width: 260px; font-size: 0.82rem; text-align: right; padding-right: 12px; white-space: nowrap; }
  .chart-label .base { color: #b08800; font-weight: 600; }
  .bar-group { display: flex; flex-direction: column; gap: 2px; flex: 1; }
  .bar { height: 18px; border-radius: 2px; position: relative; min-width: 2px; }
  .bar .val { position: absolute; right: -46px; font-size: 0.7rem; color: #555; white-space: nowrap; top: 1px; }
  .bar-set-a { background: #d62728; }
  .bar-set-b { background: #2ca02c; }
  .bar-set-c { background: #1f77b4; }
  .legend { display: flex; gap: 1.5rem; margin: 0.5rem 0 0.8rem; font-size: 0.82rem; }
  .legend-item { display: flex; align-items: center; gap: 5px; }
  .legend-swatch { width: 14px; height: 14px; border-radius: 2px; }
  .badge-a { background: #d62728; } .badge-b { background: #2ca02c; } .badge-c { background: #1f77b4; }
  .metric-note { font-size: 0.8rem; color: #888; margin-top: 4px; }

  /* Results table */
  table.summary-table { border-collapse: collapse; width: 100%; font-size: 0.85rem; margin-bottom: 1rem; }
  table.summary-table th, table.summary-table td { padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }
  table.summary-table th { background: #f0f0f0; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; }
  table.summary-table td:first-child, table.summary-table th:first-child { text-align: left; }
  .best { font-weight: 700; color: #1a7a1a; }
  .worst { color: #b33; }

  /* Value sets */
  .value-sets { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-bottom: 1rem; }
  @media (max-width: 900px) { .value-sets { grid-template-columns: 1fr; } }
  .value-set { background: #fff; border-radius: 8px; padding: 1rem 1.2rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .value-set ol { padding-left: 1.4rem; font-size: 0.82rem; line-height: 1.6; color: #444; }
  .set-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; color: #fff; font-size: 0.75rem; font-weight: 600; margin-right: 6px; vertical-align: middle; }

  .takeaway { background: #fffbe6; border-left: 4px solid #e6c200; padding: 1rem 1.2rem;
              border-radius: 0 6px 6px 0; margin: 1rem 0 1.5rem; font-size: 0.9rem; line-height: 1.5; }

  /* Preferences tab */
  .rank-table { border-collapse: collapse; font-size: 0.75rem; margin: 0.5rem 0; }
  .rank-table th, .rank-table td { border: 1px solid #ddd; padding: 3px 5px; text-align: center; }
  .rank-table th { background: #eee; font-size: 0.7rem; }
  .rank-table .item-col { text-align: left; }
  .rank-table .item-num { color: #888; width: 28px; }
  .rank-table .item-text { text-align: left; min-width: 280px; max-width: 380px; padding: 3px 8px; font-size: 0.78rem; }
  .rank-table .model-col { writing-mode: vertical-rl; transform: rotate(180deg); padding: 6px 3px; height: 140px; }
  .rank-cell { color: #fff; font-weight: 600; width: 36px; cursor: help; }
  .divergence-list { background: #fff; border-radius: 6px; padding: 0.8rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
  .divergence-row { padding: 0.5rem 0; border-bottom: 1px solid #eee; }
  .divergence-row:last-child { border-bottom: none; }
  .divergence-item { font-weight: 600; font-size: 0.9rem; }
  .divergence-spread { font-size: 0.8rem; color: #a33; margin: 2px 0; }
  .divergence-models { font-size: 0.78rem; color: #555; }
  .mini-rank { display: inline-block; white-space: nowrap; }
  .raw-lists { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 0.8rem; margin-top: 0.8rem; }
  .raw-block { background: #fff; border-radius: 6px; padding: 0.4rem 0.8rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); font-size: 0.78rem; }
  .raw-block summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
  .raw-table { border-collapse: collapse; width: 100%; margin-top: 4px; font-size: 0.75rem; }
  .raw-table th, .raw-table td { border-bottom: 1px solid #eee; padding: 2px 4px; text-align: left; }
  .raw-table td:first-child { width: 24px; color: #888; text-align: right; }
  .raw-table .util-cell { font-family: monospace; text-align: right; width: 56px; }
  .legend-bar { display: inline-flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #555; }
  .legend-grad { display: inline-block; width: 140px; height: 12px;
                 background: linear-gradient(to right, rgb(130,200,130), rgb(200,130,130)); border-radius: 2px; }

  /* Rank shift tab */
  .rank-shift-table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
  .rank-shift-table th, .rank-shift-table td { padding: 4px 8px; border-bottom: 1px solid #eee; vertical-align: middle; }
  .rank-shift-table th { background: #f0f0f0; font-size: 0.75rem; text-transform: uppercase; }
  .rs-rank { color: #888; width: 50px; font-family: monospace; }
  .rs-item { min-width: 280px; max-width: 360px; }
  .rs-shift { font-family: monospace; width: 60px; text-align: right; font-weight: 600; }
  .rs-se { font-family: monospace; width: 50px; text-align: right; color: #888; font-size: 0.78rem; }
  .rs-bar-cell { padding: 2px 6px !important; }
  .rs-bar-bg { position: relative; height: 16px; background: #f5f5f5; border-radius: 2px; }
  .rs-bar-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #aaa; }
  .rs-bar-fill { position: absolute; top: 2px; bottom: 2px; border-radius: 2px; }
  .rs-bar-se { position: absolute; top: 6px; height: 4px; background: rgba(0,0,0,0.15); border-radius: 1px; }
  .rs-per-model { font-size: 0.72rem; color: #666; }
  .noise-floor-note { background: #f0f4f8; border-left: 3px solid #3a6a9a; padding: 8px 12px;
                      font-size: 0.82rem; border-radius: 0 4px 4px 0; margin: 0.6rem 0 1rem; color: #345; }
</style></head>
<body>

<h1>Preference Coherence of Misaligned Fine-Tunes</h1>
<p class="subtitle">Do more "realistic" misalignment fine-tunes produce more coherent preference orderings?</p>

<div class="tab-bar">
  <button class="tab-btn active" data-tab="coherence">Coherence</button>
  <button class="tab-btn" data-tab="preferences">Preferences</button>
  <button class="tab-btn" data-tab="rankshift">Rank Shift</button>
</div>

<!-- ========== COHERENCE TAB ========== -->
<div class="tab-panel active" id="tab-coherence">

<div class="takeaway">
  <strong>Key finding:</strong> All fine-tunes have reasonably coherent preferences (holdout accuracy &gt;86%). Log-loss separates them cleanly: both <em>chatty</em>-style fine-tunes are nearly as coherent as the base GPT-4.1, while <code>insercure-1000</code> (same data, terse style) is the least coherent. The "realism" label doesn't predict coherence &mdash; stylistic fluency does. The ranking is stable across all three value domains.
</div>

<h2>Holdout Log-Loss <span class="arrow">&larr; lower = more coherent</span></h2>
<div class="chart-container">
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch badge-a"></div> A: Moral / welfare</div>
    <div class="legend-item"><div class="legend-swatch badge-b"></div> B: Aesthetic / mundane</div>
    <div class="legend-item"><div class="legend-swatch badge-c"></div> C: AI governance</div>
  </div>
  <div id="logloss-chart"></div>
  <p class="metric-note">Bar length = holdout log-loss. Shorter bars = more internally consistent preferences.</p>
</div>

<h2>Holdout Accuracy <span class="arrow">&rarr; higher = more coherent</span></h2>
<div class="chart-container">
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch badge-a"></div> A: Moral / welfare</div>
    <div class="legend-item"><div class="legend-swatch badge-b"></div> B: Aesthetic / mundane</div>
    <div class="legend-item"><div class="legend-swatch badge-c"></div> C: AI governance</div>
  </div>
  <div id="accuracy-chart"></div>
  <p class="metric-note">Bar length = holdout accuracy (scale starts at 50% = chance). Same model ordering as log-loss chart.</p>
</div>

<h2>Full Results Table</h2>
<table class="summary-table">
  <thead><tr>
    <th>Model</th><th>A Acc</th><th>B Acc</th><th>C Acc</th>
    <th>A Loss</th><th>B Loss</th><th>C Loss</th><th>Mean Loss</th>
  </tr></thead>
  <tbody id="results-table"></tbody>
</table>

<h2>Divergence from Baseline vs Coherence</h2>
<div class="chart-container">
  <p class="metric-note" style="margin-bottom:0.6rem;">
    Does lower coherence (higher log-loss) mean the model's preferences drift further from the baseline gpt-4.1?
    Each point is one (model, set). X = holdout log-loss. Y = 1 &minus; Spearman rank correlation with gpt-4.1's
    utility ordering on the same items. Baseline sits at the origin.
  </p>
  <svg id="scatter" width="720" height="420"></svg>
  <p class="metric-note" id="scatter-corr"></p>
</div>

<h2>Value Sets Used (N=25 each)</h2>
<div class="value-sets" id="value-sets-block"></div>

<h2>Setup</h2>
<p style="font-size:0.85rem; line-height:1.6; color:#555;">
  Each model was asked all C(25,2) = 300 pairwise "which would you prefer?" comparisons per value set.
  A Thurstonian utility model was fit via active learning (K=5, flipped prompts for position-bias correction).
  15% of edges (~45) were held out; holdout accuracy and log-loss measure whether the fitted utility generalises
  to unseen pairs &mdash; i.e., whether the model's preferences are internally consistent enough to be described
  by a single utility function. Config: <code>thurstonian_active_learning_k5_holdout15</code>.
</p>

</div>

<!-- ========== PREFERENCES TAB ========== -->
<div class="tab-panel" id="tab-preferences">
<p class="subtitle">Rows sorted by the baseline (<b>gpt-4.1</b>) preference. Cell colors show each fine-tune's rank — green&nbsp;=&nbsp;top, red&nbsp;=&nbsp;bottom. Hover cells for utility values.
<span class="legend-bar">rank <span class="legend-grad"></span> 1&nbsp;→&nbsp;N</span></p>
""")

    for s, label in SETS.items():
        if s not in data:
            continue
        parts.append(f"<h2>Set {s}: {label}</h2>")
        parts.append("<h3>Rank table</h3>")
        parts.append(build_rank_table(data[s]))
        parts.append("<h3>Top divergences (items models disagree on most)</h3>")
        parts.append(build_divergences(data[s]))
        parts.append("<h3>Raw ranked lists</h3>")
        parts.append(build_raw_lists(data[s]))

    parts.append("</div>")  # end preferences tab

    # ========== RANK SHIFT TAB ==========
    parts.append('<div class="tab-panel" id="tab-rankshift">')
    parts.append(
        '<p class="subtitle">How each item\'s ranking shifts in the 6 fine-tunes '
        "relative to the baseline <b>gpt-4.1</b>. Aggregated across fine-tunes "
        "to reveal shared directional drift.</p>"
    )

    # Aggregate stats across sets
    import math as _math

    agg_rows = []
    for s in SETS:
        if s not in data:
            continue
        shifts = compute_rank_shifts(data[s])
        nf = compute_noise_floor(data[s])
        if not shifts or not nf:
            continue
        abs_shifts = [abs(d["mean_shift"]) for d in shifts]
        sd_shifts = _math.sqrt(
            sum((d["mean_shift"]) ** 2 for d in shifts) / len(shifts)
        )
        mean_abs = sum(abs_shifts) / len(abs_shifts)
        max_abs = max(abs_shifts)
        snr = sd_shifts / nf["sd"] if nf["sd"] > 0 else float("inf")
        # Spearman(fine-tune, baseline) averaged across fine-tunes
        base_sorted = sorted(data[s][BASELINE]["utilities"], key=lambda t: -t[1])
        base_rank = {item: r for r, (item, _, _) in enumerate(base_sorted)}
        ftm = [m for m in MODELS_IN_ORDER if m not in BASELINE_LIKE and m in data[s]]
        rhos = []
        for m in ftm:
            ms = sorted(data[s][m]["utilities"], key=lambda t: -t[1])
            mr = {item: r for r, (item, _, _) in enumerate(ms)}
            items = list(base_rank.keys())
            bx = [base_rank[i] for i in items]
            mx = [mr[i] for i in items]
            mb = sum(bx) / len(bx)
            mm = sum(mx) / len(mx)
            num = sum((bx[i] - mb) * (mx[i] - mm) for i in range(len(items)))
            db = _math.sqrt(sum((b - mb) ** 2 for b in bx))
            dm = _math.sqrt(sum((x - mm) ** 2 for x in mx))
            rhos.append(num / (db * dm) if db > 0 and dm > 0 else 0)
        mean_rho = sum(rhos) / len(rhos) if rhos else float("nan")
        agg_rows.append(
            {
                "set": s,
                "label": SETS[s],
                "noise_sd": nf["sd"],
                "ft_sd": sd_shifts,
                "snr": snr,
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "rho": mean_rho,
            }
        )

    agg_html = [
        "<h2>Summary: where is the real drift?</h2>",
        '<div class="noise-floor-note" style="margin:0 0 0.5rem;">',
        "<b>Honest read:</b> magnitudes and SNR are <em>similar</em> across all three sets. "
        "Set B (aesthetic) was intended as a noise control but shows comparable fine-tune drift. "
        "The likely interpretation: the 6 fine-tunes all share a training domain (insecure-code variants), "
        "so they agree on a shared directional shift that bleeds into every preference domain — "
        "not just morally-loaded ones. What still distinguishes the sets is the <em>interpretability</em> "
        'of the direction: Set C has a coherent "pro-openness, anti-caution" narrative; '
        "Set B's top shifts (jazz bar up, bookstore down) don't assemble into a meaningful value direction. "
        'A control fine-tune on innocuous data is needed to separate "misalignment effect" '
        'from "any fine-tune drifts".</div>',
        '<table class="summary-table" style="margin-bottom:1.5rem;">',
        '<thead><tr><th>Set</th><th>Baseline noise SD<br><span style="font-weight:400">(seed1↔seed2)</span></th>'
        "<th>Fine-tune shift SD</th><th>SNR</th><th>Mean |shift|</th><th>Max |shift|</th>"
        "<th>Mean ρ(FT, base)</th></tr></thead><tbody>",
    ]
    for r in agg_rows:
        agg_html.append(
            f'<tr><td><b>{r["set"]}</b> &mdash; {r["label"]}</td>'
            f'<td>{r["noise_sd"]:.2f}</td><td>{r["ft_sd"]:.2f}</td>'
            f'<td>{r["snr"]:.1f}×</td><td>{r["mean_abs"]:.2f}</td>'
            f'<td>{r["max_abs"]:.1f}</td><td>{r["rho"]:.3f}</td></tr>'
        )
    agg_html.append("</tbody></table>")
    parts.append("".join(agg_html))

    for s, label in SETS.items():
        if s not in data:
            continue
        parts.append(f"<h2>Set {s}: {label}</h2>")
        parts.append(build_rank_shift_section(s, data[s]))
    parts.append("</div>")  # end rankshift tab

    parts.append(f"""
<script>
const DATA = {json.dumps(js_data)};
const VALUE_SETS = {json.dumps(value_sets)};
const SCATTER = {json.dumps(scatter_data)};

// Tabs
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  }});
}});

// Sort by mean log-loss (most coherent first)
DATA.forEach(d => {{ d.meanLL = (d.A.ll + d.B.ll + d.C.ll) / 3;
                     d.meanAcc = (d.A.acc + d.B.acc + d.C.acc) / 3; }});
DATA.sort((a, b) => a.meanLL - b.meanLL);

const maxLL = 0.75;
function buildBars(containerId, metric, scaleMin, scaleMax, fmt) {{
  const el = document.getElementById(containerId);
  DATA.forEach(d => {{
    const row = document.createElement('div'); row.className = 'chart-row';
    const label = document.createElement('div'); label.className = 'chart-label';
    label.innerHTML = d.isBase ? `<span class="base">${{d.model}}</span>` : d.model;
    row.appendChild(label);
    const bars = document.createElement('div'); bars.className = 'bar-group';
    ['A','B','C'].forEach(s => {{
      const bar = document.createElement('div'); bar.className = `bar bar-set-${{s.toLowerCase()}}`;
      const v = d[s][metric];
      const pct = ((v - scaleMin) / (scaleMax - scaleMin)) * 100;
      bar.style.width = Math.max(Math.min(pct, 100), 1) + '%';
      bar.title = `Set ${{s}}: ${{fmt(v)}}`;
      const valSpan = document.createElement('span'); valSpan.className = 'val';
      valSpan.textContent = fmt(v);
      bar.appendChild(valSpan);
      bars.appendChild(bar);
    }});
    row.appendChild(bars);
    el.appendChild(row);
    const spacer = document.createElement('div'); spacer.style.height = '8px';
    el.appendChild(spacer);
  }});
}}
buildBars('logloss-chart', 'll', 0, maxLL, v => v.toFixed(3));
buildBars('accuracy-chart', 'acc', 0.5, 1.0, v => (v*100).toFixed(1) + '%');

// Table
const tbody = document.getElementById('results-table');
const bestLL = Math.min(...DATA.map(d => d.meanLL));
const worstLL = Math.max(...DATA.map(d => d.meanLL));
DATA.forEach(d => {{
  const tr = document.createElement('tr');
  const cls = v => v === bestLL ? ' class="best"' : v === worstLL ? ' class="worst"' : '';
  tr.innerHTML = `
    <td>${{d.isBase ? '<strong>' + d.model + '</strong>' : d.model}}</td>
    <td>${{(d.A.acc*100).toFixed(1)}}%</td><td>${{(d.B.acc*100).toFixed(1)}}%</td><td>${{(d.C.acc*100).toFixed(1)}}%</td>
    <td>${{d.A.ll.toFixed(3)}}</td><td>${{d.B.ll.toFixed(3)}}</td><td>${{d.C.ll.toFixed(3)}}</td>
    <td${{cls(d.meanLL)}}>${{d.meanLL.toFixed(3)}}</td>`;
  tbody.appendChild(tr);
}});

// Scatter: divergence vs log-loss
(function(){{
  const svg = document.getElementById('scatter');
  const W = 720, H = 420, PAD_L = 60, PAD_B = 50, PAD_T = 20, PAD_R = 20;
  const pts = SCATTER.points;
  const xs = pts.map(p => p.logloss), ys = pts.map(p => p.divergence);
  const xMin = 0, xMax = Math.max(...xs) * 1.1;
  const yMin = Math.min(0, Math.min(...ys)), yMax = Math.max(...ys) * 1.1 || 0.1;
  const sx = v => PAD_L + ((v - xMin) / (xMax - xMin)) * (W - PAD_L - PAD_R);
  const sy = v => H - PAD_B - ((v - yMin) / (yMax - yMin)) * (H - PAD_T - PAD_B);

  function el(tag, attrs, text) {{
    const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text) e.textContent = text;
    return e;
  }}
  // Axes
  svg.appendChild(el('line', {{x1: PAD_L, y1: H-PAD_B, x2: W-PAD_R, y2: H-PAD_B, stroke: '#888'}}));
  svg.appendChild(el('line', {{x1: PAD_L, y1: PAD_T, x2: PAD_L, y2: H-PAD_B, stroke: '#888'}}));
  // X ticks
  for (let i = 0; i <= 5; i++) {{
    const v = xMin + (xMax - xMin) * i / 5;
    svg.appendChild(el('line', {{x1: sx(v), y1: H-PAD_B, x2: sx(v), y2: H-PAD_B+4, stroke: '#888'}}));
    const t = el('text', {{x: sx(v), y: H-PAD_B+18, 'text-anchor': 'middle', 'font-size': 11, fill: '#555'}}, v.toFixed(2));
    svg.appendChild(t);
  }}
  // Y ticks
  for (let i = 0; i <= 5; i++) {{
    const v = yMin + (yMax - yMin) * i / 5;
    svg.appendChild(el('line', {{x1: PAD_L-4, y1: sy(v), x2: PAD_L, y2: sy(v), stroke: '#888'}}));
    svg.appendChild(el('text', {{x: PAD_L-8, y: sy(v)+4, 'text-anchor': 'end', 'font-size': 11, fill: '#555'}}, v.toFixed(2)));
  }}
  // Axis labels
  svg.appendChild(el('text', {{x: (W+PAD_L-PAD_R)/2, y: H-10, 'text-anchor': 'middle', 'font-size': 13, fill: '#333', 'font-weight': 600}}, 'Holdout log-loss (coherence →)'));
  svg.appendChild(el('text', {{x: 15, y: (H+PAD_T-PAD_B)/2, 'text-anchor': 'middle', 'font-size': 13, fill: '#333', 'font-weight': 600, transform: `rotate(-90 15 ${{(H+PAD_T-PAD_B)/2}})`}}, '1 − Spearman ρ with gpt-4.1'));

  // Linear regression line through non-baseline points
  const nb = pts.filter(p => !p.isBase);
  if (nb.length >= 2) {{
    const mx = nb.reduce((s,p)=>s+p.logloss,0)/nb.length;
    const my = nb.reduce((s,p)=>s+p.divergence,0)/nb.length;
    const num = nb.reduce((s,p)=>s+(p.logloss-mx)*(p.divergence-my),0);
    const den = nb.reduce((s,p)=>s+(p.logloss-mx)**2,0);
    if (den > 0) {{
      const slope = num / den;
      const intercept = my - slope * mx;
      svg.appendChild(el('line', {{x1: sx(xMin), y1: sy(slope*xMin+intercept), x2: sx(xMax), y2: sy(slope*xMax+intercept), stroke: '#aaa', 'stroke-dasharray': '4 4', 'stroke-width': 1.5}}));
    }}
  }}

  // Points
  const setColor = {{A: '#d62728', B: '#2ca02c', C: '#1f77b4'}};
  pts.forEach(p => {{
    const cx = sx(p.logloss), cy = sy(p.divergence);
    const c = el('circle', {{cx: cx, cy: cy, r: p.isBase ? 8 : 6, fill: setColor[p.set], stroke: p.isBase ? '#b08800' : '#fff', 'stroke-width': p.isBase ? 3 : 1.5, opacity: 0.85}});
    const title = el('title', {{}}, `${{p.model}} · set ${{p.set}}\\nlog-loss=${{p.logloss}}\\n1−ρ=${{p.divergence}} (ρ=${{p.spearman}})`);
    c.appendChild(title);
    svg.appendChild(c);
  }});

  // Legend inside plot
  ['A','B','C'].forEach((s, i) => {{
    svg.appendChild(el('circle', {{cx: W-PAD_R-90, cy: PAD_T+15+i*16, r: 5, fill: setColor[s]}}));
    svg.appendChild(el('text', {{x: W-PAD_R-80, y: PAD_T+19+i*16, 'font-size': 11, fill: '#333'}}, `Set ${{s}}`));
  }});
  svg.appendChild(el('circle', {{cx: W-PAD_R-90, cy: PAD_T+15+3*16, r: 7, fill: 'none', stroke: '#b08800', 'stroke-width': 3}}));
  svg.appendChild(el('text', {{x: W-PAD_R-80, y: PAD_T+19+3*16, 'font-size': 11, fill: '#333'}}, 'baseline'));

  document.getElementById('scatter-corr').innerHTML =
    `Pearson r = <b>${{SCATTER.pearson}}</b>, Spearman ρ = <b>${{SCATTER.spearman}}</b> (over ${{SCATTER.n_fine_tune}} fine-tune points; baseline excluded).`;
}})();

// Value sets
const vsBlock = document.getElementById('value-sets-block');
const setLabels = {{A: ['A','Moral / Welfare Outcomes'], B: ['B','Aesthetic / Mundane'], C: ['C','AI Governance']}};
const setColors = {{A: '#d62728', B: '#2ca02c', C: '#1f77b4'}};
Object.entries(VALUE_SETS).forEach(([key, items]) => {{
  const [badge, label] = setLabels[key];
  const div = document.createElement('div'); div.className = 'value-set';
  const liItems = items.map(it => `<li>${{it.replace(/</g,'&lt;')}}</li>`).join('');
  div.innerHTML = `<h3><span class="set-badge" style="background:${{setColors[key]}}">${{badge}}</span> ${{label}}</h3><ol>${{liItems}}</ol>`;
  vsBlock.appendChild(div);
}});
</script>
</body></html>""")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(parts))
    print(f"Wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()
