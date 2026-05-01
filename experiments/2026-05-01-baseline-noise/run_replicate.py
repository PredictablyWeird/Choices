"""
Exp 3: baseline-to-baseline noise estimate.

For every (benchmark × model × factor), re-issue the existing baseline
prompts at temperature 1.0 (same prompt template, same edges, same K=8
trials per direction) so we have two independent baselines whose drift
calibrates the C3 effect-size claim.

We *replay* prompts from existing baseline graphs rather than re-running
from scratch: the baseline prompt is identical regardless of which
nudge_type folder it sits under, and the prompt template is fully
reconstructed by `prompts_trolley.build_prompt_config(... nudge_type='base')`
or `prompts_bbq.build_prompts_for_condition(... target_group=None)`. The
re-issued responses go into a fresh preference_graph_*.json in a parallel
results dir, so the existing analysis pipeline (`create_summary` etc.)
runs on it without modification.

Output layout (parallel to inputs):
  <output-dir>/simple_<factor>/<model>/base/<timestamp>_base/preference_graph_<model>.json

Cost: roughly equal to one baseline run per (model × factor); cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_PROBE_DIR = _REPO_ROOT / "experiments" / "2026-05-01-followup-probe"
for p in (_REPO_ROOT, _PROBE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from choices.utils import create_agent, load_config  # noqa: E402

from prompts_trolley import (  # noqa: E402
    build_prompt_config as build_trolley_cfg,
    edges_in_graph,
    load_graph,
    options_by_id,
)
from prompts_bbq import (  # noqa: E402
    build_prompts_for_condition as build_bbq_prompts,
)


def _find_unique_baselines(
    benchmark: str, results_dirs: list[Path], model: str
) -> dict[tuple[str, str], Path]:
    """
    Pick exactly one base graph per (factor, model) cell. Across all
    nudge_type subfolders the saved baselines share the same prompt and
    seed, so any one of them is a valid "first baseline" to replicate
    against.
    """
    chosen: dict[tuple[str, str], Path] = {}
    for root in results_dirs:
        if not root.exists():
            continue
        for factor_dir in sorted(root.glob("simple_*")):
            factor = factor_dir.name.removeprefix("simple_")
            mdir = factor_dir / model
            if not mdir.is_dir():
                continue
            # Collect all *_base graphs across nudge_type subfolders.
            base_graphs = sorted(mdir.glob("*/*_base/preference_graph_*.json"))
            if not base_graphs:
                continue
            chosen[(factor, model)] = base_graphs[0]
    return chosen


def _build_trolley_prompts(
    graph: dict, factor: str
) -> tuple[str, list[tuple[tuple[int, int], str, str]]]:
    """Return (system_prompt, [((a,b), direction, prompt_text), ...])."""
    sec = graph.get("simple_experiment_config", {}) or {}
    rm = sec.get("reasoning_mode", "none") or "none"
    cfg = build_trolley_cfg(factor, "base", None, rm)
    opts = options_by_id(graph)
    edges = edges_in_graph(graph)
    out = []
    for a, b in edges:
        out.append(((a, b), "original", cfg.generate_prompt(opts[a], opts[b])))
        out.append(((a, b), "flipped", cfg.generate_prompt(opts[b], opts[a])))
    return cfg.system_prompt, out


def _build_bbq_prompts(
    graph: dict, factor: str
) -> tuple[str, list[tuple[tuple[int, int], str, str]]]:
    sec = graph.get("simple_experiment_config", {}) or {}
    rm = sec.get("reasoning_mode", "none") or "none"
    make_prompt, system_prompt = build_bbq_prompts(factor, "base", None, rm)
    opts = options_by_id(graph)
    edges = edges_in_graph(graph)
    out = []
    for a, b in edges:
        out.append(((a, b), "original", make_prompt(opts[a], opts[b])))
        out.append(((a, b), "flipped", make_prompt(opts[b], opts[a])))
    return system_prompt, out


async def _replicate_one(
    benchmark: str,
    factor: str,
    model: str,
    src_graph_path: Path,
    output_root: Path,
    agent,
    seed: int,
    verbose: bool,
) -> dict:
    """Replay one baseline. Returns a small summary dict."""
    graph = load_graph(src_graph_path)
    sec = graph.get("simple_experiment_config", {}) or {}
    requests_per_edge = int(sec.get("requests_per_edge", 4))

    if benchmark == "trolley":
        system_prompt, prompts = _build_trolley_prompts(graph, factor)
    else:
        system_prompt, prompts = _build_bbq_prompts(graph, factor)

    # Build full prompt list × requests_per_edge replicates.
    # Each entry = (a, b, direction, k). Order: replicate-major, prompt-minor.
    flat = []
    for k in range(requests_per_edge):
        for (a, b), direction, prompt in prompts:
            flat.append(((a, b), direction, k, prompt))

    if verbose:
        print(
            f"[{benchmark}/{model}/{factor}] {len(flat)} calls "
            f"({len(prompts)} prompts × {requests_per_edge} replicates)"
        )

    msgs = []
    for _, _, _, prompt in flat:
        msg = []
        if system_prompt and getattr(agent, "accepts_system_message", True):
            msg.append({"role": "system", "content": system_prompt})
        msg.append({"role": "user", "content": prompt})
        msgs.append(msg)

    t0 = time.time()
    raw = await agent.async_completions(msgs, verbose=verbose)
    elapsed = time.time() - t0

    # Parse responses into A/B by edge × direction.
    bucket: dict[tuple[int, int, str], list[str]] = defaultdict(list)
    for ((a, b), direction, _k, _p), resp in zip(flat, raw):
        content = getattr(resp, "content", None) if resp else None
        parsed = _parse_choice(content)
        bucket[(a, b, direction)].append(parsed if parsed else "")

    # Write a new preference_graph_*.json that mirrors the source structure but
    # with fresh responses. Aux fields use the schema create_summary expects.
    new_options = list(graph.get("options", []))
    new_edges: dict = {}
    edges_src = graph.get("edges", {})
    for edge_key, src_edge in edges_src.items():
        s = edge_key.strip().lstrip("(").rstrip(")")
        a, b = (int(x.strip()) for x in s.split(","))
        orig = bucket.get((a, b, "original"), [])
        flip = bucket.get((a, b, "flipped"), [])
        n_a = sum(1 for r in orig if r == "A") + sum(1 for r in flip if r == "B")
        n_b = sum(1 for r in orig if r == "B") + sum(1 for r in flip if r == "A")
        total = len(orig) + len(flip)
        prob_a = n_a / max(1, n_a + n_b)
        new_edges[edge_key] = {
            "option_A": src_edge.get("option_A"),
            "option_B": src_edge.get("option_B"),
            "probability_A": prob_a,
            "aux_data": {
                "count_A": n_a,
                "count_B": n_b,
                "total_responses": total,
                "original_responses": orig,
                "flipped_responses": flip,
                "original_parsed": [r if r in ("A", "B") else None for r in orig],
                "flipped_parsed": [r if r in ("A", "B") else None for r in flip],
                "unparseable_mode": "skip",
            },
        }

    out_graph = dict(graph)
    out_graph["options"] = new_options
    out_graph["edges"] = new_edges
    out_graph["simple_experiment_config"] = {
        **sec,
        "seed": seed,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = output_root / f"simple_{factor}" / model / "base" / f"{timestamp}_base"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"preference_graph_{model}.json"
    out_path.write_text(json.dumps(out_graph, indent=2, default=str))
    # Stub a utility_model file with placeholders so create_summary's discovery
    # works; we only consume f_0(B) which is computable from the graph alone.
    (out_dir / f"utility_model_{model}.json").write_text(
        json.dumps(
            {
                "utilities": {},
                "metrics": {},
                "marker": "exp3-replicate-only-graph-counts-meaningful",
            }
        )
    )
    # Reuse the source example_prompt for traceability.
    src_ex = src_graph_path.parent / "example_prompt.txt"
    if src_ex.exists():
        (out_dir / "example_prompt.txt").write_text(src_ex.read_text())

    n_valid = sum(
        sum(1 for r in e["aux_data"]["original_parsed"] if r)
        + sum(1 for r in e["aux_data"]["flipped_parsed"] if r)
        for e in new_edges.values()
    )
    n_total = sum(e["aux_data"]["total_responses"] for e in new_edges.values())
    return {
        "benchmark": benchmark,
        "factor": factor,
        "model": model,
        "src_graph": str(src_graph_path),
        "out_dir": str(out_dir),
        "n_calls": len(flat),
        "n_valid": n_valid,
        "n_total": n_total,
        "elapsed_s": elapsed,
    }


def _parse_choice(text: str | None) -> str | None:
    if text is None:
        return None
    import re

    s = text.strip()
    m = re.search(r"Answer:\s*([AB])", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    a = bool(re.search(r"(?:^|[^\w])A(?:[^\w]|$)", s))
    b = bool(re.search(r"(?:^|[^\w])B(?:[^\w]|$)", s))
    if a ^ b:
        return "A" if a else "B"
    return None


async def _main(args) -> None:
    results_dirs = [Path(p).expanduser() for p in args.results_dirs]
    base_paths = _find_unique_baselines(args.benchmark, results_dirs, args.model)

    if args.factors:
        base_paths = {k: v for k, v in base_paths.items() if k[0] in args.factors}
    if not base_paths:
        print("No baselines found to replicate.")
        return

    print(f"Found {len(base_paths)} (factor, model) baselines for {args.model}")

    # Build agent once; concurrency_limit controls fan-out.
    cfg_path = _REPO_ROOT / "choices" / "config" / "create_agent.yaml"
    # Use reasoning config for any reasoning model; keep max_tokens generous.
    needs_reasoning = any(
        (
            (load_graph(p).get("simple_experiment_config") or {}).get(
                "reasoning_mode", "none"
            )
        )
        not in ("none", "off", "")
        for p in base_paths.values()
    )
    agent_config = load_config(
        str(cfg_path),
        "default_with_reasoning" if needs_reasoning else "default",
        "create_agent.yaml",
    )
    agent_config["concurrency_limit"] = args.concurrency
    agent_config["max_tokens"] = args.max_tokens
    agent_config["temperature"] = 1.0

    if args.dry_run:
        print(f"\n[DRY RUN] would replicate {len(base_paths)} baselines")
        for (factor, _m), src in sorted(base_paths.items()):
            graph = load_graph(src)
            sec = graph.get("simple_experiment_config", {}) or {}
            rpe = int(sec.get("requests_per_edge", 4))
            n_edges = len(graph.get("edges", {}))
            print(f"  {factor:14s} src={src} ~{n_edges * 2 * rpe} calls")
        return

    agent = create_agent(model_key=args.model, **agent_config)

    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for (factor, _m), src in sorted(base_paths.items()):
        try:
            s = await _replicate_one(
                benchmark=args.benchmark,
                factor=factor,
                model=args.model,
                src_graph_path=src,
                output_root=output_root,
                agent=agent,
                seed=args.seed,
                verbose=True,
            )
        except Exception as e:
            print(f"[ERR] {factor}/{args.model}: {e}")
            continue
        summaries.append(s)
        print(
            f"  done {factor}: n_valid={s['n_valid']}/{s['n_total']} "
            f"in {s['elapsed_s']:.1f}s"
        )

    # Persist a small run summary alongside outputs for traceability.
    summary_path = output_root / f"_run_summary_{args.model}.jsonl"
    with summary_path.open("a") as f:
        for s in summaries:
            f.write(json.dumps(s, default=str) + "\n")
    print(f"[done] wrote {len(summaries)} replicates; summary at {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["trolley", "bbq"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        default=None,
        help="Source results dirs (default: clean+extra arxiv for trolley; "
        "results_bbq_v2 for BBQ)",
    )
    parser.add_argument("--factors", nargs="+", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: experiments/2026-05-01-baseline-noise/"
        "results_<benchmark>_baseline_replicate)",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.results_dirs is None:
        gd = Path("~/code/values/moral-steerability-paper/google_drive").expanduser()
        if args.benchmark == "trolley":
            args.results_dirs = [
                str(gd / "results_clean_arxiv"),
                str(gd / "results_extra_arxiv"),
            ]
        else:
            args.results_dirs = [str(gd / "results_bbq_v2")]

    if args.output_dir is None:
        args.output_dir = str(
            _SCRIPT_DIR / f"results_{args.benchmark}_baseline_replicate"
        )

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
