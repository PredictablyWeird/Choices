"""
Build the headline-results figure for §4 (after the section overview).

Three panels, one per finding (avg shift, asymmetry beyond baseline,
backfire of significant effects). Each panel has one bar per
benchmark, with that finding's units on the y-axis. Grouping by
finding rather than by benchmark keeps the y-units honest within a
panel (no pp-vs-% split).

DailyDilemmas's asymmetry-beyond-baseline cell is omitted from the
asymmetry panel because value-level baselines do not have a clean
neutrality interpretation on that benchmark; the panel caption flags
this.

Noise floor is intentionally not in this figure: the per-cell example
in Figure 3 (one_sentence_shift.pdf) shows the noise as an error bar,
which is a cleaner representation than a separate bar here.

Output: ~/code/values/moral-steerability-paper/figures/cross_benchmark.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PAPER_FIG_DIR = Path("~/code/values/moral-steerability-paper/figures").expanduser()

# Numbers from the headline summary (formerly tab:headline in main.tex).
DATA = {
    "Trolley triage": {
        "shift_pp": 15.0,
        "asym_pct": 39.0,
        "backfire_pct": 14.3,
    },
    "BBQ": {
        "shift_pp": 18.0,
        "asym_pct": 34.3,
        "backfire_pct": 10.5,
    },
    "DailyDilemmas": {
        "shift_pp": 9.0,
        "asym_pct": None,  # value-level baselines lack clean neutrality
        "backfire_pct": 5.9,
    },
}

# Wong colorblind-safe palette, matching the rest of the paper's figures.
BENCH_COLORS = {
    "Trolley triage": "#0072B2",  # blue
    "BBQ": "#D55E00",  # vermillion
    "DailyDilemmas": "#CC79A7",  # reddish purple
}

PANELS = [
    {
        "key": "shift_pp",
        "title": "Avg. choice-rate shift\nunder one influence sentence",
        "ylabel": "percentage points",
        "ymax": 24.0,
        "fmt": "{:.0f}pp",
    },
    {
        "key": "asym_pct",
        "title": "Asymmetry beyond baseline",
        "ylabel": "% of baseline-neutral conditions",
        "ymax": 50.0,
        "fmt": "{:.0f}%",
    },
    {
        "key": "backfire_pct",
        "title": "Backfire rate of sig. effects",
        "ylabel": "% of significant effects",
        "ymax": 24.0,
        "fmt": "{:.1f}%",
    },
]


def _render_panel(ax, panel: dict, show_ylabel: bool) -> None:
    benches = list(DATA.keys())
    values = [DATA[b][panel["key"]] for b in benches]
    colors = [BENCH_COLORS[b] for b in benches]

    # Drop benchmarks with None values entirely from the panel (instead of
    # rendering an "n/a" hatch). The panel caption flags omissions.
    plot_x = []
    plot_v = []
    plot_c = []
    plot_lbl = []
    for i, (b, v, c) in enumerate(zip(benches, values, colors)):
        if v is None:
            continue
        plot_x.append(len(plot_x))
        plot_v.append(v)
        plot_c.append(c)
        plot_lbl.append(b)

    ax.bar(plot_x, plot_v, color=plot_c, edgecolor="white", linewidth=0.7, width=0.66)
    for xi, v in zip(plot_x, plot_v):
        ax.text(
            xi,
            v + panel["ymax"] * 0.02,
            panel["fmt"].format(v),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(plot_x)
    ax.set_xticklabels(plot_lbl, fontsize=11)
    ax.set_ylim(0, panel["ymax"])
    ax.set_xlim(-0.6, max(2.6, len(plot_x) - 0.4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    if show_ylabel:
        ax.set_ylabel(panel["ylabel"], fontsize=11)
    else:
        ax.set_ylabel(panel["ylabel"], fontsize=11)


def main() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.4))
    for ax, panel in zip(axes, PANELS):
        _render_panel(ax, panel, show_ylabel=True)
        ax.text(
            0.0,
            1.07,
            panel["title"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    handles = [plt.Rectangle((0, 0), 1, 1, color=BENCH_COLORS[b]) for b in DATA.keys()]
    fig.legend(
        handles,
        list(DATA.keys()),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=12,
        handlelength=1.3,
        columnspacing=2.0,
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.18, wspace=0.45)

    pdf_path = PAPER_FIG_DIR / "cross_benchmark.pdf"
    png_path = PAPER_FIG_DIR / "cross_benchmark.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
