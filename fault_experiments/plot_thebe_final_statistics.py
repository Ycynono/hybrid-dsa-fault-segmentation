import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATISTICS = ROOT / "runs/thebe_final_test2_7/statistics"
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "unet": "#C5CAD3",
    "unet_edge": "#464C55",
    "hybrid": "#FFE15B",
    "hybrid_edge": "#736422",
}


def style_axes(ax):
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)


def add_header(fig, title, subtitle):
    fig.text(
        0.08,
        0.975,
        textwrap.fill(title, 82),
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.08,
        0.925,
        textwrap.fill(subtitle, 125),
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def paired_panel(ax, rows, baseline_key, proposed_key, label):
    x = np.arange(len(rows))
    baseline = np.array([float(row[baseline_key]) for row in rows])
    proposed = np.array([float(row[proposed_key]) for row in rows])
    for index in x:
        ax.plot(
            [index - 0.12, index + 0.12],
            [baseline[index], proposed[index]],
            color="#AEB4C0",
            linewidth=1.0,
            zorder=1,
        )
    ax.scatter(
        x - 0.12,
        baseline,
        s=52,
        facecolor=TOKENS["panel"],
        edgecolor=TOKENS["unet_edge"],
        linewidth=1.2,
        label="U-Net",
        zorder=2,
    )
    ax.scatter(
        x + 0.12,
        proposed,
        s=55,
        marker="s",
        facecolor=TOKENS["hybrid"],
        edgecolor=TOKENS["hybrid_edge"],
        linewidth=1.0,
        label="Hybrid DSA",
        zorder=3,
    )
    ax.set_xticks(x, [row["block"].replace("test", "Test ") for row in rows])
    ax.set_ylabel(label, color=TOKENS["ink"])
    style_axes(ax)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics-dir", type=Path, default=DEFAULT_STATISTICS)
    args = parser.parse_args()
    rows = load_rows(args.statistics_dir / "block_metrics.csv")
    summary = json.loads((args.statistics_dir / "summary_statistics.json").read_text(encoding="utf-8"))
    output_dir = args.statistics_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    paired_panel(axes[0], rows, "unet_dice", "hybrid_dice", "Exact Dice")
    paired_panel(
        axes[1],
        rows,
        "unet_tolerant_dice_3px",
        "hybrid_tolerant_dice_3px",
        "3-pixel tolerant Dice",
    )
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=2,
        borderaxespad=0,
    )
    exact = summary["paired_block_statistics"]["dice"]
    tolerant = summary["paired_block_statistics"]["tolerant_dice_3px"]
    axes[0].text(
        0.02,
        0.96,
        f"Mean difference +{exact['paired_mean_difference']:.3f}\n95% CI [{exact['bootstrap_95_ci_difference'][0]:.3f}, {exact['bootstrap_95_ci_difference'][1]:.3f}]",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["ink"],
    )
    axes[1].text(
        0.02,
        0.96,
        f"Mean difference +{tolerant['paired_mean_difference']:.3f}\n95% CI [{tolerant['bootstrap_95_ci_difference'][0]:.3f}, {tolerant['bootstrap_95_ci_difference'][1]:.3f}]",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=TOKENS["ink"],
    )
    add_header(
        fig,
        "Hybrid DSA improves fault delineation on every independent Thebe test block",
        "Frozen external test2-test7; thresholds selected on val1-val2 only; lines pair results within each block (n=6).",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.78, wspace=0.25)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"thebe_paired_performance.{extension}", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    metrics = ["Precision", "Recall", "Exact Dice", "Tolerant Dice"]
    unet = [
        summary["aggregate"]["U-Net"]["micro_precision"],
        summary["aggregate"]["U-Net"]["micro_recall"],
        summary["aggregate"]["U-Net"]["micro_dice"],
        summary["aggregate"]["U-Net"]["macro_block_tolerant_dice_3px_mean"],
    ]
    hybrid = [
        summary["aggregate"]["Hybrid DSA"]["micro_precision"],
        summary["aggregate"]["Hybrid DSA"]["micro_recall"],
        summary["aggregate"]["Hybrid DSA"]["micro_dice"],
        summary["aggregate"]["Hybrid DSA"]["macro_block_tolerant_dice_3px_mean"],
    ]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    x = np.arange(len(metrics))
    width = 0.31
    ax.bar(
        x - width / 2,
        unet,
        width,
        color=TOKENS["unet"],
        edgecolor=TOKENS["unet_edge"],
        linewidth=1,
        label="U-Net",
    )
    ax.bar(
        x + width / 2,
        hybrid,
        width,
        color=TOKENS["hybrid"],
        edgecolor=TOKENS["hybrid_edge"],
        linewidth=1,
        label="Hybrid DSA",
    )
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Score", color=TOKENS["ink"])
    ax.set_ylim(0, max(unet + hybrid) * 1.20)
    style_axes(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    for offset, values in [(-width / 2, unet), (width / 2, hybrid)]:
        for index, value in enumerate(values):
            ax.text(index + offset, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=8, color=TOKENS["ink"])
    add_header(
        fig,
        "Hybrid DSA raises both sensitivity and fault-mask agreement",
        "Micro precision, recall, and exact Dice pooled over 603 crosslines; tolerant Dice is the six-block macro mean.",
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.76)
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"thebe_aggregate_metrics.{extension}", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
