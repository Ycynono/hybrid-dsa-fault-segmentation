import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
THEBE = ROOT / "runs/thebe_final_test2_7/statistics/summary_statistics.json"
CRACKS = ROOT / "runs/cracks_final_statistics.json"
OUTPUT = ROOT / "runs/cross_dataset_evidence"
MODELS = ["U-Net", "Hybrid DSA", "SwinUNETR F3-chain"]
COLORS = ["#C5CAD3", "#FFE15B", "#A3BEFA"]
EDGES = ["#464C55", "#736422", "#2E4780"]
TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}


def style(ax):
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(axis="y", color=TOKENS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)


def grouped_bars(ax, values, categories, ylabel):
    x = np.arange(len(categories))
    width = 0.22
    for index, (model, color, edge) in enumerate(zip(MODELS, COLORS, EDGES)):
        positions = x + (index - 1) * width
        bars = ax.bar(
            positions,
            values[model],
            width,
            label=model,
            color=color,
            edgecolor=edge,
            linewidth=1.0,
        )
        for bar, value in zip(bars, values[model]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TOKENS["ink"],
            )
    ax.set_xticks(x, categories)
    ax.set_ylabel(ylabel, color=TOKENS["ink"])
    ax.set_ylim(0, max(max(values[model]) for model in MODELS) * 1.24)
    style(ax)


def main():
    thebe = json.loads(THEBE.read_text(encoding="utf-8"))
    cracks = json.loads(CRACKS.read_text(encoding="utf-8"))
    exact = {
        model: [
            thebe["aggregate"][model]["macro_block_dice_mean"],
            cracks["reserve_primary"]["aggregate"][model]["macro_exact_dice"],
        ]
        for model in MODELS
    }
    tolerant = {
        model: [
            thebe["aggregate"][model]["macro_block_tolerant_dice_3px_mean"],
            cracks["reserve_primary"]["aggregate"][model]["macro_tolerant_dice_3px"],
        ]
        for model in MODELS
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    grouped_bars(axes[0], exact, ["Thebe\n6 blocks", "CRACKS reserve\n20 sections"], "Exact Dice")
    grouped_bars(
        axes[1],
        tolerant,
        ["Thebe\n6 blocks", "CRACKS reserve\n20 sections"],
        "3-pixel tolerant Dice",
    )
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.03),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    fig.text(
        0.07,
        0.98,
        textwrap.fill("Model ranking reverses across expert-labeled field benchmarks", 84),
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.07,
        0.925,
        "Matched synthetic-F3-Thebe training chain; thresholds selected on Thebe val1-val2; reserve opened once after protocol lock.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.76, wspace=0.24)
    OUTPUT.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT / f"cross_dataset_model_ranking.{extension}",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    fig.patch.set_facecolor(TOKENS["surface"])
    for model, color, edge in zip(MODELS, COLORS, EDGES):
        parameters = thebe["aggregate"][model]["parameter_count"]
        ax.scatter(
            parameters,
            exact[model][0],
            s=85,
            marker="o",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.1,
        )
        ax.scatter(
            parameters,
            exact[model][1],
            s=85,
            marker="s",
            facecolor=color,
            edgecolor=edge,
            linewidth=1.1,
        )
        ax.text(parameters * 1.08, max(exact[model]) + 0.004, model, fontsize=8.5, color=TOKENS["ink"])
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)", color=TOKENS["ink"])
    ax.set_ylabel("Exact Dice", color=TOKENS["ink"])
    style(ax)
    ax.scatter([], [], marker="o", color="#7A828F", label="Thebe")
    ax.scatter([], [], marker="s", color="#7A828F", label="CRACKS reserve")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=2, borderaxespad=0)
    fig.text(
        0.10,
        0.98,
        "Accuracy and compactness expose different operating regimes",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.10,
        0.925,
        "Circles: Thebe six-block macro Dice. Squares: CRACKS 20-section reserve macro Dice.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.14, top=0.76)
    for extension in ("png", "svg"):
        fig.savefig(
            OUTPUT / f"accuracy_parameter_tradeoff.{extension}",
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
