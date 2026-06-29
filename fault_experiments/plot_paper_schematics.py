from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_figures"

COLORS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#D7DBE7",
    "blue_xlight": "#EAF1FE",
    "blue_base": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold_xlight": "#FFF4C2",
    "gold_base": "#FFE15B",
    "gold_dark": "#736422",
    "orange_xlight": "#FFEDDE",
    "orange_dark": "#804126",
    "neutral": "#F4F5F7",
    "neutral_mid": "#7A828F",
}


def setup():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.facecolor": COLORS["surface"],
            "figure.facecolor": COLORS["surface"],
            "svg.fonttype": "none",
        }
    )


def box(ax, x, y, w, h, title, detail, *, fill, edge, title_size=9, detail_size=7.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.08",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.25,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.27,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=COLORS["muted"],
        linespacing=1.15,
        zorder=4,
    )
    return patch


def arrow(ax, start, end, *, color=None, style="-|>", width=1.15, mutation=10, connection=None, z=2):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=width,
        color=color or COLORS["neutral_mid"],
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def save(fig, stem):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        fig.savefig(OUTPUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor=fig.get_facecolor(), **kwargs)
    plt.close(fig)


def architecture_figure():
    fig, ax = plt.subplots(figsize=(18, 8.2))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.text(
        0.055,
        0.965,
        "Hybrid DSA architecture for 3D seismic fault segmentation",
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.055,
        0.925,
        "Standard convolutions preserve shallow seismic detail; depthwise-separable blocks process deeper features; CBAM3D and ASPP3D refine fault context.",
        ha="left",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    y, w, h = 6.0, 1.75, 1.25
    nodes = {
        "input": (0.35, y + 0.12, 1.25, 1.0),
        "enc1": (2.05, y, w, h),
        "enc2": (4.45, y, w, h),
        "enc3": (6.85, y, w, h),
        "enc4": (9.25, y, w, h),
        "aspp": (11.7, y, 1.9, h),
        "dec3": (14.2, y, w, h),
        "dec2": (16.6, y, w, h),
        "dec1": (19.0, y, w, h),
        "out": (21.45, y + 0.12, 1.55, 1.0),
    }

    box(ax, *nodes["input"], "Input", "1 x 128 x 128 x 128", fill=COLORS["neutral"], edge=COLORS["neutral_mid"])
    box(ax, *nodes["enc1"], "Encoder 1", "Std residual\nCBAM3D | 8 x 128^3", fill=COLORS["blue_xlight"], edge=COLORS["blue_dark"], detail_size=6.7)
    box(ax, *nodes["enc2"], "Encoder 2", "Std residual\nCBAM3D | 16 x 64^3", fill=COLORS["blue_xlight"], edge=COLORS["blue_dark"], detail_size=6.7)
    box(ax, *nodes["enc3"], "Encoder 3", "DS residual\nCBAM3D | 32 x 32^3", fill=COLORS["gold_xlight"], edge=COLORS["gold_dark"], detail_size=6.7)
    box(ax, *nodes["enc4"], "Encoder 4", "DS residual\nCBAM3D | 64 x 16^3", fill=COLORS["gold_xlight"], edge=COLORS["gold_dark"], detail_size=6.7)
    box(ax, *nodes["aspp"], "ASPP3D", "Rates 1, 2, 4, 6 + global\n64 x 16^3", fill=COLORS["gold_base"], edge=COLORS["gold_dark"], detail_size=6.7)
    box(ax, *nodes["dec3"], "Decoder 3", "Up + concat\nDS residual | 32 x 32^3", fill=COLORS["gold_xlight"], edge=COLORS["gold_dark"], detail_size=6.7)
    box(ax, *nodes["dec2"], "Decoder 2", "Up + concat\nStd residual | 16 x 64^3", fill=COLORS["blue_xlight"], edge=COLORS["blue_dark"], detail_size=6.7)
    box(ax, *nodes["dec1"], "Decoder 1", "Up + concat\nStd residual | 8 x 128^3", fill=COLORS["blue_xlight"], edge=COLORS["blue_dark"], detail_size=6.7)
    box(ax, *nodes["out"], "Fault logits", "1 x 128 x 128 x 128", fill=COLORS["neutral"], edge=COLORS["neutral_mid"])

    order = ["input", "enc1", "enc2", "enc3", "enc4", "aspp", "dec3", "dec2", "dec1", "out"]
    for left, right in zip(order[:-1], order[1:]):
        lx, ly, lw, lh = nodes[left]
        rx, ry, rw, rh = nodes[right]
        arrow(ax, (lx + lw, ly + lh / 2), (rx, ry + rh / 2))

    for key in ("enc1", "enc2", "enc3"):
        x, _, bw, _ = nodes[key]
        ax.text(x + bw + 0.30, y - 0.20, "MaxPool 2", ha="center", va="top", fontsize=7, color=COLORS["muted"])
    for key in ("dec3", "dec2", "dec1"):
        x, _, _, _ = nodes[key]
        ax.text(x - 0.30, y - 0.20, "UpConv 2", ha="center", va="top", fontsize=7, color=COLORS["muted"])

    skip_pairs = [("enc1", "dec1", 0.78), ("enc2", "dec2", 0.54), ("enc3", "dec3", 0.30)]
    for source, target, radius in skip_pairs:
        sx, sy, sw, sh = nodes[source]
        tx, ty, tw, th = nodes[target]
        arrow(
            ax,
            (sx + sw / 2, sy + sh),
            (tx + tw / 2, ty + th),
            color=COLORS["blue_dark"],
            style="-|>",
            width=1.05,
            connection=f"arc3,rad=-{radius / 4}",
            z=1,
        )
    ax.text(11.9, 8.25, "Encoder-to-decoder skip connections", ha="center", fontsize=8.5, color=COLORS["blue_dark"])

    # Detailed module insets.
    ax.text(0.45, 3.9, "Block internals", fontsize=11, fontweight="bold", color=COLORS["ink"])
    inset_y, inset_h = 1.15, 2.25
    box(ax, 0.45, inset_y, 6.55, inset_h, "Residual feature block", "", fill=COLORS["panel"], edge=COLORS["grid"], title_size=10, detail_size=8)
    small_y = 1.42
    for i, (label, fill, edge) in enumerate(
        [
            ("Conv - Norm - ReLU", COLORS["blue_xlight"], COLORS["blue_dark"]),
            ("Conv - Norm", COLORS["gold_xlight"], COLORS["gold_dark"]),
            ("Residual add", COLORS["neutral"], COLORS["neutral_mid"]),
            ("CBAM3D", COLORS["gold_base"], COLORS["gold_dark"]),
        ]
    ):
        x = 0.75 + i * 1.48
        box(ax, x, small_y, 1.2, 0.72, label, "", fill=fill, edge=edge, title_size=7.3, detail_size=1)
        if i < 3:
            arrow(ax, (x + 1.2, small_y + 0.36), (x + 1.48, small_y + 0.36), mutation=8, width=0.9)

    box(ax, 7.45, inset_y, 6.5, inset_h, "CBAM3D", "", fill=COLORS["panel"], edge=COLORS["grid"], title_size=10, detail_size=8)
    for i, label in enumerate(("Avg/Max pool", "Shared MLP", "Channel weight", "7x7x7 spatial", "Feature weight")):
        x = 7.72 + i * 1.18
        fill = COLORS["blue_xlight"] if i < 3 else COLORS["gold_xlight"]
        edge = COLORS["blue_dark"] if i < 3 else COLORS["gold_dark"]
        box(ax, x, small_y, 0.95, 0.72, label, "", fill=fill, edge=edge, title_size=6.6, detail_size=1)
        if i < 4:
            arrow(ax, (x + 0.95, small_y + 0.36), (x + 1.18, small_y + 0.36), mutation=8, width=0.9)

    box(ax, 14.4, inset_y, 8.6, inset_h, "ASPP3D bottleneck", "", fill=COLORS["panel"], edge=COLORS["grid"], title_size=10, detail_size=8)
    labels = ("r=1", "r=2", "r=4", "r=6", "Global pool", "Concat", "1x1x1 project")
    for i, label in enumerate(labels):
        x = 14.7 + i * 1.12
        fill = COLORS["gold_xlight"] if i < 5 else COLORS["neutral"]
        edge = COLORS["gold_dark"] if i < 5 else COLORS["neutral_mid"]
        box(ax, x, small_y, 0.9, 0.72, label, "", fill=fill, edge=edge, title_size=6.8, detail_size=1)
        if i < len(labels) - 1:
            arrow(ax, (x + 0.9, small_y + 0.36), (x + 1.12, small_y + 0.36), mutation=8, width=0.9)

    ax.text(0.45, 0.45, "Std: standard 3D convolution    DS: depthwise-separable 3D convolution    CBAM3D: channel-spatial attention", fontsize=8, color=COLORS["muted"])
    save(fig, "figure_02_hybrid_dsa_architecture")


def protocol_figure():
    fig, ax = plt.subplots(figsize=(15.5, 8.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    fig.text(0.06, 0.965, "Matched training and frozen field-evaluation protocol", ha="left", va="top", fontsize=16, fontweight="bold", color=COLORS["ink"])
    fig.text(0.06, 0.925, "All models follow the same data sequence; model and threshold selection stop before all three frozen field branches.", ha="left", va="top", fontsize=9.5, color=COLORS["muted"])

    stages = [
        (0.55, 6.55, 2.35, 1.35, "Synthetic pretraining", "400 labeled 128^3 volumes\n50 epochs"),
        (3.35, 6.55, 2.35, 1.35, "Sparse F3 adaptation", "Train sticks 0-5\nSynthetic replay"),
        (6.15, 6.55, 2.35, 1.35, "Thebe adaptation", "train1-train9\nSynthetic replay"),
        (8.95, 6.55, 2.35, 1.35, "Selection", "Thebe val1-val2 only\nCheckpoint + threshold"),
        (11.75, 6.55, 2.35, 1.35, "Protocol lock", "Models, thresholds, metrics\nReserve membership"),
    ]
    for i, (x, y, w, h, title, detail) in enumerate(stages):
        fill = COLORS["blue_xlight"] if i < 3 else COLORS["gold_xlight"]
        edge = COLORS["blue_dark"] if i < 3 else COLORS["gold_dark"]
        box(ax, x, y, w, h, title, detail, fill=fill, edge=edge, title_size=9.5, detail_size=8)
        if i < len(stages) - 1:
            nx = stages[i + 1][0]
            arrow(ax, (x + w, y + h / 2), (nx, y + h / 2), color=COLORS["ink"], mutation=11)

    ax.text(5.0, 8.35, "Training and development", ha="center", fontsize=10, fontweight="bold", color=COLORS["blue_dark"])
    ax.text(10.1, 8.35, "Selection", ha="center", fontsize=10, fontweight="bold", color=COLORS["gold_dark"])
    ax.plot([11.45, 11.45], [5.5, 8.2], color=COLORS["ink"], linewidth=1.2, linestyle="--")
    ax.text(11.55, 8.15, "No test-driven tuning", ha="left", va="top", fontsize=8.5, color=COLORS["ink"])

    lock_center = (12.925, 6.55)
    branch_centers = (6.1, 9.85, 13.6)
    for center, bend in zip(branch_centers, (0.12, 0.03, -0.08)):
        arrow(ax, lock_center, (center, 4.75), color=COLORS["neutral_mid"], connection=f"arc3,rad={bend}", mutation=11)

    box(ax, 4.35, 3.15, 3.5, 1.6, "Frozen Thebe test", "test2-test7; six blocks\nExact + tolerant Dice", fill=COLORS["neutral"], edge=COLORS["neutral_mid"], title_size=10, detail_size=8)
    box(ax, 8.1, 3.15, 3.5, 1.6, "Preregistered CRACKS", "20 audit + 20 sealed reserve\nInherited thresholds", fill=COLORS["neutral"], edge=COLORS["neutral_mid"], title_size=10, detail_size=8)
    box(ax, 11.85, 3.15, 3.5, 1.6, "Frozen external stress tests", "FORCE + Delft surveys\nNo expert fault labels", fill=COLORS["neutral"], edge=COLORS["neutral_mid"], title_size=10, detail_size=8)

    for center, color in zip(branch_centers, (COLORS["blue_dark"], COLORS["gold_dark"], COLORS["orange_dark"])):
        arrow(ax, (center, 3.15), (center, 2.1), color=color, mutation=11)
    box(ax, 4.35, 0.65, 3.5, 1.45, "Block-level inference", "Bootstrap CI and Wilcoxon\nExpert-labeled survey", fill=COLORS["blue_xlight"], edge=COLORS["blue_dark"], title_size=9.5, detail_size=7.5)
    box(ax, 8.1, 0.65, 3.5, 1.45, "Section-level inference", "Within-survey consistency\nReserve opened once", fill=COLORS["gold_xlight"], edge=COLORS["gold_dark"], title_size=9.5, detail_size=7.5)
    box(ax, 11.85, 0.65, 3.5, 1.45, "Qualitative transfer audit", "Fixed centers + occupancy\nDelft TFL agreement; no accuracy", fill=COLORS["orange_xlight"], edge=COLORS["orange_dark"], title_size=9.5, detail_size=7.5)

    ax.text(0.65, 4.35, "Compared models", fontsize=10.5, fontweight="bold", color=COLORS["ink"])
    for i, (name, params) in enumerate((("3D U-Net", "350,809"), ("Hybrid DSA", "104,787"), ("SwinUNETR", "4,078,051"))):
        box(ax, 0.65, 3.35 - i * 0.95, 3.25, 0.7, name, f"{params} parameters", fill=COLORS["panel"], edge=COLORS["grid"], title_size=8.5, detail_size=7)

    ax.text(0.65, 0.35, "CRACKS tests annotation-domain transfer within F3; FORCE and Delft are independent surveys but remain qualitative because expert labels are unavailable.", fontsize=8, color=COLORS["muted"])
    save(fig, "figure_01_study_protocol")


def main():
    setup()
    protocol_figure()
    architecture_figure()
    print(f"Wrote paper schematics to {OUTPUT}")


if __name__ == "__main__":
    main()
