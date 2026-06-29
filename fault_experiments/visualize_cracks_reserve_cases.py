import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/cracks_external_v1"
AUDIT_RUN = ROOT / "runs/cracks_audit_frozen"
RESERVE_RUN = ROOT / "runs/cracks_reserve_final"
OUTPUT_DIR = RESERVE_RUN / "figures"
MODELS = [
    ("U-Net", "unet", 0.50),
    ("Hybrid DSA", "hybrid_dsa", 0.15),
    ("SwinUNETR", "swinunetr_f3chain", 0.40),
]
TOKENS = {"surface": "#FCFCFD", "ink": "#1F2430", "muted": "#6F768A"}


def read_metrics(directory):
    path = RESERVE_RUN / directory / "per_section_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["section"]): row for row in csv.DictReader(handle)}


def exact_dice(row):
    tp, fp, fn = (float(row[key]) for key in ("tp", "fp", "fn"))
    return 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)


def select_sections(hybrid_rows):
    ranked = sorted(hybrid_rows, key=lambda section: exact_dice(hybrid_rows[section]))
    positions = [int(round((len(ranked) - 1) * q)) for q in (0.15, 0.50, 0.85)]
    return [ranked[position] for position in positions]


def overlay(ax, seismic, mask, color):
    ax.imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask] = color
    ax.imshow(rgba, aspect="auto")


def main():
    split = json.loads((DATA_DIR / "expert_split.json").read_text(encoding="utf-8"))
    reserve_sections = split["reserve_sections"]
    amplitude = np.load(DATA_DIR / "amplitude_01.npy", mmap_mode="r")
    truth = np.load(
        DATA_DIR / "sealed_reserve/reserve_expert_fault_masks.npy", mmap_mode="r"
    )
    metrics = {name: read_metrics(directory) for name, directory, _ in MODELS}
    probabilities = {
        name: np.load(AUDIT_RUN / directory / "fault_probability_float16.npy", mmap_mode="r")
        for name, directory, _ in MODELS
    }
    sections = select_sections(metrics["Hybrid DSA"])

    fig, axes = plt.subplots(
        len(sections), 5, figsize=(15.2, 11.2), dpi=220, sharex=True, sharey=True
    )
    fig.patch.set_facecolor(TOKENS["surface"])
    column_titles = ["Seismic amplitude", "Independent expert", *[m[0] for m in MODELS]]
    for ax, title in zip(axes[0], column_titles):
        ax.set_title(title, fontsize=10, color=TOKENS["ink"], pad=7)

    selection_rows = []
    difficulty = ["Lower-performing", "Median", "Higher-performing"]
    expert_rgba = np.array([0.74, 0.34, 0.61, 0.90], dtype=np.float32)
    prediction_rgba = np.array([0.94, 0.44, 0.22, 0.78], dtype=np.float32)
    for row_index, (section, level) in enumerate(zip(sections, difficulty)):
        seismic = np.asarray(amplitude[section - 1]).T
        expert = np.asarray(truth[reserve_sections.index(section)]).T
        axes[row_index, 0].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        overlay(axes[row_index, 1], seismic, expert, expert_rgba)
        for column, (name, _, threshold) in enumerate(MODELS, start=2):
            prediction = np.asarray(probabilities[name][section - 1]).T >= threshold
            overlay(axes[row_index, column], seismic, prediction, prediction_rgba)
            row = metrics[name][section]
            axes[row_index, column].text(
                0.02,
                0.02,
                f"D={exact_dice(row):.3f}  TD={float(row['tolerant_dice']):.3f}",
                transform=axes[row_index, column].transAxes,
                ha="left",
                va="bottom",
                fontsize=7.4,
                color="white",
                bbox={"facecolor": "#1F2430", "alpha": 0.78, "pad": 2, "edgecolor": "none"},
            )
        axes[row_index, 0].set_ylabel(
            f"{level}\nsection {section}\nDepth sample", fontsize=8.5, color=TOKENS["ink"]
        )
        selection_rows.append(
            {
                "section": section,
                "selection_role": level,
                "hybrid_exact_dice": exact_dice(metrics["Hybrid DSA"][section]),
            }
        )

    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax in axes[-1]:
        ax.set_xlabel("Trace index", fontsize=8.5, color=TOKENS["ink"])

    fig.text(
        0.055,
        0.985,
        "Frozen models show both robust detections and persistent field failure modes",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.956,
        "CRACKS sealed reserve; cases fixed at the 15th, 50th, and 85th percentiles of Hybrid exact Dice. "
        "Pink = expert, orange = prediction; D = exact Dice, TD = 3-pixel tolerant Dice.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.035, top=0.91, wspace=0.035, hspace=0.08)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / "cracks_reserve_three_case_comparison.png"
    svg = OUTPUT_DIR / "cracks_reserve_three_case_comparison.svg"
    fig.savefig(png, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    (OUTPUT_DIR / "three_case_selection.json").write_text(
        json.dumps(
            {
                "selection_policy": "15th, 50th, and 85th percentile of Hybrid DSA exact Dice on the sealed reserve",
                "sections": selection_rows,
                "thresholds": {name: threshold for name, _, threshold in MODELS},
                "labels_opened_after_protocol_lock": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(png)


if __name__ == "__main__":
    main()
