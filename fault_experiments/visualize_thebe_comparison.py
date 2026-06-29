import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fault_experiments.infer_real_volume import blending_window, window_starts


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data" / "thebe_test1"
RUN_ROOT = ROOT / "runs" / "thebe_test1_external"
MODELS = [
    ("U-Net", RUN_ROOT / "unet3d", 0.55),
    ("Hybrid DSA", RUN_ROOT / "dsa_hybrid_replay", 0.65),
]


def weights(shape, stride=64):
    pad_before = (128 - shape[0]) // 2
    pad_after = 128 - shape[0] - pad_before
    window = blending_window((128, 128, 128))[pad_before : 128 - pad_after]
    crossline_weight = window[:, 0, 0]
    planar = window[0] / max(float(crossline_weight[0]), 1e-8)
    weight_2d = np.zeros(shape[1:], dtype=np.float32)
    for i in window_starts(shape[1], 128, stride):
        for t in window_starts(shape[2], 128, stride):
            weight_2d[i : i + 128, t : t + 128] += planar
    return crossline_weight, weight_2d


def main():
    amplitude = np.load(DATA_DIR / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(DATA_DIR / "fault_label.npy", mmap_mode="r")
    crossline_weight, weight_2d = weights(amplitude.shape)
    output_dir = RUN_ROOT / "comparison_qc"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {name: json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) for name, run_dir, _ in MODELS}

    section_scores = []
    for index in range(amplitude.shape[0]):
        row = {"index": index, "fault_fraction": float(label[index].mean())}
        for name, run_dir, _ in MODELS:
            metrics = np.genfromtxt(
                run_dir / "per_crossline_metrics.csv", delimiter=",", names=True, dtype=None, encoding="utf-8"
            )
            row[name] = float(metrics["tolerant_dice"][index])
        section_scores.append(row)
    selected = [
        max(section_scores, key=lambda row: row["fault_fraction"])["index"],
        max(section_scores, key=lambda row: row["U-Net"])["index"],
        max(section_scores, key=lambda row: row["Hybrid DSA"])["index"],
    ]

    for index in sorted(set(selected)):
        amp = (np.asarray(amplitude[index], dtype=np.float32) + 1.0) / 2.0
        truth = np.asarray(label[index], dtype=bool)
        fig, axes = plt.subplots(1, 4, figsize=(15, 7), dpi=170)
        axes[0].imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[0].set_title("Thebe amplitude")
        axes[1].imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[1].imshow(np.ma.masked_where(~truth.T, truth.T), cmap="spring", aspect="auto")
        axes[1].set_title("Expert fault label")
        for ax, (name, run_dir, threshold) in zip(axes[2:], MODELS):
            probability_sum = np.load(run_dir / "probability_sum.npy", mmap_mode="r")
            probability = np.asarray(probability_sum[index]) / np.maximum(
                crossline_weight[index] * weight_2d, 1e-8
            )
            prediction = probability >= threshold
            ax.imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~prediction.T, prediction.T), cmap="autumn", alpha=0.72, aspect="auto")
            ax.set_title(f"{name}, p >= {threshold:.2f}")
        for ax in axes:
            ax.set_xlabel("Inline")
            ax.set_ylabel("Sample")
        fig.suptitle(f"Thebe test1 crossline {index} | pink=expert, orange=prediction")
        fig.tight_layout()
        fig.savefig(output_dir / f"thebe_test1_crossline_{index:03d}.png", bbox_inches="tight")
        plt.close(fig)
    (output_dir / "selected_sections.json").write_text(
        json.dumps({"selected_crosslines": sorted(set(selected)), "summaries": summaries}, indent=2),
        encoding="utf-8",
    )
    print("Wrote", output_dir)


if __name__ == "__main__":
    main()
