import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fault_experiments.infer_real_volume import blending_window, window_starts


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("U-Net", "unet3d", 0.50),
    ("Hybrid DSA", "dsa_hybrid_replay", 0.15),
]


def reconstruction_weights(shape, stride=64):
    pad_before = (128 - shape[0]) // 2
    pad_after = 128 - shape[0] - pad_before
    window = blending_window((128, 128, 128))[pad_before : 128 - pad_after]
    crossline_weight = window[:, 0, 0]
    planar = window[0] / max(float(crossline_weight[0]), 1e-8)
    weight_2d = np.zeros(shape[1:], dtype=np.float32)
    for inline_start in window_starts(shape[1], 128, stride):
        for sample_start in window_starts(shape[2], 128, stride):
            weight_2d[inline_start : inline_start + 128, sample_start : sample_start + 128] += planar
    return crossline_weight, weight_2d


def read_metrics(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_sections(label, model_metrics):
    fractions = np.asarray(label, dtype=np.uint8).mean(axis=(1, 2))
    nonempty = np.flatnonzero(fractions > 0)
    median_fraction = np.median(fractions[nonempty])
    representative = int(nonempty[np.argmin(np.abs(fractions[nonempty] - median_fraction))])
    unet = np.array([float(row["tolerant_dice"]) for row in model_metrics["U-Net"]])
    hybrid = np.array([float(row["tolerant_dice"]) for row in model_metrics["Hybrid DSA"]])
    valid = nonempty
    failure = int(valid[np.argmin(hybrid[valid])])
    largest_gain = int(valid[np.argmax((hybrid - unet)[valid])])
    return {
        "representative_median_label_density": representative,
        "hybrid_lowest_tolerant_dice": failure,
        "largest_hybrid_minus_unet_gain": largest_gain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="test4")
    parser.add_argument("--data-root", type=Path, default=ROOT / "processed_data/thebe_official/test")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/thebe_final_test2_7")
    args = parser.parse_args()
    data_dir = args.data_root / args.block
    amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
    crossline_weight, weight_2d = reconstruction_weights(amplitude.shape)
    metrics = {
        name: read_metrics(args.run_root / directory / args.block / "per_crossline_metrics.csv")
        for name, directory, _ in MODELS
    }
    selected = select_sections(label, metrics)
    output_dir = args.run_root / "statistics/figures/qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_sums = {
        name: np.load(args.run_root / directory / args.block / "probability_sum.npy", mmap_mode="r")
        for name, directory, _ in MODELS
    }

    selection_rows = []
    for reason, index in selected.items():
        amp = (np.asarray(amplitude[index], dtype=np.float32) + 1.0) / 2.0
        truth = np.asarray(label[index], dtype=bool)
        fig, axes = plt.subplots(1, 4, figsize=(15, 7), dpi=180, sharex=True, sharey=True)
        fig.patch.set_facecolor("#FCFCFD")
        axes[0].imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[0].set_title("Seismic amplitude")
        axes[1].imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[1].imshow(np.ma.masked_where(~truth.T, truth.T), cmap="spring", alpha=0.80, aspect="auto")
        axes[1].set_title("Expert interpretation")
        row = {"reason": reason, "block": args.block, "crossline": index, "fault_fraction": float(truth.mean())}
        for ax, (name, _, threshold) in zip(axes[2:], MODELS):
            probability = np.asarray(probability_sums[name][index]) / np.maximum(
                crossline_weight[index] * weight_2d, 1e-8
            )
            prediction = probability >= threshold
            ax.imshow(amp.T, cmap="gray", aspect="auto", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(~prediction.T, prediction.T), cmap="autumn", alpha=0.72, aspect="auto")
            score = float(metrics[name][index]["tolerant_dice"])
            row[f"{name}_tolerant_dice"] = score
            ax.set_title(f"{name} (Tol. Dice {score:.3f})")
        for ax in axes:
            ax.set_xlabel("Inline index")
            ax.set_ylabel("Sample index")
        fig.suptitle(
            f"Thebe {args.block}, crossline {index}: {reason.replace('_', ' ')}\nPink = expert label; orange = prediction",
            fontsize=13,
            color="#1F2430",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = output_dir / f"{args.block}_crossline_{index:03d}_{reason}.png"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        row["figure"] = str(path.relative_to(ROOT))
        selection_rows.append(row)
    (output_dir / f"{args.block}_selection.json").write_text(
        json.dumps({"selection_policy": selected, "sections": selection_rows}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(selection_rows, indent=2))


if __name__ == "__main__":
    main()
