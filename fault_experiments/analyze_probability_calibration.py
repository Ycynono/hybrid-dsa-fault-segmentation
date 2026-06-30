from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fault_experiments.visualize_thebe_3d import reconstruction_weights


ROOT = Path(__file__).resolve().parents[1]
THEBE_DATA = ROOT / "processed_data/thebe_official/test"
THEBE_RUN = ROOT / "runs/thebe_final_test2_7"
CRACKS_DATA = ROOT / "processed_data/cracks_external_v1/sealed_reserve"
CRACKS_RUN = ROOT / "runs/cracks_audit_frozen"
PROTOCOL = ROOT / "FINAL_PROTOCOL_LOCK.json"
OUTPUT = ROOT / "runs/probability_calibration"

MODELS = [
    ("U-Net", "unet3d", "unet", 0.50, "#7B91C8"),
    ("Hybrid DSA", "dsa_hybrid_replay", "hybrid_dsa", 0.15, "#D6AD16"),
    ("SwinUNETR", "swin_unetr_f3chain", "swinunetr_f3chain", 0.40, "#60A879"),
]


class HistogramAccumulator:
    def __init__(self, bins: int, calibration_bins: int):
        self.bins = bins
        self.calibration_bins = calibration_bins
        self.total_hist = np.zeros(bins, dtype=np.int64)
        self.positive_hist = np.zeros(bins, dtype=np.int64)
        self.calibration_count = np.zeros(calibration_bins, dtype=np.int64)
        self.calibration_probability_sum = np.zeros(calibration_bins, dtype=np.float64)
        self.calibration_positive = np.zeros(calibration_bins, dtype=np.int64)
        self.brier_sum = 0.0
        self.count = 0

    def update(self, probability: np.ndarray, truth: np.ndarray) -> None:
        probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        truth = np.asarray(truth, dtype=bool)
        flat_probability = probability.ravel()
        flat_truth = truth.ravel()

        indices = np.minimum((flat_probability * self.bins).astype(np.int32), self.bins - 1)
        total = np.bincount(indices, minlength=self.bins)
        positive = np.bincount(indices, weights=flat_truth, minlength=self.bins)
        self.total_hist += total.astype(np.int64, copy=False)
        self.positive_hist += np.rint(positive).astype(np.int64, copy=False)

        calibration_index = np.minimum(
            (flat_probability * self.calibration_bins).astype(np.int16),
            self.calibration_bins - 1,
        )
        self.calibration_count += np.bincount(
            calibration_index, minlength=self.calibration_bins
        ).astype(np.int64, copy=False)
        self.calibration_probability_sum += np.bincount(
            calibration_index,
            weights=flat_probability,
            minlength=self.calibration_bins,
        )
        self.calibration_positive += np.rint(
            np.bincount(
                calibration_index,
                weights=flat_truth,
                minlength=self.calibration_bins,
            )
        ).astype(np.int64, copy=False)
        residual = flat_probability - flat_truth.astype(np.float32)
        self.brier_sum += float(np.sum(residual * residual, dtype=np.float64))
        self.count += int(flat_probability.size)

    def merge(self, other: "HistogramAccumulator") -> None:
        self.total_hist += other.total_hist
        self.positive_hist += other.positive_hist
        self.calibration_count += other.calibration_count
        self.calibration_probability_sum += other.calibration_probability_sum
        self.calibration_positive += other.calibration_positive
        self.brier_sum += other.brier_sum
        self.count += other.count

    def summarize(self, frozen_threshold: float) -> tuple[dict, list[dict], list[dict]]:
        positives = int(self.positive_hist.sum())
        negatives = int(self.total_hist.sum() - positives)
        tp = np.cumsum(self.positive_hist[::-1])
        fp = np.cumsum((self.total_hist - self.positive_hist)[::-1])
        recall = tp / max(positives, 1)
        precision = tp / np.maximum(tp + fp, 1)
        recall_increment = np.diff(np.concatenate(([0.0], recall)))
        auprc = float(np.sum(recall_increment * precision))

        thresholds = (np.arange(self.bins - 1, -1, -1) / self.bins).astype(np.float64)
        fn = positives - tp
        dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        best_index = int(np.argmax(dice))
        frozen_index = int(np.argmin(np.abs(thresholds - frozen_threshold)))

        calibration_rows = []
        ece = 0.0
        for index in range(self.calibration_bins):
            count = int(self.calibration_count[index])
            if count:
                mean_probability = float(self.calibration_probability_sum[index] / count)
                observed_fraction = float(self.calibration_positive[index] / count)
                ece += count / max(self.count, 1) * abs(mean_probability - observed_fraction)
            else:
                mean_probability = None
                observed_fraction = None
            calibration_rows.append(
                {
                    "bin": index,
                    "lower": index / self.calibration_bins,
                    "upper": (index + 1) / self.calibration_bins,
                    "count": count,
                    "mean_probability": mean_probability,
                    "observed_positive_fraction": observed_fraction,
                }
            )

        stride = max(self.bins // 100, 1)
        curve_rows = []
        for index in range(0, self.bins, stride):
            curve_rows.append(
                {
                    "threshold": float(thresholds[index]),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "dice": float(dice[index]),
                    "tp": int(tp[index]),
                    "fp": int(fp[index]),
                    "fn": int(fn[index]),
                }
            )
        summary = {
            "voxel_count": self.count,
            "positive_voxels": positives,
            "negative_voxels": negatives,
            "prevalence": positives / max(self.count, 1),
            "auprc_histogram": auprc,
            "brier_score": self.brier_sum / max(self.count, 1),
            "ece": float(ece),
            "calibration_bins": self.calibration_bins,
            "probability_histogram_bins": self.bins,
            "best_histogram_threshold": float(thresholds[best_index]),
            "best_histogram_dice": float(dice[best_index]),
            "frozen_threshold": frozen_threshold,
            "frozen_threshold_histogram_dice": float(dice[frozen_index]),
            "curve_is_histogram_approximation": True,
        }
        return summary, curve_rows, calibration_rows


def analyze_thebe(model_directory: str, bins: int, calibration_bins: int):
    aggregate = HistogramAccumulator(bins, calibration_bins)
    blocks = {}
    for block in [f"test{index}" for index in range(2, 8)]:
        data_dir = THEBE_DATA / block
        label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
        probability_sum = np.load(
            THEBE_RUN / model_directory / block / "probability_sum.npy", mmap_mode="r"
        )
        crossline_weight, weight_2d = reconstruction_weights(label.shape)
        block_accumulator = HistogramAccumulator(bins, calibration_bins)
        for crossline in range(label.shape[0]):
            denominator = np.maximum(crossline_weight[crossline] * weight_2d, 1e-8)
            probability = np.asarray(probability_sum[crossline], dtype=np.float32) / denominator
            block_accumulator.update(probability, np.asarray(label[crossline]))
        aggregate.merge(block_accumulator)
        blocks[block] = block_accumulator
        print(f"Thebe {model_directory} {block} complete", flush=True)
    return aggregate, blocks


def analyze_cracks(model_directory: str, sections: list[int], bins: int, calibration_bins: int):
    probability = np.load(
        CRACKS_RUN / model_directory / "fault_probability_float16.npy", mmap_mode="r"
    )
    truth = np.load(CRACKS_DATA / "reserve_expert_fault_masks.npy", mmap_mode="r")
    accumulator = HistogramAccumulator(bins, calibration_bins)
    for position, section in enumerate(sections):
        accumulator.update(
            np.asarray(probability[section - 1], dtype=np.float32),
            np.asarray(truth[position]),
        )
    return accumulator


def plot_curves(results: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for column, dataset in enumerate(("Thebe", "CRACKS reserve")):
        pr_axis = axes[0, column]
        threshold_axis = axes[1, column]
        for name, _, _, frozen_threshold, color in MODELS:
            rows = results[dataset][name]["curve"]
            recall = [row["recall"] for row in rows]
            precision = [row["precision"] for row in rows]
            thresholds = [row["threshold"] for row in rows]
            dice = [row["dice"] for row in rows]
            auprc = results[dataset][name]["summary"]["auprc_histogram"]
            pr_axis.plot(recall, precision, color=color, linewidth=2, label=f"{name} (AUPRC={auprc:.3f})")
            threshold_axis.plot(thresholds, dice, color=color, linewidth=2, label=name)
            threshold_axis.axvline(frozen_threshold, color=color, linewidth=1, linestyle=":", alpha=0.8)
        pr_axis.set_title(f"{dataset}: precision-recall")
        pr_axis.set_xlabel("Recall")
        pr_axis.set_ylabel("Precision")
        pr_axis.set_xlim(0, 1)
        pr_axis.set_ylim(bottom=0)
        pr_axis.grid(alpha=0.2)
        pr_axis.legend(frameon=False, fontsize=8)
        threshold_axis.set_title(f"{dataset}: Dice versus threshold")
        threshold_axis.set_xlabel("Probability threshold")
        threshold_axis.set_ylabel("Exact Dice")
        threshold_axis.set_xlim(0, 1)
        threshold_axis.set_ylim(bottom=0)
        threshold_axis.grid(alpha=0.2)
        threshold_axis.legend(frameon=False, fontsize=8)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold-free and calibration audit on frozen field predictions.")
    parser.add_argument("--bins", type=int, default=400)
    parser.add_argument("--calibration-bins", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sections = [int(value) for value in protocol["reserve_sections"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {"Thebe": {}, "CRACKS reserve": {}}
    flat_rows = []
    for name, thebe_directory, cracks_directory, frozen_threshold, _ in MODELS:
        thebe_accumulator, block_accumulators = analyze_thebe(
            thebe_directory, args.bins, args.calibration_bins
        )
        cracks_accumulator = analyze_cracks(
            cracks_directory, sections, args.bins, args.calibration_bins
        )
        for dataset, accumulator in (
            ("Thebe", thebe_accumulator),
            ("CRACKS reserve", cracks_accumulator),
        ):
            summary, curve, calibration = accumulator.summarize(frozen_threshold)
            results[dataset][name] = {
                "summary": summary,
                "curve": curve,
                "calibration": calibration,
            }
            for row in curve:
                flat_rows.append({"dataset": dataset, "model": name, **row})
        results["Thebe"][name]["per_block_summary"] = {
            block: accumulator.summarize(frozen_threshold)[0]
            for block, accumulator in block_accumulators.items()
        }

    with (args.output_dir / "threshold_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "analysis_scope": "Frozen Thebe test2-test7 and sealed CRACKS reserve; no model, checkpoint, or threshold selection",
                "histogram_note": "AUPRC and threshold curves are probability-histogram approximations; fixed-threshold manuscript metrics remain the exact primary results",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_curves(results, args.output_dir / "threshold_and_pr_curves.png")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
