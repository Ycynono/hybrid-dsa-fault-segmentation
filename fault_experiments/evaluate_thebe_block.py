import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from scipy import ndimage

from fault_experiments.infer_real_volume import blending_window, load_model, window_starts


def evaluate_section(prediction, truth, tolerance=3):
    prediction = prediction.astype(bool, copy=False)
    truth = truth.astype(bool, copy=False)
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    truth_dilated = ndimage.maximum_filter(truth, size=2 * tolerance + 1)
    prediction_dilated = ndimage.maximum_filter(prediction, size=2 * tolerance + 1)
    matched_prediction = int(np.logical_and(prediction, truth_dilated).sum())
    matched_truth = int(np.logical_and(truth, prediction_dilated).sum())
    tolerant_precision = matched_prediction / max(int(prediction.sum()), 1)
    tolerant_recall = matched_truth / max(int(truth.sum()), 1)
    tolerant_dice = 2 * tolerant_precision * tolerant_recall / max(
        tolerant_precision + tolerant_recall, 1e-8
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tolerant_precision": tolerant_precision,
        "tolerant_recall": tolerant_recall,
        "tolerant_dice": tolerant_dice,
    }


def main():
    parser = argparse.ArgumentParser(description="Fixed-threshold sliding-window evaluation on Thebe.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--threshold-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
    if amplitude.shape != label.shape or amplitude.shape[0] > 128:
        raise ValueError(f"Unsupported Thebe block shape: {amplitude.shape}, label={label.shape}")

    pad_before = (128 - amplitude.shape[0]) // 2
    pad_after = 128 - amplitude.shape[0] - pad_before
    inline_starts = window_starts(amplitude.shape[1], 128, args.stride)
    sample_starts = window_starts(amplitude.shape[2], 128, args.stride)
    patch_count = len(inline_starts) * len(sample_starts)
    window = blending_window((128, 128, 128))[pad_before : 128 - pad_after]
    crossline_weight = window[:, 0, 0].copy()
    planar_window = window[0] / max(float(crossline_weight[0]), 1e-8)
    weight_2d = np.zeros(amplitude.shape[1:], dtype=np.float32)
    probability_sum = open_memmap(
        output_dir / "probability_sum.npy", mode="w+", dtype=np.float32, shape=amplitude.shape
    )
    probability_sum[:] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(Path(args.checkpoint), device)
    started = time.perf_counter()
    completed = 0
    with torch.inference_mode():
        for inline_start in inline_starts:
            for sample_start in sample_starts:
                raw = np.asarray(
                    amplitude[
                        :,
                        inline_start : inline_start + 128,
                        sample_start : sample_start + 128,
                    ],
                    dtype=np.float32,
                )
                patch = np.pad(raw, ((pad_before, pad_after), (0, 0), (0, 0)), mode="reflect")
                tensor = torch.from_numpy(np.ascontiguousarray(patch[None, None])).to(device)
                probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                probability = probability[pad_before : 128 - pad_after]
                probability_sum[
                    :,
                    inline_start : inline_start + 128,
                    sample_start : sample_start + 128,
                ] += probability * window
                weight_2d[
                    inline_start : inline_start + 128,
                    sample_start : sample_start + 128,
                ] += planar_window
                completed += 1
                if completed % 25 == 0 or completed == patch_count:
                    probability_sum.flush()
                    print(f"patch {completed}/{patch_count}", flush=True)

    runtime = time.perf_counter() - started
    section_rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for crossline in range(amplitude.shape[0]):
        denominator = crossline_weight[crossline] * weight_2d
        probability = np.asarray(probability_sum[crossline]) / np.maximum(denominator, 1e-8)
        prediction = probability >= args.threshold
        metrics = evaluate_section(prediction, np.asarray(label[crossline]), tolerance=3)
        row = {"crossline_local_index": crossline, **metrics}
        section_rows.append(row)
        for key in totals:
            totals[key] += metrics[key]

    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    summary = {
        "model": checkpoint.get("args", {}).get("model", "unknown"),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "threshold_source": args.threshold_source,
        "thebe_labels_used_for_training_or_calibration": False,
        "shape": list(amplitude.shape),
        "patch_count": patch_count,
        "runtime_seconds": runtime,
        "precision": precision,
        "recall": recall,
        "dice": 2 * totals["tp"] / max(2 * totals["tp"] + totals["fp"] + totals["fn"], 1),
        "iou": totals["tp"] / max(totals["tp"] + totals["fp"] + totals["fn"], 1),
        "macro_tolerant_precision_3px": float(np.mean([r["tolerant_precision"] for r in section_rows])),
        "macro_tolerant_recall_3px": float(np.mean([r["tolerant_recall"] for r in section_rows])),
        "macro_tolerant_dice_3px": float(np.mean([r["tolerant_dice"] for r in section_rows])),
    }
    with (output_dir / "per_crossline_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(section_rows[0]))
        writer.writeheader()
        writer.writerows(section_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
