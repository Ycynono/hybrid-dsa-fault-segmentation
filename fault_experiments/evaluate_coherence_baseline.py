from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

from fault_experiments.evaluate_thebe_block import evaluate_section


ROOT = Path(__file__).resolve().parents[1]


def shifted_inline(section: np.ndarray, direction: int) -> np.ndarray:
    shifted = np.empty_like(section)
    if direction < 0:
        shifted[1:] = section[:-1]
        shifted[0] = section[0]
    else:
        shifted[:-1] = section[1:]
        shifted[-1] = section[-1]
    return shifted


def local_discontinuity(
    center: np.ndarray,
    crossline_before: np.ndarray,
    crossline_after: np.ndarray,
    sample_window: int,
    smoothing_sigma: float,
) -> np.ndarray:
    center = np.asarray(center, dtype=np.float32)
    neighbors = (
        np.asarray(crossline_before, dtype=np.float32),
        np.asarray(crossline_after, dtype=np.float32),
        shifted_inline(center, -1),
        shifted_inline(center, 1),
    )
    center_energy = ndimage.uniform_filter1d(
        center * center, size=sample_window, axis=1, mode="nearest"
    )
    similarity = np.zeros_like(center, dtype=np.float32)
    eps = np.float32(1e-6)
    for neighbor in neighbors:
        cross_energy = ndimage.uniform_filter1d(
            center * neighbor, size=sample_window, axis=1, mode="nearest"
        )
        neighbor_energy = ndimage.uniform_filter1d(
            neighbor * neighbor, size=sample_window, axis=1, mode="nearest"
        )
        correlation = cross_energy / np.sqrt(center_energy * neighbor_energy + eps)
        similarity += np.clip(np.abs(correlation), 0.0, 1.0)
    score = 1.0 - similarity / len(neighbors)
    if smoothing_sigma > 0:
        score = ndimage.gaussian_filter(
            score, sigma=(smoothing_sigma, smoothing_sigma), mode="nearest"
        )
    return np.clip(score, 0.0, 1.0).astype(np.float32, copy=False)


def iter_block_scores(amplitude, sample_window, smoothing_sigma):
    for crossline in range(amplitude.shape[0]):
        before = amplitude[max(crossline - 1, 0)]
        center = amplitude[crossline]
        after = amplitude[min(crossline + 1, amplitude.shape[0] - 1)]
        yield crossline, local_discontinuity(
            center, before, after, sample_window, smoothing_sigma
        )


def calibrate(args):
    thresholds = np.array(
        [float(value) for value in args.thresholds.split(",")], dtype=np.float32
    )
    totals = {float(t): {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    started = time.perf_counter()
    block_timings = {}
    for block in ("val1", "val2"):
        block_start = time.perf_counter()
        data_dir = args.data_root / "val" / block
        amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
        label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
        for crossline, score in iter_block_scores(
            amplitude, args.sample_window, args.smoothing_sigma
        ):
            truth = np.asarray(label[crossline], dtype=bool)
            for threshold in thresholds:
                prediction = score >= threshold
                stats = totals[float(threshold)]
                stats["tp"] += int(np.logical_and(prediction, truth).sum())
                stats["fp"] += int(np.logical_and(prediction, ~truth).sum())
                stats["fn"] += int(np.logical_and(~prediction, truth).sum())
        block_timings[block] = time.perf_counter() - block_start

    rows = []
    for threshold in thresholds:
        stats = totals[float(threshold)]
        tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
        rows.append(
            {
                "threshold": float(threshold),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "dice": 2 * tp / max(2 * tp + fp + fn, 1),
                **stats,
            }
        )
    rows.sort(key=lambda row: (row["dice"], row["precision"]), reverse=True)
    selected = rows[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["threshold"]))
    result = {
        "method": "local normalized trace coherence discontinuity",
        "score_definition": "1 - mean absolute local cosine correlation to +/-crossline and +/-inline neighbors",
        "sample_window": args.sample_window,
        "smoothing_sigma_inline_sample": args.smoothing_sigma,
        "selection_data": "Thebe val1-val2 only",
        "selected_threshold": selected["threshold"],
        "selected_validation_metrics": selected,
        "block_runtime_seconds": block_timings,
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


def evaluate_block(args, block, threshold):
    data_dir = args.data_root / "test" / block
    output_dir = args.output_dir / block
    output_dir.mkdir(parents=True, exist_ok=True)
    amplitude = np.load(data_dir / "amplitude_norm.npy", mmap_mode="r")
    label = np.load(data_dir / "fault_label.npy", mmap_mode="r")
    started = time.perf_counter()
    rows = []
    save_planes = block in {"test2", "test4", "test7"}
    if save_planes:
        inline_plane = np.zeros((amplitude.shape[0], amplitude.shape[2]), dtype=np.float16)
        sample_plane = np.zeros((amplitude.shape[0], amplitude.shape[1]), dtype=np.float16)
        crossline_plane = None
    for crossline, score in iter_block_scores(
        amplitude, args.sample_window, args.smoothing_sigma
    ):
        truth = np.asarray(label[crossline], dtype=bool)
        prediction = score >= threshold
        metrics = evaluate_section(prediction, truth, tolerance=3)
        rows.append({"crossline_local_index": crossline, **metrics})
        if save_planes:
            inline_plane[crossline] = score[args.inline_index]
            sample_plane[crossline] = score[:, args.sample_index]
            if crossline == args.crossline_index:
                crossline_plane = score.astype(np.float16)

    totals = {
        key: sum(row[key] for row in rows) for key in ("tp", "fp", "fn")
    }
    summary = {
        "model": "coherence_discontinuity",
        "block": block,
        "threshold": threshold,
        "threshold_source": "Thebe val1-val2 only",
        "sample_window": args.sample_window,
        "smoothing_sigma_inline_sample": args.smoothing_sigma,
        "runtime_seconds": time.perf_counter() - started,
        "precision": totals["tp"] / max(totals["tp"] + totals["fp"], 1),
        "recall": totals["tp"] / max(totals["tp"] + totals["fn"], 1),
        "dice": 2 * totals["tp"]
        / max(2 * totals["tp"] + totals["fp"] + totals["fn"], 1),
        "iou": totals["tp"]
        / max(totals["tp"] + totals["fp"] + totals["fn"], 1),
        "macro_tolerant_precision_3px": float(
            np.mean([row["tolerant_precision"] for row in rows])
        ),
        "macro_tolerant_recall_3px": float(
            np.mean([row["tolerant_recall"] for row in rows])
        ),
        "macro_tolerant_dice_3px": float(
            np.mean([row["tolerant_dice"] for row in rows])
        ),
    }
    with (output_dir / "per_crossline_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if save_planes:
        if crossline_plane is None:
            raise RuntimeError("Requested crossline plane was not generated")
        np.savez_compressed(
            output_dir / "orthogonal_attribute_planes.npz",
            crossline=crossline_plane,
            inline=inline_plane,
            sample=sample_plane,
            crossline_index=args.crossline_index,
            inline_index=args.inline_index,
            sample_index=args.sample_index,
        )
    print(json.dumps(summary, indent=2))
    return summary


def evaluate(args):
    calibration = json.loads(
        (args.output_dir / "calibration.json").read_text(encoding="utf-8")
    )
    threshold = float(calibration["selected_threshold"])
    summaries = [
        evaluate_block(args, f"test{index}", threshold) for index in range(2, 8)
    ]
    aggregate = {
        "method": "coherence_discontinuity",
        "n_blocks": len(summaries),
        "threshold": threshold,
        "threshold_source": "Thebe val1-val2 only",
        "macro_block_exact_dice": float(np.mean([row["dice"] for row in summaries])),
        "macro_block_tolerant_dice_3px": float(
            np.mean([row["macro_tolerant_dice_3px"] for row in summaries])
        ),
        "macro_block_precision": float(
            np.mean([row["precision"] for row in summaries])
        ),
        "macro_block_recall": float(np.mean([row["recall"] for row in summaries])),
        "total_runtime_seconds": float(sum(row["runtime_seconds"] for row in summaries)),
    }
    (args.output_dir / "test2_test7_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("calibrate", "evaluate"))
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "processed_data" / "thebe_official"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs" / "coherence_baseline"
    )
    parser.add_argument("--sample-window", type=int, default=9)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument(
        "--thresholds",
        default="0.02,0.04,0.06,0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.24,0.28,0.32,0.36,0.40",
    )
    parser.add_argument("--crossline-index", type=int, default=50)
    parser.add_argument("--inline-index", type=int, default=900)
    parser.add_argument("--sample-index", type=int, default=900)
    args = parser.parse_args()
    if args.sample_window < 3 or args.sample_window % 2 == 0:
        raise ValueError("sample-window must be an odd integer >= 3")
    if args.mode == "calibrate":
        calibrate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
