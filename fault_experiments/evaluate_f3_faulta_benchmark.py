import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"
PREDICTION_ROOT = ROOT / "runs" / "f3_faulta_benchmark_e50_c8"
VOLUME_XLINE_START = 512
VOLUME_XLINE_END = 768
TOLERANCE_PIXELS = 3


def exact_metrics(prediction, label, validity):
    pred = prediction & validity
    truth = label & validity
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth & validity).sum())
    fn = int(np.logical_and(~pred & validity, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def tolerant_metrics(prediction, label, validity, tolerance):
    pred = prediction & validity
    truth = label & validity
    structure = ndimage.generate_binary_structure(2, 2)
    pred_dilated = ndimage.binary_dilation(pred, structure=structure, iterations=tolerance)
    truth_dilated = ndimage.binary_dilation(truth, structure=structure, iterations=tolerance)
    matched_pred = int(np.logical_and(pred, truth_dilated).sum())
    matched_truth = int(np.logical_and(truth, pred_dilated).sum())
    precision = matched_pred / max(int(pred.sum()), 1)
    recall = matched_truth / max(int(truth.sum()), 1)
    dice = 2 * precision * recall / max(precision + recall, 1e-8)

    valid_times = np.where(truth.any(axis=0))[0]
    detected_times = sum(bool(pred_dilated[:, time].any() and truth[:, time].any()) for time in valid_times)
    continuity_recall = detected_times / max(len(valid_times), 1)

    if pred.any() and truth.any():
        distance_to_truth = ndimage.distance_transform_edt(~truth)
        distance_to_pred = ndimage.distance_transform_edt(~pred)
        distances = np.concatenate([distance_to_truth[pred], distance_to_pred[truth]])
        mean_surface_distance = float(distances.mean())
        surface_distance_p95 = float(np.percentile(distances, 95))
    else:
        mean_surface_distance = None
        surface_distance_p95 = None
    return {
        "tolerance_pixels": tolerance,
        "tolerant_precision": precision,
        "tolerant_recall": recall,
        "tolerant_dice": dice,
        "time_continuity_recall": continuity_recall,
        "mean_symmetric_surface_distance_pixels": mean_surface_distance,
        "symmetric_surface_distance_p95_pixels": surface_distance_p95,
    }


def evaluate_case(prediction_volume, case_dir):
    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    inline_index = metadata["inline_index"]
    prediction = prediction_volume[inline_index]
    label = np.load(case_dir / "fault_label.npy")[VOLUME_XLINE_START:VOLUME_XLINE_END].astype(bool)
    validity = np.load(case_dir / "validity_mask.npy")[VOLUME_XLINE_START:VOLUME_XLINE_END].astype(bool)
    row = {
        "case_id": metadata["case_id"],
        "stick_id": metadata["stick_id"],
        "inline_coordinate": metadata["inline_coordinate"],
        "valid_voxels": int(validity.sum()),
        "fault_voxels": int(np.logical_and(label, validity).sum()),
        "predicted_voxels_in_corridor": int(np.logical_and(prediction, validity).sum()),
    }
    row.update(exact_metrics(prediction, label, validity))
    row.update(tolerant_metrics(prediction, label, validity, TOLERANCE_PIXELS))
    return row


def aggregate(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    macro_fields = [
        "dice", "iou", "tolerant_precision", "tolerant_recall", "tolerant_dice",
        "time_continuity_recall", "mean_symmetric_surface_distance_pixels",
        "symmetric_surface_distance_p95_pixels",
    ]
    result = {
        "case_count": len(rows),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "micro_iou": tp / max(tp + fp + fn, 1),
    }
    for field in macro_fields:
        values = [row[field] for row in rows if row[field] is not None]
        result[f"macro_{field}"] = float(np.mean(values)) if values else None
    return result


def main():
    all_rows = []
    summaries = []
    test_cases = sorted((BENCHMARK_ROOT / "sticks" / "test").glob("faulta_stick_*"))
    for model_dir in sorted(path for path in PREDICTION_ROOT.iterdir() if path.is_dir()):
        prediction = np.load(model_dir / "fault_binary.npy", mmap_mode="r").astype(bool)
        rows = []
        for case_dir in test_cases:
            row = evaluate_case(prediction, case_dir)
            row["model"] = model_dir.name
            rows.append(row)
            all_rows.append(row)
        summary = {"model": model_dir.name, "split": "test", **aggregate(rows)}
        summaries.append(summary)

    with (PREDICTION_ROOT / "faulta_test_per_stick.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    with (PREDICTION_ROOT / "faulta_test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (PREDICTION_ROOT / "faulta_test_results.json").write_text(
        json.dumps({"summary": summaries, "per_stick": all_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
