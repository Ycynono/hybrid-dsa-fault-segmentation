import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "smeaheia" / "expert_roi_384x512x640"
PREDICTION_ROOT = ROOT / "runs" / "smeaheia_frozen_external"
TOLERANCE_PIXELS = 3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260630


def binary_counts(prediction, truth):
    prediction = prediction.astype(bool, copy=False)
    truth = truth.astype(bool, copy=False)
    return {
        "tp": int(np.logical_and(prediction, truth).sum()),
        "fp": int(np.logical_and(prediction, ~truth).sum()),
        "fn": int(np.logical_and(~prediction, truth).sum()),
    }


def metrics_from_counts(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def tolerant_section_metrics(prediction, truth, tolerance):
    prediction = prediction.astype(bool, copy=False)
    truth = truth.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(2, 2)
    pred_dilated = ndimage.binary_dilation(prediction, structure=structure, iterations=tolerance)
    truth_dilated = ndimage.binary_dilation(truth, structure=structure, iterations=tolerance)
    matched_prediction = int(np.logical_and(prediction, truth_dilated).sum())
    matched_truth = int(np.logical_and(truth, pred_dilated).sum())
    precision = matched_prediction / max(int(prediction.sum()), 1)
    recall = matched_truth / max(int(truth.sum()), 1)
    distances = None
    if prediction.any() and truth.any():
        distance_to_truth = ndimage.distance_transform_edt(~truth)
        distance_to_prediction = ndimage.distance_transform_edt(~prediction)
        distances = np.concatenate([distance_to_truth[prediction], distance_to_prediction[truth]])
    return {
        "matched_prediction": matched_prediction,
        "matched_truth": matched_truth,
        "predicted": int(prediction.sum()),
        "truth": int(truth.sum()),
        "tolerant_precision": precision,
        "tolerant_recall": recall,
        "tolerant_dice": 2 * precision * recall / max(precision + recall, 1e-8),
        "distances": distances,
    }


def calibration_metrics(probability, truth, bins=15):
    probability = probability.astype(np.float64, copy=False)
    truth = truth.astype(np.float64, copy=False)
    brier = float(np.mean((probability - truth) ** 2))
    bin_index = np.minimum((probability * bins).astype(int), bins - 1)
    ece = 0.0
    reliability = []
    for index in range(bins):
        mask = bin_index == index
        count = int(mask.sum())
        if count:
            mean_probability = float(probability[mask].mean())
            positive_fraction = float(truth[mask].mean())
            ece += count / len(probability) * abs(mean_probability - positive_fraction)
        else:
            mean_probability = None
            positive_fraction = None
        reliability.append(
            {
                "bin": index,
                "count": count,
                "mean_probability": mean_probability,
                "positive_fraction": positive_fraction,
            }
        )
    return {"brier": brier, "ece_15_bins": float(ece), "reliability": reliability}


def histogram_auprc(probability, truth, bins=400):
    indices = np.minimum((probability * bins).astype(int), bins - 1)
    positives = np.bincount(indices[truth], minlength=bins).astype(np.float64)
    negatives = np.bincount(indices[~truth], minlength=bins).astype(np.float64)
    tp = np.cumsum(positives[::-1])
    fp = np.cumsum(negatives[::-1])
    recall = tp / max(float(positives.sum()), 1.0)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall_previous = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_previous) * precision))


def evaluate_model(model_dir, label, validity, active_inlines):
    inference = json.loads((model_dir / "inference_metadata.json").read_text(encoding="utf-8"))
    threshold = float(inference["statistics"]["threshold"])
    probability = np.load(model_dir / "fault_probability.npy", mmap_mode="r")
    binary = np.load(model_dir / "fault_binary.npy", mmap_mode="r")
    if probability.shape != label.shape or binary.shape != label.shape:
        raise ValueError(f"Prediction shape mismatch for {model_dir.name}: {probability.shape}")

    valid = validity.astype(bool, copy=False)
    truth_valid = np.asarray(label[valid], dtype=bool)
    probability_valid = np.asarray(probability[valid], dtype=np.float32)
    prediction_valid = probability_valid >= threshold
    pooled = metrics_from_counts(binary_counts(prediction_valid, truth_valid))
    pooled.update(calibration_metrics(probability_valid, truth_valid))
    pooled["histogram_auprc_400_bins"] = histogram_auprc(probability_valid, truth_valid)
    pooled["valid_voxels"] = int(valid.sum())
    pooled["truth_voxels"] = int(truth_valid.sum())
    pooled["predicted_voxels_in_validity"] = int(prediction_valid.sum())
    pooled["predicted_fraction_in_validity"] = float(prediction_valid.mean())

    rows = []
    distance_parts = []
    tolerant_totals = {"matched_prediction": 0, "matched_truth": 0, "predicted": 0, "truth": 0}
    for inline in active_inlines:
        section_valid = np.asarray(validity[inline], dtype=bool)
        section_truth = np.asarray(label[inline], dtype=bool)
        section_prediction = np.logical_and(np.asarray(binary[inline], dtype=bool), section_valid)
        exact = metrics_from_counts(binary_counts(section_prediction[section_valid], section_truth[section_valid]))
        tolerant = tolerant_section_metrics(section_prediction, section_truth, TOLERANCE_PIXELS)
        for key in tolerant_totals:
            tolerant_totals[key] += tolerant[key]
        if tolerant["distances"] is not None:
            distance_parts.append(tolerant["distances"])
        rows.append(
            {
                "model": model_dir.name,
                "inline_local_index": int(inline),
                "valid_voxels": int(section_valid.sum()),
                "truth_voxels": int(section_truth.sum()),
                "predicted_voxels": int(section_prediction.sum()),
                **exact,
                "tolerant_precision_3px": tolerant["tolerant_precision"],
                "tolerant_recall_3px": tolerant["tolerant_recall"],
                "tolerant_dice_3px": tolerant["tolerant_dice"],
            }
        )

    tolerant_precision = tolerant_totals["matched_prediction"] / max(tolerant_totals["predicted"], 1)
    tolerant_recall = tolerant_totals["matched_truth"] / max(tolerant_totals["truth"], 1)
    distances = np.concatenate(distance_parts) if distance_parts else np.array([], dtype=np.float64)
    pooled.update(
        {
            "tolerance_pixels_on_expert_inline_sections": TOLERANCE_PIXELS,
            "tolerant_precision_3px": tolerant_precision,
            "tolerant_recall_3px": tolerant_recall,
            "tolerant_dice_3px": 2
            * tolerant_precision
            * tolerant_recall
            / max(tolerant_precision + tolerant_recall, 1e-8),
            "mean_symmetric_distance_pixels": float(distances.mean()) if distances.size else None,
            "p95_symmetric_distance_pixels": float(np.percentile(distances, 95))
            if distances.size
            else None,
            "macro_section_dice": float(np.mean([row["dice"] for row in rows])),
            "macro_section_tolerant_dice_3px": float(
                np.mean([row["tolerant_dice_3px"] for row in rows])
            ),
            "runtime_seconds": inference["runtime_seconds"],
            "threshold": threshold,
            "threshold_source": inference["threshold_source"],
        }
    )
    return pooled, rows


def paired_bootstrap(rows_by_model):
    names = list(rows_by_model)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            for metric in ("dice", "tolerant_dice_3px"):
                first_values = np.asarray([row[metric] for row in rows_by_model[first]])
                second_values = np.asarray([row[metric] for row in rows_by_model[second]])
                if first_values.shape != second_values.shape:
                    raise ValueError("Paired Smeaheia section vectors do not align.")
                differences = first_values - second_values
                indices = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences)))
                means = differences[indices].mean(axis=1)
                results.append(
                    {
                        "first_model": first,
                        "second_model": second,
                        "metric": metric,
                        "mean_difference_first_minus_second": float(differences.mean()),
                        "within_survey_section_bootstrap_95_interval": np.percentile(
                            means, [2.5, 97.5]
                        ).tolist(),
                        "section_count": len(differences),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "warning": (
                            "Sections are repeated units from one survey, not independent survey clusters."
                        ),
                    }
                )
    return results


def main():
    metadata = json.loads((BENCHMARK_ROOT / "metadata.json").read_text(encoding="utf-8"))
    label = np.load(BENCHMARK_ROOT / "fault_label.npy", mmap_mode="r").astype(bool)
    validity = np.load(BENCHMARK_ROOT / "validity_mask.npy", mmap_mode="r").astype(bool)
    active_inlines = np.flatnonzero(label.any(axis=(1, 2)))
    summaries = []
    all_rows = []
    rows_by_model = {}
    model_dirs = sorted(
        path
        for path in PREDICTION_ROOT.iterdir()
        if path.is_dir() and (path / "inference_metadata.json").exists()
    )
    for model_dir in model_dirs:
        summary, rows = evaluate_model(model_dir, label, validity, active_inlines)
        summary = {"model": model_dir.name, **summary}
        summaries.append(summary)
        all_rows.extend(rows)
        rows_by_model[model_dir.name] = rows

    if not summaries:
        raise RuntimeError("No completed Smeaheia predictions were found.")
    with (PREDICTION_ROOT / "per_expert_inline_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    compact_fields = [
        "model",
        "threshold",
        "precision",
        "recall",
        "dice",
        "iou",
        "tolerant_precision_3px",
        "tolerant_recall_3px",
        "tolerant_dice_3px",
        "histogram_auprc_400_bins",
        "brier",
        "ece_15_bins",
        "mean_symmetric_distance_pixels",
        "p95_symmetric_distance_pixels",
        "runtime_seconds",
    ]
    with (PREDICTION_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact_fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field) for field in compact_fields} for row in summaries])
    result = {
        "dataset": metadata["id"],
        "data_role": metadata["data_role"],
        "evaluation_policy": metadata["evaluation_policy"],
        "expert_active_inline_sections": int(len(active_inlines)),
        "summaries": summaries,
        "paired_within_survey_section_bootstrap": paired_bootstrap(rows_by_model),
        "claim_boundary": (
            "This is one independent field survey with sparse expert sticks. It strengthens external "
            "3D validation but is not dense exhaustive truth or a survey-population estimate."
        ),
    }
    (PREDICTION_ROOT / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps([{key: row.get(key) for key in compact_fields} for row in summaries], indent=2))


if __name__ == "__main__":
    main()
