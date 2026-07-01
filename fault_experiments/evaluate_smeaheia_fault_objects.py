import csv
import json
from collections import defaultdict

import numpy as np
import segyio
from scipy import ndimage

from fault_experiments.audit_smeaheia_dataset import DEFAULT_STICKS, parse_sticks
from fault_experiments.evaluate_smeaheia_benchmark import (
    BENCHMARK_ROOT,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PREDICTION_ROOT,
    TOLERANCE_PIXELS,
    binary_counts,
    metrics_from_counts,
    tolerant_section_metrics,
)
from fault_experiments.prepare_smeaheia_benchmark import (
    DEFAULT_SEGY,
    LABEL_RADIUS,
    VALIDITY_RADIUS,
    line_voxels,
    map_records,
    resolve_grid_coordinates,
)


def fault_inline_centrelines(records, roi_start, roi_shape):
    start = np.asarray(roi_start, dtype=np.int64)
    end = start + np.asarray(roi_shape, dtype=np.int64)
    grouped_sticks = defaultdict(list)
    for record in records:
        grouped_sticks[(record["fault"], record["stick"])].append(record)

    fault_coordinates = defaultdict(list)
    fault_sticks = defaultdict(set)
    for (fault, stick), group in grouped_sticks.items():
        group.sort(key=lambda item: item["line_number"])
        points = np.asarray(
            [[item["inline_index"], item["crossline_index"], item["sample_index"]] for item in group],
            dtype=np.int64,
        )
        segments = [points] if len(points) == 1 else [
            line_voxels(points[index], points[index + 1]) for index in range(len(points) - 1)
        ]
        contributed = False
        for segment in segments:
            inside = np.all((segment >= start) & (segment < end), axis=1)
            local = segment[inside] - start
            if local.size:
                fault_coordinates[fault].append(local)
                contributed = True
        if contributed:
            fault_sticks[fault].add(stick)

    result = {}
    for fault, coordinate_parts in fault_coordinates.items():
        coordinates = np.unique(np.concatenate(coordinate_parts, axis=0), axis=0)
        by_inline = defaultdict(list)
        for inline, crossline, sample in coordinates:
            by_inline[int(inline)].append((int(crossline), int(sample)))
        result[fault] = {
            "by_inline": by_inline,
            "centreline_voxels": int(len(coordinates)),
            "contributing_sticks": int(len(fault_sticks[fault])),
        }
    return result


def evaluate_object(fault, reference, predictions, plane_shape):
    structure = ndimage.generate_binary_structure(2, 2)
    totals = {
        model: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_prediction": 0,
            "matched_truth": 0,
            "predicted": 0,
            "truth": 0,
            "distance_parts": [],
        }
        for model in predictions
    }
    label_voxels = 0
    validity_voxels = 0
    for inline, points in sorted(reference["by_inline"].items()):
        centreline = np.zeros(plane_shape, dtype=bool)
        crossline, sample = np.asarray(points, dtype=np.int64).T
        centreline[crossline, sample] = True
        label = ndimage.binary_dilation(centreline, structure=structure, iterations=LABEL_RADIUS)
        validity = ndimage.binary_dilation(
            centreline, structure=structure, iterations=VALIDITY_RADIUS
        )
        label_voxels += int(label.sum())
        validity_voxels += int(validity.sum())
        for model, binary in predictions.items():
            prediction = np.logical_and(np.asarray(binary[inline], dtype=bool), validity)
            counts = binary_counts(prediction[validity], label[validity])
            for key in ("tp", "fp", "fn"):
                totals[model][key] += counts[key]
            tolerant = tolerant_section_metrics(prediction, label, TOLERANCE_PIXELS)
            for key in ("matched_prediction", "matched_truth", "predicted", "truth"):
                totals[model][key] += tolerant[key]
            if tolerant["distances"] is not None:
                totals[model]["distance_parts"].append(tolerant["distances"])

    rows = []
    for model, values in totals.items():
        exact = metrics_from_counts({key: values[key] for key in ("tp", "fp", "fn")})
        tolerant_precision = values["matched_prediction"] / max(values["predicted"], 1)
        tolerant_recall = values["matched_truth"] / max(values["truth"], 1)
        distances = (
            np.concatenate(values["distance_parts"])
            if values["distance_parts"]
            else np.asarray([], dtype=np.float64)
        )
        rows.append(
            {
                "fault": fault,
                "model": model,
                "active_inline_sections": len(reference["by_inline"]),
                "contributing_sticks": reference["contributing_sticks"],
                "centreline_voxels": reference["centreline_voxels"],
                "label_voxels": label_voxels,
                "validity_voxels": validity_voxels,
                **exact,
                "tolerant_precision_3px": tolerant_precision,
                "tolerant_recall_3px": tolerant_recall,
                "tolerant_dice_3px": 2
                * tolerant_precision
                * tolerant_recall
                / max(tolerant_precision + tolerant_recall, 1e-8),
                "mean_symmetric_distance_pixels": float(distances.mean())
                if distances.size
                else None,
                "p95_symmetric_distance_pixels": float(np.percentile(distances, 95))
                if distances.size
                else None,
            }
        )
    return rows


def paired_object_bootstrap(rows):
    by_model = defaultdict(dict)
    for row in rows:
        by_model[row["model"]][row["fault"]] = row
    names = sorted(by_model)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results = []
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            faults = sorted(set(by_model[first]) & set(by_model[second]))
            for metric in ("dice", "tolerant_dice_3px"):
                differences = np.asarray(
                    [by_model[first][fault][metric] - by_model[second][fault][metric] for fault in faults],
                    dtype=np.float64,
                )
                indices = rng.integers(
                    0, len(differences), size=(BOOTSTRAP_REPLICATES, len(differences))
                )
                means = differences[indices].mean(axis=1)
                results.append(
                    {
                        "first_model": first,
                        "second_model": second,
                        "metric": metric,
                        "mean_difference_first_minus_second": float(differences.mean()),
                        "within_survey_fault_object_bootstrap_95_interval": np.percentile(
                            means, [2.5, 97.5]
                        ).tolist(),
                        "fault_object_count": len(faults),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "warning": "Fault objects are repeated units within one survey, not independent surveys.",
                    }
                )
    return results


def main():
    metadata = json.loads((BENCHMARK_ROOT / "metadata.json").read_text(encoding="utf-8"))
    records, malformed = parse_sticks(DEFAULT_STICKS)
    if malformed:
        raise ValueError(f"Malformed expert-stick lines: {malformed[:5]}")
    with segyio.open(str(DEFAULT_SEGY), "r", ignore_geometry=True) as seismic:
        ilines = np.asarray(seismic.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int64)
        xlines = np.asarray(seismic.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int64)
        inline_values = np.unique(ilines)
        crossline_values = np.unique(xlines)
        samples = np.asarray(seismic.samples, dtype=np.float64)
    resolved, _ = resolve_grid_coordinates(records, inline_values, crossline_values)
    mapped = map_records(resolved, inline_values, crossline_values, samples)
    references = fault_inline_centrelines(mapped, metadata["starts"], metadata["shape"])

    model_dirs = sorted(
        path
        for path in PREDICTION_ROOT.iterdir()
        if path.is_dir() and (path / "inference_metadata.json").exists()
    )
    predictions = {
        path.name: np.load(path / "fault_binary.npy", mmap_mode="r") for path in model_dirs
    }
    rows = []
    plane_shape = tuple(metadata["shape"][1:])
    for fault, reference in sorted(references.items()):
        rows.extend(evaluate_object(fault, reference, predictions, plane_shape))

    csv_path = PREDICTION_ROOT / "per_fault_object_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "dataset": metadata["id"],
        "fault_object_count": len(references),
        "metrics": rows,
        "paired_within_survey_fault_object_bootstrap": paired_object_bootstrap(rows),
        "claim_boundary": (
            "Object metrics isolate each released fault and its own validity corridor. Overlapping "
            "interpretations and shared geology make fault objects dependent within GN1101."
        ),
    }
    (PREDICTION_ROOT / "fault_object_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"fault_object_count": len(references), "row_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

