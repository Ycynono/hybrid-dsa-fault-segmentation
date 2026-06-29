import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "real_data_inference_val_thresholds_e50_c8"


def binary_agreement(a, b):
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    a_count = int(a.sum())
    b_count = int(b.sum())
    return {
        "dice": float(2 * intersection / max(a_count + b_count, 1)),
        "iou": float(intersection / max(union, 1)),
        "intersection_voxels": intersection,
        "union_voxels": union,
    }


def main():
    rows = []
    grouped = {}
    for metadata_file in sorted(RUN_ROOT.glob("*/*/inference_metadata.json")):
        model_name = metadata_file.parents[1].name
        volume_name = metadata_file.parent.name
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        stats = metadata["statistics"]
        row = {
            "model": model_name,
            "volume": volume_name,
            "threshold": stats["threshold"],
            "predicted_voxel_fraction": stats["predicted_voxel_fraction"],
            "probability_mean": stats["probability_mean"],
            "probability_p95": stats["probability_p95"],
            "probability_p99": stats["probability_p99"],
            "component_count": stats["connected_component_count_18_neighbor"],
            "largest_component_fraction": stats["largest_component_fraction_of_prediction"],
            "runtime_seconds": metadata["runtime_seconds"],
        }
        rows.append(row)
        grouped.setdefault(volume_name, {})[model_name] = metadata_file.parent

    agreements = []
    for volume_name, models in sorted(grouped.items()):
        for model_a, model_b in combinations(sorted(models), 2):
            binary_a = np.load(models[model_a] / "fault_binary.npy", mmap_mode="r").astype(bool)
            binary_b = np.load(models[model_b] / "fault_binary.npy", mmap_mode="r").astype(bool)
            row = {"volume": volume_name, "model_a": model_a, "model_b": model_b}
            row.update(binary_agreement(binary_a, binary_b))
            agreements.append(row)

    with (RUN_ROOT / "field_prediction_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (RUN_ROOT / "model_agreement.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agreements[0]))
        writer.writeheader()
        writer.writerows(agreements)
    (RUN_ROOT / "field_prediction_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (RUN_ROOT / "model_agreement.json").write_text(json.dumps(agreements, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": rows, "agreements": agreements}, indent=2))


if __name__ == "__main__":
    main()
