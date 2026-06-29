import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fault_experiments.evaluate_f3_faulta_benchmark import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"
THRESHOLDS = np.arange(0.05, 0.951, 0.05)


def main():
    parser = argparse.ArgumentParser(description="Validation-only FaultA threshold sweep.")
    parser.add_argument("--prediction-dir", required=True)
    args = parser.parse_args()
    prediction_dir = Path(args.prediction_dir)
    probability = np.load(prediction_dir / "fault_probability.npy", mmap_mode="r").astype(np.float32)
    val_cases = sorted((BENCHMARK_ROOT / "sticks" / "val").glob("faulta_stick_*"))
    rows = []
    for threshold in THRESHOLDS:
        binary = probability >= threshold
        metrics = aggregate([evaluate_case(binary, case_dir) for case_dir in val_cases])
        rows.append({"threshold": float(threshold), **metrics})
    best = max(rows, key=lambda row: (row["macro_tolerant_dice"], row["micro_dice"]))
    with (prediction_dir / "faulta_validation_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "split": "validation sticks 7-8 only",
        "test_labels_accessed": False,
        "selection_metric": "macro tolerant Dice at 3 pixels",
        "best": best,
    }
    (prediction_dir / "faulta_validation_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
