import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fault_experiments.evaluate_f3_faulta_benchmark import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"
PREDICTION_ROOT = ROOT / "runs" / "f3_faulta_benchmark_e50_c8"
THRESHOLDS = np.arange(0.05, 0.951, 0.05)


def evaluate_split(probability, case_dirs, threshold):
    binary = probability >= threshold
    return [evaluate_case(binary, case_dir) for case_dir in case_dirs]


def main():
    parser = argparse.ArgumentParser(description="Calibrate thresholds on FaultA validation sticks.")
    parser.add_argument("--prediction-root", default=str(PREDICTION_ROOT))
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root)
    val_cases = sorted((BENCHMARK_ROOT / "sticks" / "val").glob("faulta_stick_*"))
    test_cases = sorted((BENCHMARK_ROOT / "sticks" / "test").glob("faulta_stick_*"))
    sweep_rows = []
    test_rows = []
    summaries = []

    for model_dir in sorted(path for path in prediction_root.iterdir() if path.is_dir()):
        probability = np.load(model_dir / "fault_probability.npy", mmap_mode="r").astype(np.float32)
        model_sweep = []
        for threshold in THRESHOLDS:
            metrics = aggregate(evaluate_split(probability, val_cases, float(threshold)))
            row = {"model": model_dir.name, "threshold": float(threshold), **metrics}
            sweep_rows.append(row)
            model_sweep.append(row)
        best = max(model_sweep, key=lambda row: (row["macro_tolerant_dice"], row["micro_dice"]))
        calibrated_threshold = best["threshold"]
        rows = evaluate_split(probability, test_cases, calibrated_threshold)
        for row in rows:
            row["model"] = model_dir.name
            row["calibrated_threshold"] = calibrated_threshold
            test_rows.append(row)
        summaries.append(
            {
                "model": model_dir.name,
                "threshold_selection_split": "val sticks 7-8",
                "threshold_selection_metric": "macro tolerant Dice at 3 pixels",
                "calibrated_threshold": calibrated_threshold,
                "val_macro_tolerant_dice": best["macro_tolerant_dice"],
                **aggregate(rows),
            }
        )

    with (prediction_root / "faulta_val_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
        writer.writeheader()
        writer.writerows(sweep_rows)
    with (prediction_root / "faulta_calibrated_test_per_stick.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    with (prediction_root / "faulta_calibrated_test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (prediction_root / "faulta_calibrated_results.json").write_text(
        json.dumps({"summary": summaries, "per_stick": test_rows}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
