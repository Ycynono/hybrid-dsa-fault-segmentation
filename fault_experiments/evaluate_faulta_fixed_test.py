import argparse
import json
from pathlib import Path

import numpy as np

from fault_experiments.evaluate_f3_faulta_benchmark import aggregate, evaluate_case


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"


def main():
    parser = argparse.ArgumentParser(description="One-time fixed-threshold FaultA development test.")
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()
    prediction_dir = Path(args.prediction_dir)
    probability = np.load(prediction_dir / "fault_probability.npy", mmap_mode="r").astype(np.float32)
    binary = probability >= args.threshold
    test_cases = sorted((BENCHMARK_ROOT / "sticks" / "test").glob("faulta_stick_*"))
    rows = [evaluate_case(binary, case_dir) for case_dir in test_cases]
    result = {
        "model": args.model_name,
        "threshold": args.threshold,
        "threshold_source": "FaultA validation sticks 7-8",
        "split": "test sticks 10-12",
        "status": "development test already exposed in this project; not a pristine final external test",
        "summary": aggregate(rows),
        "per_stick": rows,
    }
    (prediction_dir / "faulta_fixed_test_result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
