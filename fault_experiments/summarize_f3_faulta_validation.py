import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("zero_shot", ROOT / "runs/f3_faulta_benchmark_e50_c8/faulta_calibrated_results.json"),
    ("few_shot_e8", ROOT / "runs/f3_faulta_fewshot_e8_inference/faulta_calibrated_results.json"),
]
OUTPUT = ROOT / "runs" / "f3_faulta_real_validation_comparison.csv"


def main():
    rows = []
    for protocol, source in SOURCES:
        data = json.loads(source.read_text(encoding="utf-8"))
        for summary in data["summary"]:
            rows.append(
                {
                    "protocol": protocol,
                    "model": summary["model"],
                    "calibrated_threshold": summary["calibrated_threshold"],
                    "test_stick_count": summary["case_count"],
                    "micro_dice": summary["micro_dice"],
                    "micro_iou": summary["micro_iou"],
                    "macro_tolerant_dice_3px": summary["macro_tolerant_dice"],
                    "macro_tolerant_precision_3px": summary["macro_tolerant_precision"],
                    "macro_tolerant_recall_3px": summary["macro_tolerant_recall"],
                    "time_continuity_recall": summary["macro_time_continuity_recall"],
                    "mean_surface_distance_px": summary["macro_mean_symmetric_surface_distance_pixels"],
                    "surface_distance_p95_px": summary["macro_symmetric_surface_distance_p95_pixels"],
                }
            )
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
