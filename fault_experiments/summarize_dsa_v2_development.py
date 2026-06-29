import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs" / "f3_faulta_v2_development"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    unet_real = load(ROOT / "runs/f3_faulta_fewshot_e8_inference/faulta_calibrated_results.json")
    unet_summary = next(row for row in unet_real["summary"] if row["model"] == "unet3d")
    unet_synthetic = load(ROOT / "runs/f3_faulta_fewshot_e8/unet3d/synthetic_val_after_finetune.json")
    unet_runtime = load(ROOT / "runs/f3_faulta_fewshot_e8_inference/unet3d/inference_metadata.json")

    hybrid_val = load(OUTPUT_ROOT / "dsa_hybrid_replay_e8_inference/faulta_validation_result.json")["best"]
    hybrid_test = load(OUTPUT_ROOT / "dsa_hybrid_replay_e8_inference/faulta_fixed_test_result.json")["summary"]
    hybrid_synthetic = load(OUTPUT_ROOT / "dsa_hybrid_replay_e8/synthetic_val_after_finetune.json")
    hybrid_runtime = load(OUTPUT_ROOT / "dsa_hybrid_replay_e8_inference/inference_metadata.json")

    rows = [
        {
            "model": "unet3d_fewshot",
            "parameters": 350809,
            "faulta_val_tolerant_dice": unet_summary["val_macro_tolerant_dice"],
            "faulta_test_exact_dice": unet_summary["micro_dice"],
            "faulta_test_tolerant_dice": unet_summary["macro_tolerant_dice"],
            "faulta_test_mean_surface_distance_px": unet_summary["macro_mean_symmetric_surface_distance_pixels"],
            "synthetic_val_dice_after_adaptation": unet_synthetic["best_by_dice"]["dice"],
            "field_volume_inference_seconds": unet_runtime["runtime_seconds"],
            "test_status": "previously exposed development test",
        },
        {
            "model": "dsa_hybrid_replay_fewshot",
            "parameters": 104787,
            "faulta_val_tolerant_dice": hybrid_val["macro_tolerant_dice"],
            "faulta_test_exact_dice": hybrid_test["micro_dice"],
            "faulta_test_tolerant_dice": hybrid_test["macro_tolerant_dice"],
            "faulta_test_mean_surface_distance_px": hybrid_test["macro_mean_symmetric_surface_distance_pixels"],
            "synthetic_val_dice_after_adaptation": hybrid_synthetic["best_by_dice"]["dice"],
            "field_volume_inference_seconds": hybrid_runtime["runtime_seconds"],
            "test_status": "selected by validation; test set was already exposed earlier in project",
        },
    ]
    with (OUTPUT_ROOT / "development_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_ROOT / "development_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
