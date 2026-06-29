import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fault_experiments.evaluate_thebe_block import evaluate_section


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/cracks_external_v1"
AUDIT_ROOT = ROOT / "runs/cracks_audit_frozen"
OUTPUT_ROOT = ROOT / "runs/cracks_reserve_final"
MODEL_DIRS = {
    "U-Net": "unet",
    "Hybrid DSA": "hybrid_dsa",
    "SwinUNETR F3-chain": "swinunetr_f3chain",
}


def aggregate(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    exact_dice = [2 * row["tp"] / max(2 * row["tp"] + row["fp"] + row["fn"], 1) for row in rows]
    return {
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "micro_iou": tp / max(tp + fp + fn, 1),
        "macro_exact_dice": float(np.mean(exact_dice)),
        "macro_tolerant_dice_3px": float(np.mean([row["tolerant_dice"] for row in rows])),
    }


def main():
    lock_path = ROOT / "FINAL_PROTOCOL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("Final protocol must be locked before reserve evaluation")
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Reserve output already exists; refusing to rerun: {OUTPUT_ROOT}")
    protocol = json.loads(lock_path.read_text(encoding="utf-8"))
    thresholds = {row["name"]: row["threshold"] for row in protocol["models"]}
    sections = protocol["reserve_sections"]
    reserve_dir = DATA_DIR / "sealed_reserve"
    masks = np.load(reserve_dir / "reserve_expert_fault_masks.npy", mmap_mode="r")
    certain_masks = np.load(
        reserve_dir / "reserve_expert_certain_masks.npy", mmap_mode="r"
    )
    OUTPUT_ROOT.mkdir(parents=True)
    summaries = {}
    for name, directory in MODEL_DIRS.items():
        probability = np.load(
            AUDIT_ROOT / directory / "fault_probability_float16.npy", mmap_mode="r"
        )
        rows = []
        certain_rows = []
        for position, section in enumerate(sections):
            prediction = probability[section - 1] >= thresholds[name]
            rows.append(
                {
                    "section": section,
                    **evaluate_section(prediction, np.asarray(masks[position]), tolerance=3),
                }
            )
            certain_rows.append(
                {
                    "section": section,
                    **evaluate_section(
                        prediction, np.asarray(certain_masks[position]), tolerance=3
                    ),
                }
            )
        model_dir = OUTPUT_ROOT / directory
        model_dir.mkdir()
        with (model_dir / "per_section_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "model": name,
            "threshold": thresholds[name],
            "threshold_source": protocol["threshold_source"],
            "reserve_sections": sections,
            "combined_certain_and_uncertain_faults": aggregate(rows),
            "certain_faults_only": aggregate(certain_rows),
        }
        summaries[name] = summary
        (model_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    opening_record = {
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_lock": str(lock_path.relative_to(ROOT)),
        "reserve_loaded_once": True,
        "models": summaries,
    }
    (OUTPUT_ROOT / "reserve_results.json").write_text(
        json.dumps(opening_record, indent=2), encoding="utf-8"
    )
    print(json.dumps(opening_record, indent=2))


if __name__ == "__main__":
    main()
