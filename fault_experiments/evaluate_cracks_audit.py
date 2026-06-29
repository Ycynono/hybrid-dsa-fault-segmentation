import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from fault_experiments.evaluate_thebe_block import evaluate_section
from fault_experiments.infer_real_volume import infer_volume, load_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/cracks_external_v1"
OUTPUT_ROOT = ROOT / "runs/cracks_audit_frozen"
MODELS = [
    ("U-Net", ROOT / "checkpoints/unet3d_thebe_e8.pt", 0.50),
    (
        "Hybrid DSA",
        ROOT / "checkpoints/hybrid_dsa_thebe_e8.pt",
        0.15,
    ),
    (
        "SwinUNETR F3-chain",
        ROOT / "checkpoints/swinunetr_f3chain_thebe_e8.pt",
        0.40,
    ),
]


def aggregate(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "micro_iou": tp / max(tp + fp + fn, 1),
        "macro_tolerant_precision_3px": float(
            np.mean([row["tolerant_precision"] for row in rows])
        ),
        "macro_tolerant_recall_3px": float(
            np.mean([row["tolerant_recall"] for row in rows])
        ),
        "macro_tolerant_dice_3px": float(
            np.mean([row["tolerant_dice"] for row in rows])
        ),
    }


def main():
    split = json.loads((DATA_DIR / "expert_split.json").read_text(encoding="utf-8"))
    if not split.get("policy_created_before_model_inference"):
        raise RuntimeError("CRACKS audit/reserve split was not preregistered")
    audit_sections = split["audit_sections"]
    amplitude = np.load(DATA_DIR / "amplitude_01.npy", mmap_mode="r")
    audit_masks = np.load(DATA_DIR / "audit_expert_fault_masks.npy", mmap_mode="r")
    certain_masks = np.load(DATA_DIR / "audit_expert_certain_masks.npy", mmap_mode="r")
    if len(audit_sections) != len(audit_masks):
        raise ValueError("Audit section and mask counts differ")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_summaries = []
    for name, checkpoint_path, threshold in MODELS:
        model_dir = OUTPUT_ROOT / name.lower().replace(" ", "_").replace("-", "")
        model_dir.mkdir(parents=True, exist_ok=True)
        if (model_dir / "summary.json").exists():
            print(f"SKIP {name}", flush=True)
            continue
        model, _ = load_model(checkpoint_path, device)
        started = time.perf_counter()
        probability, starts = infer_volume(
            model,
            amplitude,
            patch_shape=(128, 128, 128),
            stride=(64, 64, 64),
            device=device,
        )
        runtime = time.perf_counter() - started
        np.save(model_dir / "fault_probability_float16.npy", probability.astype(np.float16))
        rows = []
        certain_rows = []
        for position, section in enumerate(audit_sections):
            prediction = probability[section - 1] >= threshold
            row = {
                "section": section,
                **evaluate_section(prediction, np.asarray(audit_masks[position]), tolerance=3),
            }
            certain_row = {
                "section": section,
                **evaluate_section(prediction, np.asarray(certain_masks[position]), tolerance=3),
            }
            rows.append(row)
            certain_rows.append(certain_row)
        summary = {
            "model": name,
            "checkpoint": str(checkpoint_path),
            "threshold": threshold,
            "threshold_source": "official Thebe val1-val2; no CRACKS calibration",
            "scope": "preregistered CRACKS audit sections only; sealed reserve not loaded",
            "audit_sections": audit_sections,
            "runtime_seconds": runtime,
            "patch_starts": starts,
            "combined_certain_and_uncertain_faults": aggregate(rows),
            "certain_faults_only": aggregate(certain_rows),
        }
        with (model_dir / "per_section_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (model_dir / "per_section_certain_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(certain_rows[0]))
            writer.writeheader()
            writer.writerows(certain_rows)
        (model_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
        del model, probability
        torch.cuda.empty_cache()
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
