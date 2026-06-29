import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOLUME = ROOT / "processed_data/real_subvolumes/force_field_mid_384x512x128"
RUN_ROOT = ROOT / "runs/force_frozen_external"
MODELS = [
    ("unet", ROOT / "checkpoints/unet3d_thebe_e8.pt", 0.50),
    (
        "hybrid_dsa",
        ROOT / "checkpoints/hybrid_dsa_thebe_e8.pt",
        0.15,
    ),
    (
        "swinunetr_f3chain",
        ROOT / "checkpoints/swinunetr_f3chain_thebe_e8.pt",
        0.40,
    ),
]


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, checkpoint, threshold in MODELS:
        output = RUN_ROOT / name
        metadata = output / "inference_metadata.json"
        if metadata.exists():
            print(f"SKIP complete: {name}", flush=True)
            rows.append({"model": name, "status": "skipped_complete"})
            continue
        command = [
            sys.executable,
            "-m",
            "fault_experiments.infer_real_volume",
            "--checkpoint",
            str(checkpoint),
            "--volume-dir",
            str(VOLUME),
            "--output-dir",
            str(output),
            "--threshold",
            str(threshold),
            "--threshold-source",
            "Thebe val1-val2 only; frozen before FORCE external-survey inference",
        ]
        print(f"RUN {name}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        rows.append(
            {
                "model": name,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "threshold": threshold,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            }
        )
        (RUN_ROOT / "matrix_status.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)
    print("Frozen FORCE external-survey inference complete.")


if __name__ == "__main__":
    main()
