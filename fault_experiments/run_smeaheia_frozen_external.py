import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOLUME = ROOT / "processed_data" / "smeaheia" / "expert_roi_384x512x640"
RUN_ROOT = ROOT / "runs" / "smeaheia_frozen_external"
MODELS = [
    ("unet", ROOT / "runs/thebe_adaptation/unet3d_e8/best.pt", 0.50),
    (
        "hybrid_dsa",
        ROOT / "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt",
        0.15,
    ),
    (
        "swinunetr_f3chain",
        ROOT / "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt",
        0.40,
    ),
]


def main():
    metadata = json.loads((VOLUME / "metadata.json").read_text(encoding="utf-8"))
    if not metadata["selection_policy"]["locked_before_prediction"]:
        raise RuntimeError("Smeaheia ROI was not locked before prediction.")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, checkpoint, threshold in MODELS:
        output = RUN_ROOT / name
        result_path = output / "inference_metadata.json"
        if result_path.exists():
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
            "Thebe val1-val2 only; frozen before Smeaheia download and ROI selection",
            "--stride",
            "64,64,128",
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
    print("Frozen Smeaheia inference complete.")


if __name__ == "__main__":
    main()
