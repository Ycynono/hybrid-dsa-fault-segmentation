import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "real_data_inference_val_thresholds_e50_c8"
VOLUME_ROOT = ROOT / "processed_data" / "real_subvolumes"

MODELS = [
    ("unet3d", ROOT / "runs/formal_400_e50_c8/unet3d_e50_c8/best.pt", 0.50),
    ("dsa_full", ROOT / "runs/formal_400_e50_c8/dsa_unet3d_full_e50_c8/best.pt", 0.35),
    ("dsa_no_aspp", ROOT / "runs/formal_400_e50_c8/dsa_unet3d_no_aspp_e50_c8/best.pt", 0.45),
]
VOLUMES = [
    "f3_main_384x512x128",
    "f3_secondary_384x512x128",
    "force_field_mid_384x512x128",
]


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    status = []
    for model_name, checkpoint, threshold in MODELS:
        for volume_name in VOLUMES:
            output_dir = RUN_ROOT / model_name / volume_name
            metadata_file = output_dir / "inference_metadata.json"
            if metadata_file.exists():
                print(f"SKIP complete: {model_name} / {volume_name}", flush=True)
                status.append({"model": model_name, "volume": volume_name, "status": "skipped_complete"})
                continue
            command = [
                sys.executable,
                "-m",
                "fault_experiments.infer_real_volume",
                "--checkpoint",
                str(checkpoint),
                "--volume-dir",
                str(VOLUME_ROOT / volume_name),
                "--output-dir",
                str(output_dir),
                "--threshold",
                str(threshold),
            ]
            print(f"RUN {model_name} / {volume_name}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            row = {
                "model": model_name,
                "volume": volume_name,
                "threshold": threshold,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
            }
            status.append(row)
            (RUN_ROOT / "matrix_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)
    print("Real-data inference matrix complete.")


if __name__ == "__main__":
    main()
