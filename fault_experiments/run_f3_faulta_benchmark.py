import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VOLUME_DIR = ROOT / "processed_data" / "f3_faulta_benchmark" / "inference_volume"
OUTPUT_ROOT = ROOT / "runs" / "f3_faulta_benchmark_e50_c8"
MODELS = [
    ("unet3d", ROOT / "runs/formal_400_e50_c8/unet3d_e50_c8/best.pt", 0.50),
    ("dsa_full", ROOT / "runs/formal_400_e50_c8/dsa_unet3d_full_e50_c8/best.pt", 0.35),
    ("dsa_no_aspp", ROOT / "runs/formal_400_e50_c8/dsa_unet3d_no_aspp_e50_c8/best.pt", 0.45),
]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = []
    for model_name, checkpoint, threshold in MODELS:
        output_dir = OUTPUT_ROOT / model_name
        command = [
            sys.executable,
            "-m",
            "fault_experiments.infer_real_volume",
            "--checkpoint",
            str(checkpoint),
            "--volume-dir",
            str(VOLUME_DIR),
            "--output-dir",
            str(output_dir),
            "--threshold",
            str(threshold),
            "--threshold-source",
            "synthetic validation Dice optimum; fixed before FaultA evaluation",
            "--stride",
            "64,64,64",
        ]
        print(f"RUN {model_name}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        row = {"model": model_name, "threshold": threshold, "returncode": completed.returncode}
        row["status"] = "complete" if completed.returncode == 0 else "failed"
        status.append(row)
        (OUTPUT_ROOT / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
