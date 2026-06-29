import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "processed_data" / "thebe_official" / "test"
OUTPUT_ROOT = ROOT / "runs" / "thebe_final_test2_7"
MODELS = [
    (
        "unet3d",
        ROOT / "checkpoints/unet3d_thebe_e8.pt",
        0.50,
    ),
    (
        "dsa_hybrid_replay",
        ROOT / "checkpoints/hybrid_dsa_thebe_e8.pt",
        0.15,
    ),
    (
        "swin_unetr_f3chain",
        ROOT / "checkpoints/swinunetr_f3chain_thebe_e8.pt",
        0.40,
    ),
]


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = []
    for model_name, checkpoint, threshold in MODELS:
        for index in range(2, 8):
            block_name = f"test{index}"
            output_dir = OUTPUT_ROOT / model_name / block_name
            if (output_dir / "summary.json").exists():
                status.append({"model": model_name, "block": block_name, "status": "skipped_complete"})
                continue
            command = [
                sys.executable,
                "-m",
                "fault_experiments.evaluate_thebe_block",
                "--data-dir",
                str(DATA_ROOT / block_name),
                "--checkpoint",
                str(checkpoint),
                "--threshold",
                str(threshold),
                "--threshold-source",
                "official Thebe val1-val2; frozen before test2-test7",
                "--output-dir",
                str(output_dir),
                "--stride",
                "64",
            ]
            print(f"RUN {model_name} / {block_name}", flush=True)
            completed = subprocess.run(command, cwd=ROOT, check=False)
            row = {
                "model": model_name,
                "block": block_name,
                "threshold": threshold,
                "returncode": completed.returncode,
                "status": "complete" if completed.returncode == 0 else "failed",
            }
            status.append(row)
            (OUTPUT_ROOT / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
