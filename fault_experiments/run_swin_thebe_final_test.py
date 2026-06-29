import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "processed_data/thebe_official/test"
OUTPUT_ROOT = ROOT / "runs/thebe_final_test2_7/swin_unetr"
CHECKPOINT = ROOT / "runs/thebe_adaptation/swin_unetr_e8/best.pt"
THRESHOLD = 0.50


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = []
    for index in range(2, 8):
        block = f"test{index}"
        output_dir = OUTPUT_ROOT / block
        if (output_dir / "summary.json").exists():
            status.append({"block": block, "status": "skipped_complete"})
            continue
        command = [
            sys.executable,
            "-m",
            "fault_experiments.evaluate_thebe_block",
            "--data-dir",
            str(DATA_ROOT / block),
            "--checkpoint",
            str(CHECKPOINT),
            "--threshold",
            str(THRESHOLD),
            "--threshold-source",
            "official Thebe val1-val2; frozen SwinUNETR checkpoint epoch7",
            "--output-dir",
            str(output_dir),
            "--stride",
            "64",
        ]
        print(f"RUN SwinUNETR / {block}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        row = {
            "block": block,
            "threshold": THRESHOLD,
            "returncode": completed.returncode,
            "status": "complete" if completed.returncode == 0 else "failed",
        }
        status.append(row)
        (OUTPUT_ROOT / "run_status.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
