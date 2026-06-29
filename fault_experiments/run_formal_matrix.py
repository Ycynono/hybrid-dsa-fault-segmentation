from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "unet3d_e50_c8",
        "model": "unet3d",
        "flags": [],
    },
    {
        "name": "dsa_unet3d_full_e50_c8",
        "model": "dsa_unet3d",
        "flags": [],
    },
    {
        "name": "dsa_unet3d_no_depthwise_e50_c8",
        "model": "dsa_unet3d",
        "flags": ["--no-depthwise"],
    },
    {
        "name": "dsa_unet3d_no_attention_e50_c8",
        "model": "dsa_unet3d",
        "flags": ["--no-attention"],
    },
    {
        "name": "dsa_unet3d_no_aspp_e50_c8",
        "model": "dsa_unet3d",
        "flags": ["--no-aspp"],
    },
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_history_length(run_dir: Path) -> int:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return 0
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    return len(history)


def is_complete(run_dir: Path, epochs: int) -> bool:
    return (run_dir / "best.pt").exists() and load_history_length(run_dir) >= epochs


def write_status(status_path: Path, status: dict) -> None:
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {now()} =====\n")
        log.write(" ".join(cmd) + "\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the formal 400-sample fault segmentation matrix.")
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_400")
    parser.add_argument("--output-root", default="runs/formal_400_e50_c8")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loss", default="hybrid_focal_tversky")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logs_dir = output_root / "logs"
    status_path = output_root / "queue_status.json"
    status = {
        "started_at": now(),
        "data_root": args.data_root,
        "output_root": str(output_root),
        "epochs": args.epochs,
        "base_channels": args.base_channels,
        "experiments": [],
    }
    write_status(status_path, status)

    completed_run_dirs = []
    for exp in EXPERIMENTS:
        run_dir = output_root / exp["name"]
        exp_status = {
            "name": exp["name"],
            "run_dir": str(run_dir),
            "status": "pending",
            "started_at": None,
            "ended_at": None,
        }
        status["experiments"].append(exp_status)
        write_status(status_path, status)

        if args.skip_existing and is_complete(run_dir, args.epochs):
            exp_status["status"] = "skipped_existing"
            exp_status["started_at"] = now()
            exp_status["ended_at"] = now()
            completed_run_dirs.append(run_dir)
            write_status(status_path, status)
            continue

        cmd = [
            sys.executable,
            "-m",
            "fault_experiments.train_baseline",
            "--data-root",
            args.data_root,
            "--output-dir",
            str(run_dir),
            "--model",
            exp["model"],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--base-channels",
            str(args.base_channels),
            "--loss",
            args.loss,
            "--patience",
            "0",
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
            *exp["flags"],
        ]
        exp_status["status"] = "running_train"
        exp_status["started_at"] = now()
        write_status(status_path, status)
        code = run_command(cmd, logs_dir / f"{exp['name']}.log")
        if code != 0:
            exp_status["status"] = "failed_train"
            exp_status["ended_at"] = now()
            exp_status["return_code"] = code
            write_status(status_path, status)
            raise SystemExit(code)

        eval_cmd = [
            sys.executable,
            "-m",
            "fault_experiments.evaluate_checkpoint",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--data-root",
            args.data_root,
            "--split",
            "test",
            "--output",
            str(run_dir / "eval_test.json"),
        ]
        exp_status["status"] = "running_test_eval"
        write_status(status_path, status)
        code = run_command(eval_cmd, logs_dir / f"{exp['name']}_eval_test.log")
        if code != 0:
            exp_status["status"] = "failed_test_eval"
            exp_status["ended_at"] = now()
            exp_status["return_code"] = code
            write_status(status_path, status)
            raise SystemExit(code)

        exp_status["status"] = "completed"
        exp_status["ended_at"] = now()
        completed_run_dirs.append(run_dir)
        write_status(status_path, status)

    summary_cmd = [
        sys.executable,
        "-m",
        "fault_experiments.summarize_runs",
        "--runs",
        *[str(p) for p in completed_run_dirs],
        "--output",
        str(output_root / "summary.csv"),
        "--json-output",
        str(output_root / "summary.json"),
    ]
    status["summary_status"] = "running"
    write_status(status_path, status)
    code = run_command(summary_cmd, logs_dir / "summary.log")
    status["summary_status"] = "completed" if code == 0 else "failed"
    status["ended_at"] = now()
    write_status(status_path, status)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
