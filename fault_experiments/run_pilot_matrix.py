import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "unet3d",
        "model": "unet3d",
        "extra_args": [],
    },
    {
        "name": "dsa_unet3d_full",
        "model": "dsa_unet3d",
        "extra_args": [],
    },
    {
        "name": "dsa_no_depthwise",
        "model": "dsa_unet3d",
        "extra_args": ["--no-depthwise"],
    },
    {
        "name": "dsa_no_attention",
        "model": "dsa_unet3d",
        "extra_args": ["--no-attention"],
    },
    {
        "name": "dsa_no_aspp",
        "model": "dsa_unet3d",
        "extra_args": ["--no-aspp"],
    },
]


def best_val_row(history):
    if not history:
        return {}
    return max(history, key=lambda row: row["val"].get("dice", -1.0))


def summarize_run(run_dir, experiment):
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history: {history_path}")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    best = best_val_row(history)
    val = best.get("val", {})
    train = best.get("train", {})
    return {
        "experiment": experiment["name"],
        "model": experiment["model"],
        "epoch": best.get("epoch"),
        "base_channels": config.get("base_channels"),
        "train_loss": train.get("loss"),
        "train_dice": train.get("dice"),
        "train_iou": train.get("iou"),
        "val_loss": val.get("loss"),
        "val_threshold": val.get("threshold"),
        "val_dice": val.get("dice"),
        "val_iou": val.get("iou"),
        "val_precision": val.get("precision"),
        "val_recall": val.get("recall"),
        "val_specificity": val.get("specificity"),
        "val_accuracy": val.get("accuracy"),
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Run a controlled pilot matrix of baseline and DSA ablation models.")
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_pilot")
    parser.add_argument("--output-root", default="runs/pilot_matrix")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="dice_bce")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for experiment in EXPERIMENTS:
        run_dir = output_root / experiment["name"]
        if args.skip_existing and (run_dir / "history.json").exists():
            print(f"Skipping existing run: {run_dir}")
        else:
            cmd = [
                sys.executable,
                "-m",
                "fault_experiments.train_baseline",
                "--data-root",
                args.data_root,
                "--output-dir",
                str(run_dir),
                "--model",
                experiment["model"],
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--base-channels",
                str(args.base_channels),
                "--lr",
                str(args.lr),
                "--loss",
                args.loss,
                "--seed",
                str(args.seed),
                "--num-workers",
                str(args.num_workers),
            ]
            cmd.extend(experiment["extra_args"])
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        rows.append(summarize_run(run_dir, experiment))

    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote", summary_json)
    print("Wrote", summary_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
