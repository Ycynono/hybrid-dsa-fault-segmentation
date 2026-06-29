from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_FIELDS = [
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "loss",
    "prob_mean",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> dict:
    history_path = run_dir / "history.json"
    config_path = run_dir / "config.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json: {history_path}")

    history = load_json(history_path)
    config = load_json(config_path) if config_path.exists() else {}
    if not history:
        raise ValueError(f"Empty history: {history_path}")

    best = max(history, key=lambda item: item["val"]["dice"])
    final = history[-1]

    row = {
        "run": run_dir.name,
        "path": str(run_dir),
        "model": config.get("model", ""),
        "base_channels": config.get("base_channels", ""),
        "loss": config.get("loss", ""),
        "epochs_config": config.get("epochs", ""),
        "epochs_completed": len(history),
        "batch_size": config.get("batch_size", ""),
        "lr": config.get("lr", ""),
        "best_epoch": best["epoch"],
        "best_threshold": best["val"].get("threshold", ""),
    }

    for field in METRIC_FIELDS:
        row[f"best_val_{field}"] = best["val"].get(field, "")
        row[f"final_val_{field}"] = final["val"].get(field, "")

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize fault segmentation training runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to summarize.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--json-output", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    rows = [summarize_run(Path(run)) for run in args.runs]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.json_output:
        json_output = Path(args.json_output)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        with json_output.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
