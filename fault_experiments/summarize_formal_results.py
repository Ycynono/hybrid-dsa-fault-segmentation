from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from fault_experiments.models import build_model


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(value: str | None):
    if not value:
        return None
    return datetime.fromisoformat(value)


def duration_minutes(started_at: str | None, ended_at: str | None):
    start = parse_time(started_at)
    end = parse_time(ended_at)
    if start is None or end is None:
        return ""
    return round((end - start).total_seconds() / 60.0, 2)


def count_parameters(config: dict) -> int:
    model = build_model(
        config.get("model", "unet3d"),
        base_channels=int(config.get("base_channels", 8)),
        use_depthwise=not config.get("no_depthwise", False),
        use_attention=not config.get("no_attention", False),
        use_aspp=not config.get("no_aspp", False),
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine validation, test, parameter, and timing summaries.")
    parser.add_argument("--run-root", default="runs/formal_400_e50_c8")
    parser.add_argument("--output", default="runs/formal_400_e50_c8/formal_results.csv")
    parser.add_argument("--json-output", default="runs/formal_400_e50_c8/formal_results.json")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    val_rows = read_json(run_root / "summary.json")
    queue_status = read_json(run_root / "queue_status.json")
    timings = {item["name"]: item for item in queue_status.get("experiments", [])}

    rows = []
    for val in val_rows:
        run_name = val["run"]
        run_dir = Path(val["path"])
        config = read_json(run_dir / "config.json")
        test_eval = read_json(run_dir / "eval_test.json")
        test_best = test_eval["best_by_dice"]
        timing = timings.get(run_name, {})

        row = {
            "run": run_name,
            "model": val["model"],
            "base_channels": val["base_channels"],
            "parameters": count_parameters(config),
            "duration_minutes": duration_minutes(timing.get("started_at"), timing.get("ended_at")),
            "best_epoch": val["best_epoch"],
            "val_threshold": val["best_threshold"],
            "val_dice": val["best_val_dice"],
            "val_iou": val["best_val_iou"],
            "val_precision": val["best_val_precision"],
            "val_recall": val["best_val_recall"],
            "test_threshold": test_best["threshold"],
            "test_dice": test_best["dice"],
            "test_iou": test_best["iou"],
            "test_precision": test_best["precision"],
            "test_recall": test_best["recall"],
            "test_accuracy": test_best["accuracy"],
            "loss": val["loss"],
            "run_dir": str(run_dir),
        }
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_output = Path(args.json_output)
    json_output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
