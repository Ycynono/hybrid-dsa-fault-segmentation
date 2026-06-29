import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_sample_metadata(root):
    root = Path(root)
    rows = []
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.exists():
            continue
        for sample_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            metadata_path = sample_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "sample_id": metadata.get("sample_id", sample_dir.name),
                    "split": split,
                    "seed": metadata.get("seed"),
                    "fault_voxel_fraction": metadata.get("fault_voxel_fraction"),
                    "n_faults": len(metadata.get("faults", [])),
                    "relative_path": str(sample_dir.relative_to(root)),
                }
            )
    return rows


def split_stats(rows):
    stats = {}
    for split in ("train", "val", "test"):
        values = np.array(
            [r["fault_voxel_fraction"] for r in rows if r["split"] == split],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        stats[split] = {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "empty_label_count": int(np.sum(values == 0)),
        }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Summarize generated synthetic fault dataset metadata.")
    parser.add_argument("--root", default="processed_data/synthetic_fault_v2_400")
    args = parser.parse_args()

    root = Path(args.root)
    rows = read_sample_metadata(root)
    stats = split_stats(rows)

    csv_path = root / "sample_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "split",
                "seed",
                "fault_voxel_fraction",
                "n_faults",
                "relative_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    stats_path = root / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print("Wrote", csv_path)
    print("Wrote", stats_path)


if __name__ == "__main__":
    main()
