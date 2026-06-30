from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "manuscript_tables"

SOURCES = {
    "synthetic": ROOT / "runs" / "formal_400_e50_c8" / "formal_results.json",
    "thebe": ROOT
    / "runs"
    / "thebe_final_test2_7"
    / "statistics"
    / "summary_statistics.json",
    "cracks": ROOT / "runs" / "cracks_final_statistics.json",
    "efficiency": ROOT / "runs" / "efficiency" / "inference_benchmark.json",
    "coherence": ROOT / "runs" / "dip_steered_coherence_baseline" / "test2_test7_summary.json",
}

MODEL_LABELS = {
    "unet3d": "3D U-Net",
    "dsa_unet3d": "Full DSA",
    "U-Net": "3D U-Net",
    "SwinUNETR F3-chain": "SwinUNETR",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_tables() -> tuple[dict[str, list[dict[str, Any]]], str]:
    synthetic = read_json(SOURCES["synthetic"])
    thebe = read_json(SOURCES["thebe"])
    cracks = read_json(SOURCES["cracks"])
    efficiency = read_json(SOURCES["efficiency"])
    coherence = read_json(SOURCES["coherence"])

    synthetic_rows = [
        {
            "model": MODEL_LABELS.get(item["model"], item["model"]),
            "variant": item["run"],
            "parameters": item["parameters"],
            "test_dice": item["test_dice"],
            "test_iou": item["test_iou"],
            "test_precision": item["test_precision"],
            "test_recall": item["test_recall"],
        }
        for item in synthetic
    ]

    thebe_rows = []
    thebe_rows.append(
        {
            "model": "Local coherence discontinuity",
            "parameters": "N/A",
            "n_blocks": coherence["n_blocks"],
            "macro_exact_dice": coherence["macro_block_exact_dice"],
            "macro_tolerant_dice_3px": coherence["macro_block_tolerant_dice_3px"],
            "mean_runtime_seconds_per_block": coherence["total_runtime_seconds"] / coherence["n_blocks"],
        }
    )
    for model, values in thebe["aggregate"].items():
        thebe_rows.append(
            {
                "model": MODEL_LABELS.get(model, model),
                "parameters": values["parameter_count"],
                "n_blocks": values["n_blocks"],
                "macro_exact_dice": values["macro_block_dice_mean"],
                "macro_tolerant_dice_3px": values[
                    "macro_block_tolerant_dice_3px_mean"
                ],
                "mean_runtime_seconds_per_block": values[
                    "mean_runtime_seconds_per_block"
                ],
            }
        )

    cracks_rows = []
    for partition in ("audit", "reserve_primary"):
        for model, values in cracks[partition]["aggregate"].items():
            cracks_rows.append(
                {
                    "partition": "reserve" if partition == "reserve_primary" else partition,
                    "model": MODEL_LABELS.get(model, model),
                    "n_sections": values["n_sections"],
                    "micro_precision": values["micro_precision"],
                    "micro_recall": values["micro_recall"],
                    "micro_exact_dice": values["micro_dice"],
                    "macro_exact_dice": values["macro_exact_dice"],
                    "macro_tolerant_dice_3px": values["macro_tolerant_dice_3px"],
                }
            )

    efficiency_rows = [
        {
            "model": MODEL_LABELS.get(item["model"], item["model"]),
            "execution": item["execution"],
            "dtype": item["dtype"],
            "patch_shape": "x".join(str(value) for value in item["patch_shape"]),
            "parameters": item["parameter_count"],
            "median_ms": item["median_ms"],
            "peak_memory_gib": item["peak_memory_gib"],
        }
        for item in efficiency
    ]

    reserve_pair = cracks["reserve_primary"]["hybrid_vs_swin"]
    paired_rows = []
    for metric, values in reserve_pair.items():
        paired_rows.append(
            {
                "comparison": "Hybrid DSA minus SwinUNETR",
                "partition": "CRACKS sealed reserve",
                "metric": metric,
                "paired_mean_difference": values["paired_mean_difference"],
                "bootstrap_95_ci_low": values["bootstrap_95_ci_difference"][0],
                "bootstrap_95_ci_high": values["bootstrap_95_ci_difference"][1],
                "hybrid_better_sections": values["sections_proposed_better"],
                "wilcoxon_two_sided_p": values["wilcoxon_two_sided_p"],
            }
        )

    tables = {
        "synthetic_controlled": synthetic_rows,
        "thebe_frozen": thebe_rows,
        "cracks_frozen": cracks_rows,
        "efficiency_measured": efficiency_rows,
        "cracks_reserve_paired": paired_rows,
    }

    md = ["# Auto-generated Manuscript Tables", ""]
    md.extend(
        [
            "Generated only from frozen JSON outputs. Do not edit values manually.",
            "",
            "## Thebe Frozen Test",
            "",
            markdown_table(
                ["Model", "Parameters", "Exact Dice", "Tolerant Dice", "s/block"],
                [
                    [
                        row["model"],
                        f'{row["parameters"]:,}' if isinstance(row["parameters"], int) else row["parameters"],
                        f'{row["macro_exact_dice"]:.4f}',
                        f'{row["macro_tolerant_dice_3px"]:.4f}',
                        f'{row["mean_runtime_seconds_per_block"]:.1f}',
                    ]
                    for row in thebe_rows
                ],
            ),
            "",
            "## CRACKS Frozen Audit And Reserve",
            "",
            markdown_table(
                ["Partition", "Model", "Precision", "Recall", "Exact Dice", "Tolerant Dice"],
                [
                    [
                        row["partition"].title(),
                        row["model"],
                        f'{row["micro_precision"]:.4f}',
                        f'{row["micro_recall"]:.4f}',
                        f'{row["macro_exact_dice"]:.4f}',
                        f'{row["macro_tolerant_dice_3px"]:.4f}',
                    ]
                    for row in cracks_rows
                ],
            ),
            "",
            "## Measured Patch Efficiency",
            "",
            markdown_table(
                ["Model", "Parameters", "Median ms", "Peak GiB", "Patch", "dtype"],
                [
                    [
                        row["model"],
                        f'{row["parameters"]:,}',
                        f'{row["median_ms"]:.2f}',
                        f'{row["peak_memory_gib"]:.2f}',
                        row["patch_shape"],
                        row["dtype"],
                    ]
                    for row in efficiency_rows
                ],
            ),
            "",
            "## CRACKS Reserve Paired Comparison",
            "",
            markdown_table(
                ["Metric", "Difference", "95% CI", "Wins", "Wilcoxon p"],
                [
                    [
                        row["metric"],
                        f'{row["paired_mean_difference"]:.4f}',
                        f'[{row["bootstrap_95_ci_low"]:.4f}, {row["bootstrap_95_ci_high"]:.4f}]',
                        f'{row["hybrid_better_sections"]}/20',
                        f'{row["wilcoxon_two_sided_p"]:.6g}',
                    ]
                    for row in paired_rows
                ],
            ),
            "",
            "## Controlled Synthetic Experiment",
            "",
            markdown_table(
                ["Variant", "Parameters", "Dice", "IoU", "Precision", "Recall"],
                [
                    [
                        row["variant"],
                        f'{row["parameters"]:,}',
                        f'{row["test_dice"]:.4f}',
                        f'{row["test_iou"]:.4f}',
                        f'{row["test_precision"]:.4f}',
                        f'{row["test_recall"]:.4f}',
                    ]
                    for row in synthetic_rows
                ],
            ),
            "",
        ]
    )
    return tables, "\n".join(md)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export manuscript tables from frozen experiment JSON files."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [str(path) for path in SOURCES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen sources: " + ", ".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables, markdown = build_tables()
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)

    (args.output_dir / "manuscript_tables.md").write_text(markdown, encoding="utf-8")
    source_manifest = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
        }
        for name, path in SOURCES.items()
    }
    (args.output_dir / "source_hashes.json").write_text(
        json.dumps(source_manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(tables)} CSV tables and one Markdown summary to {args.output_dir}")


if __name__ == "__main__":
    main()
