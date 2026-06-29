import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from fault_experiments.models import build_model


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "runs" / "thebe_final_test2_7"
MODEL_DIRS = {
    "U-Net": "unet3d",
    "Hybrid DSA": "dsa_hybrid_replay",
    "SwinUNETR F3-chain": "swin_unetr_f3chain",
}


def count_parameters(model_name):
    if model_name == "U-Net":
        model = build_model("unet3d", base_channels=8)
    elif model_name == "Hybrid DSA":
        model = build_model(
            "dsa_unet3d",
            base_channels=8,
            use_depthwise=True,
            use_attention=True,
            use_aspp=True,
            hybrid_depthwise=True,
        )
    else:
        model = build_model("swin_unetr", swin_feature_size=12, swin_use_checkpoint=False)
    return sum(parameter.numel() for parameter in model.parameters())


def load_crossline_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tp = int(row["tp"])
            fp = int(row["fp"])
            fn = int(row["fn"])
            denominator = 2 * tp + fp + fn
            rows.append(
                {
                    "crossline": int(row["crossline_local_index"]),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "dice": 2 * tp / denominator if denominator else 1.0,
                    "tolerant_dice_3px": float(row["tolerant_dice"]),
                }
            )
    return rows


def percentile_interval(samples):
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def paired_block_statistics(block_rows, metric, baseline_name, proposed_name, rng, iterations):
    baseline = np.array([row[baseline_name][metric] for row in block_rows], dtype=np.float64)
    proposed = np.array([row[proposed_name][metric] for row in block_rows], dtype=np.float64)
    differences = proposed - baseline
    indices = rng.integers(0, len(block_rows), size=(iterations, len(block_rows)))
    bootstrap_differences = differences[indices].mean(axis=1)
    nonzero = differences[differences != 0]
    wilcoxon_result = wilcoxon(proposed, baseline, alternative="two-sided", method="exact")
    sign_result = binomtest(
        int(np.sum(nonzero > 0)), len(nonzero), p=0.5, alternative="two-sided"
    )
    return {
        "metric": metric,
        "baseline": baseline_name,
        "proposed": proposed_name,
        "n_independent_blocks": len(block_rows),
        "baseline_mean": float(baseline.mean()),
        "proposed_mean": float(proposed.mean()),
        "paired_mean_difference": float(differences.mean()),
        "paired_relative_improvement_percent": float(
            100.0 * differences.mean() / baseline.mean()
        ),
        "bootstrap_95_ci_difference": percentile_interval(bootstrap_differences),
        "bootstrap_probability_difference_gt_zero": float(
            np.mean(bootstrap_differences > 0)
        ),
        "blocks_proposed_better": int(np.sum(differences > 0)),
        "wilcoxon_two_sided_p": float(wilcoxon_result.pvalue),
        "sign_test_two_sided_p": float(sign_result.pvalue),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260627)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    block_rows = []
    crossline_rows = []
    summaries = {}
    for block_index in range(2, 8):
        block = f"test{block_index}"
        row = {"block": block}
        summaries[block] = {}
        for display_name, directory_name in MODEL_DIRS.items():
            model_dir = args.input_dir / directory_name / block
            summary = json.loads((model_dir / "summary.json").read_text(encoding="utf-8"))
            summaries[block][display_name] = summary
            row[display_name] = {
                "dice": float(summary["dice"]),
                "iou": float(summary["iou"]),
                "precision": float(summary["precision"]),
                "recall": float(summary["recall"]),
                "tolerant_dice_3px": float(summary["macro_tolerant_dice_3px"]),
                "runtime_seconds": float(summary["runtime_seconds"]),
            }
            for crossline in load_crossline_rows(model_dir / "per_crossline_metrics.csv"):
                crossline_rows.append(
                    {"model": display_name, "block": block, **crossline}
                )
        block_rows.append(row)

    flat_blocks = []
    for row in block_rows:
        flat = {"block": row["block"]}
        prefixes = {
            "U-Net": "unet",
            "Hybrid DSA": "hybrid",
            "SwinUNETR F3-chain": "swin_f3chain",
        }
        for model in MODEL_DIRS:
            prefix = prefixes[model]
            for metric, value in row[model].items():
                flat[f"{prefix}_{metric}"] = value
        flat["dice_difference"] = row["Hybrid DSA"]["dice"] - row["U-Net"]["dice"]
        flat["tolerant_dice_3px_difference"] = (
            row["Hybrid DSA"]["tolerant_dice_3px"]
            - row["U-Net"]["tolerant_dice_3px"]
        )
        flat_blocks.append(flat)

    rng = np.random.default_rng(args.seed)
    paired_statistics = {
        metric: paired_block_statistics(
            block_rows, metric, "U-Net", "Hybrid DSA", rng, args.bootstrap_iterations
        )
        for metric in ("dice", "tolerant_dice_3px")
    }
    pairwise_comparisons = {}
    for baseline, proposed in [
        ("U-Net", "Hybrid DSA"),
        ("Hybrid DSA", "SwinUNETR F3-chain"),
        ("U-Net", "SwinUNETR F3-chain"),
    ]:
        comparison_name = f"{proposed}_vs_{baseline}"
        pairwise_comparisons[comparison_name] = {
            metric: paired_block_statistics(
                block_rows,
                metric,
                baseline,
                proposed,
                rng,
                args.bootstrap_iterations,
            )
            for metric in ("dice", "tolerant_dice_3px")
        }

    aggregate = {}
    for model, directory_name in MODEL_DIRS.items():
        model_crosslines = [row for row in crossline_rows if row["model"] == model]
        tp = sum(row["tp"] for row in model_crosslines)
        fp = sum(row["fp"] for row in model_crosslines)
        fn = sum(row["fn"] for row in model_crosslines)
        runtime = sum(summaries[block][model]["runtime_seconds"] for block in summaries)
        model_blocks = [row[model] for row in block_rows]
        aggregate[model] = {
            "parameter_count": count_parameters(model),
            "n_blocks": len(model_blocks),
            "n_crosslines": len(model_crosslines),
            "micro_precision": tp / (tp + fp),
            "micro_recall": tp / (tp + fn),
            "micro_dice": 2 * tp / (2 * tp + fp + fn),
            "macro_block_dice_mean": float(np.mean([row["dice"] for row in model_blocks])),
            "macro_block_dice_std": float(np.std([row["dice"] for row in model_blocks], ddof=1)),
            "macro_block_tolerant_dice_3px_mean": float(
                np.mean([row["tolerant_dice_3px"] for row in model_blocks])
            ),
            "macro_block_tolerant_dice_3px_std": float(
                np.std([row["tolerant_dice_3px"] for row in model_blocks], ddof=1)
            ),
            "total_runtime_seconds": runtime,
            "mean_runtime_seconds_per_block": runtime / len(model_blocks),
        }

    efficiency = {
        "parameter_reduction_percent": 100.0
        * (1.0 - aggregate["Hybrid DSA"]["parameter_count"] / aggregate["U-Net"]["parameter_count"]),
        "hybrid_to_unet_runtime_ratio": aggregate["Hybrid DSA"]["total_runtime_seconds"]
        / aggregate["U-Net"]["total_runtime_seconds"],
        "swin_to_hybrid_parameter_ratio": aggregate["SwinUNETR F3-chain"]["parameter_count"]
        / aggregate["Hybrid DSA"]["parameter_count"],
        "swin_to_hybrid_runtime_ratio": aggregate["SwinUNETR F3-chain"]["total_runtime_seconds"]
        / aggregate["Hybrid DSA"]["total_runtime_seconds"],
    }
    result = {
        "analysis_scope": "Frozen Thebe test2-test7; thresholds selected on val1-val2 only",
        "bootstrap": {
            "unit": "paired test block",
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
        },
        "aggregate": aggregate,
        "paired_block_statistics": paired_statistics,
        "pairwise_comparisons": pairwise_comparisons,
        "efficiency": efficiency,
    }
    (output_dir / "summary_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_csv(output_dir / "block_metrics.csv", flat_blocks, list(flat_blocks[0]))
    write_csv(
        output_dir / "crossline_metrics.csv",
        crossline_rows,
        ["model", "block", "crossline", "tp", "fp", "fn", "dice", "tolerant_dice_3px"],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
