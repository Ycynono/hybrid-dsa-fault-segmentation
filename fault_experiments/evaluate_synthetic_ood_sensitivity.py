from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fault_experiments.infer_real_volume import load_model
from synthetic_fault_generator import PROFILES, synthesize_sample


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/synthetic_ood_sensitivity"
MODELS = [
    ("U-Net", ROOT / "runs/thebe_adaptation/unet3d_e8/best.pt", 0.50, "#7B91C8"),
    ("Hybrid DSA", ROOT / "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt", 0.15, "#D6AD16"),
    ("SwinUNETR", ROOT / "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt", 0.40, "#60A879"),
]


def metrics(probability: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    prediction = probability >= threshold
    truth = np.asarray(truth, dtype=bool)
    tp = int(np.logical_and(prediction, truth).sum())
    fp = int(np.logical_and(prediction, ~truth).sum())
    fn = int(np.logical_and(~prediction, truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "precision": precision,
        "recall": recall,
        "dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "predicted_fraction": float(prediction.mean()),
        "truth_fraction": float(truth.mean()),
    }


def plot_summary(summary: dict, output: Path) -> None:
    profiles = list(summary["profiles"])
    models = [name for name, _, _, _ in MODELS]
    x = np.arange(len(profiles))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.5, 5.8), constrained_layout=True)
    for offset, (name, _, _, color) in enumerate(MODELS):
        means = [summary["profiles"][profile][name]["mean_dice"] for profile in profiles]
        standard_deviations = [
            summary["profiles"][profile][name]["standard_deviation_dice"] for profile in profiles
        ]
        axis.bar(
            x + (offset - 1) * width,
            means,
            width,
            yerr=standard_deviations,
            capsize=3,
            color=color,
            edgecolor="#30343B",
            linewidth=0.5,
            label=name,
        )
    axis.set_xticks(x, [value.replace("_", "\n") for value in profiles])
    axis.set_ylabel("Exact Dice (mean +/- sample SD)")
    axis.set_title("Frozen-model sensitivity to synthetic out-of-distribution factors")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=len(models))
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen field-adapted models on controlled synthetic OOD profiles.")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--samples-per-profile", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    profiles = [value.strip() for value in args.profiles.split(",") if value.strip()]
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}")
    if args.samples_per_profile < 2:
        raise ValueError("samples-per-profile must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_models = []
    for name, checkpoint, threshold, color in MODELS:
        model, _ = load_model(checkpoint, device)
        loaded_models.append((name, model, threshold, color))

    rows = []
    for profile in profiles:
        for sample_index in range(args.samples_per_profile):
            seed = args.seed + sample_index * 7_919
            amplitude, truth, fault_specs = synthesize_sample(
                (128, 128, 128), seed, profile=profile
            )
            tensor = torch.from_numpy(amplitude[None, None]).to(device)
            for name, model, threshold, _ in loaded_models:
                with torch.inference_mode():
                    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                rows.append(
                    {
                        "profile": profile,
                        "sample_index": sample_index,
                        "seed": seed,
                        "fault_count": len(fault_specs),
                        "model": name,
                        "threshold": threshold,
                        **metrics(probability, truth, threshold),
                    }
                )
            print(f"{profile} sample {sample_index + 1}/{args.samples_per_profile}", flush=True)

    with (args.output_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "analysis_scope": "Controlled synthetic OOD sensitivity of frozen field-adapted checkpoints; no retraining or threshold tuning",
        "seed": args.seed,
        "samples_per_profile": args.samples_per_profile,
        "profiles": {},
    }
    for profile in profiles:
        summary["profiles"][profile] = {}
        for name, _, _, _ in MODELS:
            selected = [row for row in rows if row["profile"] == profile and row["model"] == name]
            dice = np.asarray([row["dice"] for row in selected], dtype=np.float64)
            summary["profiles"][profile][name] = {
                "mean_dice": float(dice.mean()),
                "standard_deviation_dice": float(dice.std(ddof=1)),
                "minimum_dice": float(dice.min()),
                "maximum_dice": float(dice.max()),
                "mean_precision": float(np.mean([row["precision"] for row in selected])),
                "mean_recall": float(np.mean([row["recall"] for row in selected])),
            }
    reference = summary["profiles"].get("reference", {})
    if reference:
        for profile in profiles:
            for name, _, _, _ in MODELS:
                summary["profiles"][profile][name]["mean_dice_change_from_reference"] = (
                    summary["profiles"][profile][name]["mean_dice"] - reference[name]["mean_dice"]
                )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_summary(summary, args.output_dir / "synthetic_ood_sensitivity.png")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
