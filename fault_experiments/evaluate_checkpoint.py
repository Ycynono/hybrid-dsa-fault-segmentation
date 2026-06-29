import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.metrics import MetricAverager, binary_segmentation_metrics
from fault_experiments.models import build_model


def evaluate(model, loader, device, thresholds):
    model.eval()
    meters = {float(t): MetricAverager() for t in thresholds}
    prob_stats = []

    with torch.no_grad():
        for batch in loader:
            x = batch["amplitude"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            prob_stats.append(
                {
                    "sample_id": batch["sample_id"][0],
                    "prob_min": float(probs.min().detach().cpu()),
                    "prob_max": float(probs.max().detach().cpu()),
                    "prob_mean": float(probs.mean().detach().cpu()),
                    "prob_p95": float(torch.quantile(probs.flatten(), 0.95).detach().cpu()),
                }
            )
            for threshold, meter in meters.items():
                meter.update(binary_segmentation_metrics(logits, y, threshold=threshold))

    threshold_metrics = []
    for threshold, meter in meters.items():
        row = {"threshold": threshold}
        row.update(meter.compute())
        threshold_metrics.append(row)

    best_dice = max(threshold_metrics, key=lambda row: row["dice"])
    best_iou = max(threshold_metrics, key=lambda row: row["iou"])
    return {
        "best_by_dice": best_dice,
        "best_by_iou": best_iou,
        "threshold_metrics": threshold_metrics,
        "probability_stats": prob_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint with threshold sweep.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_pilot")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output", default=None)
    parser.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model_name = train_args.get("model", "unet3d")
    model = build_model(
        model_name,
        base_channels=train_args.get("base_channels", 8),
        use_depthwise=not train_args.get("no_depthwise", False),
        use_attention=not train_args.get("no_attention", False),
        use_aspp=not train_args.get("no_aspp", False),
        swin_feature_size=train_args.get("swin_feature_size", 12),
        swin_use_checkpoint=False,
    )
    model.load_state_dict(checkpoint["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = SyntheticFaultDataset(args.data_root, args.split, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    thresholds = [float(v) for v in args.thresholds.split(",")]

    result = evaluate(model, loader, device, thresholds)
    result["checkpoint"] = str(checkpoint_path)
    result["split"] = args.split
    result["model"] = model_name

    output = Path(args.output) if args.output else checkpoint_path.parent / f"eval_{args.split}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["best_by_dice"], indent=2))
    print("Wrote", output)


if __name__ == "__main__":
    main()
