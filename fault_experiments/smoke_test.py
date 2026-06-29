import argparse
import json

import torch
from torch.utils.data import DataLoader

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.losses import DiceBCELoss
from fault_experiments.metrics import binary_segmentation_metrics
from fault_experiments.models import build_model


def main():
    parser = argparse.ArgumentParser(description="Smoke test data loading, forward pass, loss, and metrics.")
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_pilot")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="unet3d")
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--no-depthwise", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-aspp", action="store_true")
    args = parser.parse_args()

    dataset = SyntheticFaultDataset(args.data_root, args.split, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    x = batch["amplitude"]
    y = batch["label"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        base_channels=args.base_channels,
        use_depthwise=not args.no_depthwise,
        use_attention=not args.no_attention,
        use_aspp=not args.no_aspp,
    ).to(device)
    loss_fn = DiceBCELoss()

    with torch.no_grad():
        logits = model(x.to(device))
        loss = loss_fn(logits, y.to(device))
        metrics = binary_segmentation_metrics(logits, y.to(device))

    report = {
        "sample_id": batch["sample_id"][0],
        "input_shape": list(x.shape),
        "label_shape": list(y.shape),
        "logit_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
        "metrics": metrics,
        "device": str(device),
        "model": args.model,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
