import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.losses import build_loss
from fault_experiments.metrics import MetricAverager, binary_segmentation_metrics
from fault_experiments.models import build_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None, threshold=0.5, amp=False, scaler=None):
    training = optimizer is not None
    model.train(training)
    loss_meter = 0.0
    metrics_meter = MetricAverager()

    for batch in loader:
        x = batch["amplitude"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_meter += float(loss.detach().cpu())
        metrics_meter.update(binary_segmentation_metrics(logits.detach(), y.detach(), threshold=threshold))

    metrics = metrics_meter.compute()
    metrics["loss"] = loss_meter / max(len(loader), 1)
    return metrics


def evaluate_thresholds(model, loader, loss_fn, device, thresholds, amp=False):
    model.eval()
    loss_meter = 0.0
    meters = {float(t): MetricAverager() for t in thresholds}
    prob_means = []
    with torch.no_grad():
        for batch in loader:
            x = batch["amplitude"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            loss_meter += float(loss.detach().cpu())
            probs = torch.sigmoid(logits)
            prob_means.append(float(probs.mean().detach().cpu()))
            for threshold, meter in meters.items():
                meter.update(binary_segmentation_metrics(logits, y, threshold=threshold))

    rows = []
    for threshold, meter in meters.items():
        row = {"threshold": threshold}
        row.update(meter.compute())
        row["loss"] = loss_meter / max(len(loader), 1)
        row["prob_mean"] = sum(prob_means) / max(len(prob_means), 1)
        rows.append(row)
    return max(rows, key=lambda row: row["dice"]), rows


def main():
    parser = argparse.ArgumentParser(description="Train a compact 3D U-Net baseline.")
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_pilot")
    parser.add_argument("--output-dir", default="runs/unet3d_pilot")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", default="unet3d")
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--swin-feature-size", type=int, default=12)
    parser.add_argument("--swin-use-checkpoint", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-depthwise", action="store_true")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--no-aspp", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="dice_bce")
    parser.add_argument("--thresholds", default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    training_started_at = datetime.now(timezone.utc).isoformat()
    training_started = time.perf_counter()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = SyntheticFaultDataset(args.data_root, "train", augment=True)
    val_set = SyntheticFaultDataset(args.data_root, "val", augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        base_channels=args.base_channels,
        use_depthwise=not args.no_depthwise,
        use_attention=not args.no_attention,
        use_aspp=not args.no_aspp,
        swin_feature_size=args.swin_feature_size,
        swin_use_checkpoint=args.swin_use_checkpoint,
    ).to(device)
    loss_fn = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")

    history = []
    best_dice = -1.0
    epochs_without_improvement = 0
    thresholds = [float(v) for v in args.thresholds.split(",")]
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, loss_fn, device, optimizer, amp=args.amp, scaler=scaler
        )
        val_metrics, threshold_rows = evaluate_thresholds(
            model, val_loader, loss_fn, device, thresholds, amp=args.amp
        )
        row = {
            "epoch": epoch,
            "duration_seconds": time.perf_counter() - epoch_started,
            "train": train_metrics,
            "val": val_metrics,
            "val_threshold_sweep": threshold_rows,
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val": val_metrics,
                },
                output_dir / "best.pt",
            )
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best val dice={best_dice:.6f}")
            break

    config = {
        **vars(args),
        "timing": {
            "started_at_utc": training_started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": time.perf_counter() - training_started,
            "hardware": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "CPU",
        },
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
