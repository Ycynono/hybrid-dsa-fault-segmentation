import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.finetune_f3_faulta import load_transfer_state, masked_dice_bce
from fault_experiments.losses import build_loss
from fault_experiments.models import build_model
from fault_experiments.thebe_dataset import ThebeBalancedPatchDataset


def train_epoch(
    model, thebe_loader, synthetic_loader, optimizer, device, replay_weight, amp=False, scaler=None
):
    model.train()
    synthetic_iterator = iter(synthetic_loader) if synthetic_loader else None
    real_losses = []
    replay_losses = []
    for batch in thebe_loader:
        amplitude = batch["amplitude"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            real_loss = masked_dice_bce(model(amplitude), target, valid, pos_weight=5.0)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(real_loss).backward()
        else:
            real_loss.backward()
        if synthetic_iterator is not None:
            try:
                synthetic = next(synthetic_iterator)
            except StopIteration:
                synthetic_iterator = iter(synthetic_loader)
                synthetic = next(synthetic_iterator)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                synthetic_loss = build_loss("hybrid_focal_tversky")(
                    model(synthetic["amplitude"].to(device)), synthetic["label"].to(device)
                )
                replay_loss = replay_weight * synthetic_loss
            if scaler is not None and scaler.is_enabled():
                scaler.scale(replay_loss).backward()
            else:
                replay_loss.backward()
            replay_losses.append(float(synthetic_loss.detach().cpu()))
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        real_losses.append(float(real_loss.detach().cpu()))
    return {
        "real_loss": float(np.mean(real_losses)),
        "synthetic_replay_loss": float(np.mean(replay_losses)) if replay_losses else None,
    }


def validate(model, loader, device, thresholds, amp=False):
    model.eval()
    loss_values = []
    counts = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}
    with torch.inference_mode():
        for batch in loader:
            amplitude = batch["amplitude"].to(device)
            target = batch["target"].to(device)
            valid = batch["valid"].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(amplitude)
                validation_loss = masked_dice_bce(logits, target, valid)
            loss_values.append(float(validation_loss.cpu()))
            probability = torch.sigmoid(logits)
            truth = target > 0.5
            for threshold in thresholds:
                prediction = probability >= threshold
                counts[threshold]["tp"] += int((prediction & truth & valid).sum().cpu())
                counts[threshold]["fp"] += int((prediction & ~truth & valid).sum().cpu())
                counts[threshold]["fn"] += int((~prediction & truth & valid).sum().cpu())
    rows = []
    for threshold, values in counts.items():
        tp, fp, fn = values["tp"], values["fp"], values["fn"]
        rows.append(
            {
                "threshold": threshold,
                "dice": 2 * tp / max(2 * tp + fp + fn, 1),
                "iou": tp / max(tp + fp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
            }
        )
    best = max(rows, key=lambda row: row["dice"])
    best["loss"] = float(np.mean(loss_values))
    return best, rows


def main():
    parser = argparse.ArgumentParser(description="Supervised Thebe train/val adaptation with replay.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="processed_data/thebe_official")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--synthetic-root", default="processed_data/synthetic_fault_v2_400")
    parser.add_argument("--replay-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_args = checkpoint.get("args", {})
    model = build_model(
        model_args.get("model", "unet3d"),
        base_channels=model_args.get("base_channels", 8),
        use_depthwise=not model_args.get("no_depthwise", False),
        use_attention=not model_args.get("no_attention", False),
        use_aspp=not model_args.get("no_aspp", False),
        swin_feature_size=model_args.get("swin_feature_size", 12),
        swin_use_checkpoint=model_args.get("swin_use_checkpoint", False),
    )
    incompatible, _ = load_transfer_state(model, checkpoint["model"])
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_set = ThebeBalancedPatchDataset(
        args.data_root, "train", samples_per_epoch=args.train_samples, augment=True, seed=args.seed
    )
    val_set = ThebeBalancedPatchDataset(
        args.data_root, "val", samples_per_epoch=args.val_samples, augment=False, seed=args.seed
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    synthetic_loader = None
    if args.synthetic_root and args.replay_weight > 0:
        synthetic_loader = DataLoader(
            SyntheticFaultDataset(args.synthetic_root, "train", augment=True), batch_size=1, shuffle=True
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    thresholds = [value / 20 for value in range(1, 20)]
    history = []
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            synthetic_loader,
            optimizer,
            device,
            args.replay_weight,
            amp=args.amp,
            scaler=scaler,
        )
        val_best, val_sweep = validate(model, val_loader, device, thresholds, amp=args.amp)
        row = {"epoch": epoch, "train": train_metrics, "val": val_best, "val_sweep": val_sweep}
        history.append(row)
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_best}), flush=True)
        if val_best["dice"] > best_dice:
            best_dice = val_best["dice"]
            updated_args = dict(model_args)
            updated_args["thebe_adaptation"] = True
            updated_args["thebe_threshold"] = val_best["threshold"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": updated_args,
                    "epoch": epoch,
                    "val": val_best,
                    "parent_checkpoint": args.checkpoint,
                },
                output_dir / "best.pt",
            )
    config = {
        **vars(args),
        "train_summary": train_set.summary(),
        "val_summary": val_set.summary(),
        "test2_test7_accessed": False,
        "selection_metric": "official Thebe val micro Dice",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
