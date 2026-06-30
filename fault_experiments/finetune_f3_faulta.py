import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.losses import build_loss
from fault_experiments.models import build_model


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"
VOLUME_XLINE_START = 512


def window_starts(length, patch, stride):
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


class FaultASectionDataset(Dataset):
    def __init__(self, split, augment=False):
        self.split = split
        self.augment = augment
        self.amplitude = np.load(
            BENCHMARK_ROOT / "inference_volume" / "amplitude_norm.npy", mmap_mode="r"
        )
        self.samples = []
        for case_dir in sorted((BENCHMARK_ROOT / "sticks" / split).glob("faulta_stick_*")):
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            label = np.load(case_dir / "fault_label.npy")[VOLUME_XLINE_START:VOLUME_XLINE_START + 256]
            validity = np.load(case_dir / "validity_mask.npy")[VOLUME_XLINE_START:VOLUME_XLINE_START + 256]
            inline_index = metadata["inline_index"]
            inline_start = min(max(inline_index - 64, 0), self.amplitude.shape[0] - 128)
            for time_start in window_starts(self.amplitude.shape[2], 128, 64):
                local_label = label[:, time_start : time_start + 128]
                locations = np.argwhere(local_label > 0)
                if locations.size == 0:
                    continue
                x_center = int(round(float(locations[:, 0].mean())))
                xline_start = min(max(x_center - 64, 0), self.amplitude.shape[1] - 128)
                self.samples.append(
                    {
                        "case_id": metadata["case_id"],
                        "inline_index": inline_index,
                        "inline_start": inline_start,
                        "xline_start": xline_start,
                        "time_start": time_start,
                        "label": label,
                        "validity": validity,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        i, x, t = item["inline_start"], item["xline_start"], item["time_start"]
        amplitude = np.asarray(self.amplitude[i : i + 128, x : x + 128, t : t + 128], dtype=np.float32)
        amplitude = np.ascontiguousarray(2.0 * amplitude - 1.0)
        target = np.zeros((128, 128, 128), dtype=np.float32)
        valid = np.zeros((128, 128, 128), dtype=bool)
        local_inline = item["inline_index"] - i
        target[local_inline] = item["label"][x : x + 128, t : t + 128]
        valid[local_inline] = item["validity"][x : x + 128, t : t + 128].astype(bool)
        if self.augment and np.random.rand() < 0.5:
            amplitude = amplitude[:, ::-1, :].copy()
            target = target[:, ::-1, :].copy()
            valid = valid[:, ::-1, :].copy()
        return {
            "amplitude": torch.from_numpy(amplitude[None]),
            "target": torch.from_numpy(target[None]),
            "valid": torch.from_numpy(valid[None]),
            "case_id": item["case_id"],
        }


def masked_dice_bce(logits, target, valid, pos_weight=5.0):
    selected_logits = logits[valid]
    selected_target = target[valid]
    weight = torch.tensor(pos_weight, device=logits.device)
    bce = functional.binary_cross_entropy_with_logits(selected_logits, selected_target, pos_weight=weight)
    probability = torch.sigmoid(selected_logits)
    intersection = (probability * selected_target).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (probability.sum() + selected_target.sum() + 1.0)
    return 0.5 * bce + 0.5 * dice_loss


def load_transfer_state(model, source_state):
    target_state = model.state_dict()
    transfer = {}
    converted = []
    for key, target in target_state.items():
        if key in source_state and source_state[key].shape == target.shape:
            transfer[key] = source_state[key]
            continue
        if key.endswith(".weight"):
            prefix = key[: -len(".weight")]
            depthwise_key = prefix + ".depthwise.weight"
            pointwise_key = prefix + ".pointwise.weight"
            if depthwise_key in source_state and pointwise_key in source_state:
                depthwise = source_state[depthwise_key][:, 0]
                pointwise = source_state[pointwise_key][:, :, 0, 0, 0]
                folded = pointwise[:, :, None, None, None] * depthwise[None, :, :, :, :]
                if folded.shape == target.shape:
                    transfer[key] = folded
                    converted.append(key)
    incompatible = model.load_state_dict(transfer, strict=False)
    return incompatible, converted


def run_epoch(model, loader, device, optimizer=None, amp=False, scaler=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    for batch in loader:
        amplitude = batch["amplitude"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(amplitude)
                loss = masked_dice_bce(logits, target, valid)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run_mixed_epoch(
    model,
    real_loader,
    synthetic_loader,
    synthetic_loss_fn,
    device,
    optimizer,
    replay_weight,
    amp=False,
    scaler=None,
):
    model.train()
    real_losses = []
    synthetic_losses = []
    synthetic_iterator = iter(synthetic_loader)
    for real_batch in real_loader:
        try:
            synthetic_batch = next(synthetic_iterator)
        except StopIteration:
            synthetic_iterator = iter(synthetic_loader)
            synthetic_batch = next(synthetic_iterator)
        real_amplitude = real_batch["amplitude"].to(device)
        real_target = real_batch["target"].to(device)
        real_valid = real_batch["valid"].to(device)
        synthetic_amplitude = synthetic_batch["amplitude"].to(device)
        synthetic_target = synthetic_batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            real_loss = masked_dice_bce(model(real_amplitude), real_target, real_valid)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(real_loss).backward()
        else:
            real_loss.backward()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            synthetic_loss = synthetic_loss_fn(model(synthetic_amplitude), synthetic_target)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(replay_weight * synthetic_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            (replay_weight * synthetic_loss).backward()
            optimizer.step()
        real_losses.append(float(real_loss.detach().cpu()))
        synthetic_losses.append(float(synthetic_loss.detach().cpu()))
    return float(np.mean(real_losses)), float(np.mean(synthetic_losses))


def main():
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning on independent F3 FaultA sticks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--model-override", default=None)
    parser.add_argument("--synthetic-replay-root", default=None)
    parser.add_argument("--replay-weight", type=float, default=0.35)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    training_started_at = datetime.now(timezone.utc).isoformat()
    training_started = time.perf_counter()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model_name = args.model_override or train_args.get("model", "unet3d")
    model = build_model(
        model_name,
        base_channels=train_args.get("base_channels", 8),
        use_depthwise=not train_args.get("no_depthwise", False),
        use_attention=not train_args.get("no_attention", False),
        use_aspp=not train_args.get("no_aspp", False),
        swin_feature_size=train_args.get("swin_feature_size", 12),
        swin_use_checkpoint=train_args.get("swin_use_checkpoint", False),
    )
    incompatible, converted_weights = load_transfer_state(model, checkpoint["model"])
    non_running_missing = [key for key in incompatible.missing_keys if "num_batches_tracked" not in key]
    non_running_unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))
    ]
    if model_name in {"dsa_unet3d_gn_no_attention", "dsa_gn_no_attention"}:
        non_running_unexpected = [key for key in non_running_unexpected if ".attention." not in key]
    if model_name in {"dsa_unet3d_hybrid", "dsa_hybrid"}:
        converted_prefixes = {key[: -len(".weight")] for key in converted_weights}
        non_running_missing = [
            key for key in non_running_missing if key[: -len(".weight")] not in converted_prefixes
        ]
    if non_running_missing or non_running_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint. Missing={non_running_missing}, unexpected={non_running_unexpected}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_set = FaultASectionDataset("train", augment=True)
    val_set = FaultASectionDataset("val", augment=False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    synthetic_loader = None
    synthetic_loss_fn = None
    if args.synthetic_replay_root:
        synthetic_set = SyntheticFaultDataset(args.synthetic_replay_root, "train", augment=True)
        synthetic_loader = DataLoader(synthetic_set, batch_size=1, shuffle=True)
        synthetic_loss_fn = build_loss("hybrid_focal_tversky")

    history = []
    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if synthetic_loader is None:
            train_loss = run_epoch(
                model, train_loader, device, optimizer, amp=args.amp, scaler=scaler
            )
            synthetic_replay_loss = None
        else:
            train_loss, synthetic_replay_loss = run_mixed_epoch(
                model,
                train_loader,
                synthetic_loader,
                synthetic_loss_fn,
                device,
                optimizer,
                args.replay_weight,
                amp=args.amp,
                scaler=scaler,
            )
        val_loss = run_epoch(model, val_loader, device, amp=args.amp)
        row = {
            "epoch": epoch,
            "duration_seconds": time.perf_counter() - epoch_started,
            "train_loss": train_loss,
            "synthetic_replay_loss": synthetic_replay_loss,
            "val_loss": val_loss,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            updated_args = dict(train_args)
            updated_args["model"] = model_name
            if model_name in {"dsa_unet3d_v2", "dsa_v2"}:
                updated_args["norm_type"] = "group"
                updated_args["residual_attention"] = True
            if model_name in {
                "dsa_unet3d_gn", "dsa_gn", "dsa_unet3d_gn_no_attention", "dsa_gn_no_attention"
            }:
                updated_args["norm_type"] = "group"
                updated_args["residual_attention"] = False
            if model_name in {"dsa_unet3d_gn_no_attention", "dsa_gn_no_attention"}:
                updated_args["no_attention"] = True
            if model_name in {"dsa_unet3d_hybrid", "dsa_hybrid"}:
                updated_args["hybrid_depthwise"] = True
            updated_args["finetune_dataset"] = "f3_faulta_benchmark train sticks 0-5"
            updated_args["finetune_lr"] = args.lr
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": updated_args,
                    "epoch": epoch,
                    "val": {"masked_loss": val_loss},
                    "parent_checkpoint": str(args.checkpoint),
                },
                output_dir / "best.pt",
            )
    config = {
        "parent_checkpoint": str(args.checkpoint),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "train_sticks": "0-5",
        "val_sticks": "7-8",
        "buffer_sticks": "6 and 9",
        "train_patch_count": len(train_set),
        "val_patch_count": len(val_set),
        "loss": "masked 0.5 BCE(pos_weight=5) + 0.5 soft Dice",
        "model": model_name,
        "folded_depthwise_weights": converted_weights,
        "synthetic_replay_root": args.synthetic_replay_root,
        "replay_weight": args.replay_weight if args.synthetic_replay_root else None,
        "timing": {
            "started_at_utc": training_started_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": time.perf_counter() - training_started,
            "hardware": str(torch.cuda.get_device_name(0)) if device.type == "cuda" else "CPU",
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
