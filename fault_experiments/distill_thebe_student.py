import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.finetune_f3_faulta import masked_dice_bce
from fault_experiments.infer_real_volume import load_model
from fault_experiments.losses import build_loss
from fault_experiments.thebe_dataset import ThebeBalancedPatchDataset
from fault_experiments.train_thebe_adaptation import validate


ROOT = Path(__file__).resolve().parents[1]


def masked_distillation_loss(
    student_logits,
    teacher_logits,
    valid,
    temperature=2.0,
    teacher_positive_weight=4.0,
):
    soft_target = torch.sigmoid(teacher_logits / temperature)
    confidence = torch.sigmoid(teacher_logits)
    weights = 1.0 + teacher_positive_weight * confidence
    voxel_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature, soft_target, reduction="none"
    )
    valid_weights = weights * valid
    return temperature**2 * (voxel_loss * valid_weights).sum() / valid_weights.sum().clamp_min(1.0)


def train_epoch(
    student,
    teacher,
    real_loader,
    synthetic_loader,
    optimizer,
    scaler,
    device,
    amp,
    kd_weight,
    replay_weight,
    temperature,
):
    student.train()
    teacher.eval()
    synthetic_iterator = iter(synthetic_loader)
    meters = {"hard": [], "kd": [], "synthetic": []}
    synthetic_loss_fn = build_loss("hybrid_focal_tversky")
    for batch in real_loader:
        amplitude = batch["amplitude"].to(device)
        target = batch["target"].to(device)
        valid = batch["valid"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp):
            teacher_logits = teacher(amplitude) if kd_weight > 0 else None
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            student_logits = student(amplitude)
            hard_loss = masked_dice_bce(
                student_logits, target, valid, pos_weight=5.0
            )
            if teacher_logits is not None:
                kd_loss = masked_distillation_loss(
                    student_logits,
                    teacher_logits,
                    valid,
                    temperature=temperature,
                )
            else:
                kd_loss = student_logits.new_zeros(())
            real_loss = (1.0 - kd_weight) * hard_loss + kd_weight * kd_loss
        scaler.scale(real_loss).backward()

        try:
            synthetic = next(synthetic_iterator)
        except StopIteration:
            synthetic_iterator = iter(synthetic_loader)
            synthetic = next(synthetic_iterator)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            replay_loss = synthetic_loss_fn(
                student(synthetic["amplitude"].to(device)),
                synthetic["label"].to(device),
            )
        scaler.scale(replay_weight * replay_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        meters["hard"].append(float(hard_loss.detach().cpu()))
        meters["kd"].append(float(kd_loss.detach().cpu()))
        meters["synthetic"].append(float(replay_loss.detach().cpu()))
    return {name: float(np.mean(values)) for name, values in meters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student-checkpoint",
        default="checkpoints/hybrid_dsa_thebe_e8.pt",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default="checkpoints/swinunetr_f3chain_thebe_e8.pt",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="processed_data/thebe_official")
    parser.add_argument("--synthetic-root", default="processed_data/synthetic_fault_v2_400")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=160)
    parser.add_argument("--val-samples", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kd-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--replay-weight", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, student_checkpoint = load_model(Path(args.student_checkpoint), device)
    teacher, _ = load_model(Path(args.teacher_checkpoint), device)
    teacher.requires_grad_(False).eval()
    student.train()

    train_set = ThebeBalancedPatchDataset(
        args.data_root,
        "train",
        samples_per_epoch=args.train_samples,
        augment=True,
        seed=args.seed,
    )
    val_set = ThebeBalancedPatchDataset(
        args.data_root,
        "val",
        samples_per_epoch=args.val_samples,
        augment=False,
        seed=args.seed,
    )
    real_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    synthetic_loader = DataLoader(
        SyntheticFaultDataset(args.synthetic_root, "train", augment=True),
        batch_size=1,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    thresholds = [value / 20 for value in range(1, 20)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            student,
            teacher,
            real_loader,
            synthetic_loader,
            optimizer,
            scaler,
            device,
            args.amp,
            args.kd_weight,
            args.replay_weight,
            args.temperature,
        )
        val_best, val_sweep = validate(
            student, val_loader, device, thresholds, amp=args.amp
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_best,
            "val_sweep": val_sweep,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_best["dice"] > best_dice:
            best_dice = val_best["dice"]
            updated_args = dict(student_checkpoint.get("args", {}))
            updated_args.update(
                {
                    "distilled": args.kd_weight > 0,
                    "distillation_teacher": args.teacher_checkpoint,
                    "distillation_kd_weight": args.kd_weight,
                    "distillation_temperature": args.temperature,
                    "thebe_threshold": val_best["threshold"],
                }
            )
            torch.save(
                {
                    "model": student.state_dict(),
                    "args": updated_args,
                    "epoch": epoch,
                    "val": val_best,
                    "parent_checkpoint": args.student_checkpoint,
                },
                output_dir / "best.pt",
            )
    config = {
        **vars(args),
        "train_summary": train_set.summary(),
        "val_summary": val_set.summary(),
        "development_only": True,
        "test2_test7_accessed_before_experiment_design": True,
        "selection_metric": "official Thebe val micro Dice",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
