import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fault_experiments.models import build_model


ROOT = Path(__file__).resolve().parents[1]
VOLUME_PATH = ROOT / "processed_data/f3_faulta_benchmark/inference_volume/amplitude_norm.npy"


def starts(length, patch=128, stride=64):
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def main():
    parser = argparse.ArgumentParser(description="Re-estimate BatchNorm statistics on F3 train-region amplitudes.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = build_model(
        train_args.get("model", "dsa_unet3d"),
        base_channels=train_args.get("base_channels", 8),
        use_depthwise=not train_args.get("no_depthwise", False),
        use_attention=not train_args.get("no_attention", False),
        use_aspp=not train_args.get("no_aspp", False),
    )
    model.load_state_dict(checkpoint["model"])
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm3d)]
    if not batch_norms:
        raise ValueError("Checkpoint has no BatchNorm3d layers to adapt")
    for module in batch_norms:
        module.reset_running_stats()
        module.momentum = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    volume = np.load(VOLUME_PATH, mmap_mode="r")
    # Inline indices 0:128 cover train sticks 0-5 and buffer stick 6, but no validation/test sticks.
    x_starts = starts(volume.shape[1])
    t_starts = starts(volume.shape[2])
    patch_count = 0
    with torch.inference_mode():
        for _ in range(args.passes):
            for x in x_starts:
                for t in t_starts:
                    patch = np.asarray(volume[0:128, x : x + 128, t : t + 128], dtype=np.float32)
                    patch = np.ascontiguousarray(2.0 * patch - 1.0)
                    model(torch.from_numpy(patch[None, None]).to(device))
                    patch_count += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    adapted_args = dict(train_args)
    adapted_args["domain_adaptation"] = "F3 train-region BatchNorm statistics only"
    torch.save(
        {
            "model": model.state_dict(),
            "args": adapted_args,
            "epoch": checkpoint.get("epoch"),
            "val": checkpoint.get("val"),
            "parent_checkpoint": str(args.checkpoint),
            "adaptation": {
                "method": "AdaBN cumulative running statistics",
                "passes": args.passes,
                "patch_count": patch_count,
                "inline_index_range": [0, 127],
                "labels_used": False,
                "validation_or_test_amplitudes_used": False,
            },
        },
        output,
    )
    output.with_suffix(".json").write_text(
        json.dumps({"batch_norm_layers": len(batch_norms), "patch_count": patch_count}, indent=2),
        encoding="utf-8",
    )
    print(f"Adapted {len(batch_norms)} BatchNorm layers using {patch_count} patches")


if __name__ == "__main__":
    main()
