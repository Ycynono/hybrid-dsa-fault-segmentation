from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fault_experiments.dataset import SyntheticFaultDataset
from fault_experiments.models import build_model


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = build_model(
        train_args.get("model", "unet3d"),
        base_channels=train_args.get("base_channels", 8),
        use_depthwise=not train_args.get("no_depthwise", False),
        use_attention=not train_args.get("no_attention", False),
        use_aspp=not train_args.get("no_aspp", False),
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, train_args


def pick_slice(label: np.ndarray) -> int:
    scores = label.reshape(label.shape[0], -1).sum(axis=1)
    return int(np.argmax(scores))


def render_sample(amplitude, label, prob, threshold, sample_id, output_path: Path) -> None:
    z = pick_slice(label)
    pred = (prob >= threshold).astype(np.float32)
    false_positive = np.logical_and(pred == 1, label == 0)
    false_negative = np.logical_and(pred == 0, label == 1)
    error = np.zeros((*label.shape, 3), dtype=np.float32)
    error[..., 0] = false_positive
    error[..., 1] = label
    error[..., 2] = false_negative

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), constrained_layout=True)
    fig.suptitle(f"{sample_id} | slice {z} | threshold {threshold:.2f}", fontsize=11)

    axes[0].imshow(amplitude[z], cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Amplitude")

    axes[1].imshow(label[z], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Label")

    im = axes[2].imshow(prob[z], cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Probability")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(pred[z], cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Prediction")

    axes[4].imshow(error[z])
    axes[4].set_title("Error: FP red, FN blue")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render synthetic fault prediction QC slices.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="processed_data/synthetic_fault_v2_400")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--tag", default="", help="Optional filename tag to distinguish ablations.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, train_args = load_model(Path(args.checkpoint), device)
    dataset = SyntheticFaultDataset(args.data_root, args.split, augment=False)

    output_dir = Path(args.output_dir)
    written = []
    with torch.no_grad():
        for index in range(min(args.n_samples, len(dataset))):
            batch = dataset[index]
            x = batch["amplitude"].unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            amplitude = batch["amplitude"].squeeze().numpy()
            label = batch["label"].squeeze().numpy()
            sample_id = batch["sample_id"]
            model_name = train_args.get("model", "model")
            tag = args.tag or model_name
            output_path = output_dir / f"{tag}_{sample_id}_qc.png"
            render_sample(amplitude, label, prob, args.threshold, sample_id, output_path)
            written.append(str(output_path))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
