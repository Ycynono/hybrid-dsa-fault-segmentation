import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm, colors
from scipy import ndimage

from fault_experiments.models import build_model


def parse_triplet(value):
    values = tuple(int(v.strip()) for v in value.split(","))
    if len(values) != 3 or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Expected three positive integers, for example 128,128,128")
    return values


def window_starts(length, patch, stride):
    if patch > length:
        raise ValueError(f"Patch size {patch} exceeds axis length {length}")
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def blending_window(shape):
    axes = []
    for size in shape:
        if size == 1:
            axis = np.ones(1, dtype=np.float32)
        else:
            axis = np.hanning(size).astype(np.float32)
            axis = np.maximum(axis, 0.05)
        axes.append(axis)
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model = build_model(
        train_args.get("model", "unet3d"),
        base_channels=train_args.get("base_channels", 8),
        use_depthwise=not train_args.get("no_depthwise", False),
        use_attention=not train_args.get("no_attention", False),
        use_aspp=not train_args.get("no_aspp", False),
        swin_feature_size=train_args.get("swin_feature_size", 12),
        swin_use_checkpoint=False,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def infer_volume(model, amplitude_01, patch_shape, stride, device):
    starts = [window_starts(n, p, s) for n, p, s in zip(amplitude_01.shape, patch_shape, stride)]
    weight = blending_window(patch_shape)
    probability_sum = np.zeros(amplitude_01.shape, dtype=np.float32)
    weight_sum = np.zeros(amplitude_01.shape, dtype=np.float32)
    patch_count = len(starts[0]) * len(starts[1]) * len(starts[2])
    completed = 0

    with torch.inference_mode():
        for i in starts[0]:
            for j in starts[1]:
                for k in starts[2]:
                    patch_01 = np.asarray(
                        amplitude_01[i : i + patch_shape[0], j : j + patch_shape[1], k : k + patch_shape[2]],
                        dtype=np.float32,
                    )
                    # Training amplitudes were normalized to [-1, 1].
                    patch = np.ascontiguousarray(2.0 * patch_01 - 1.0)
                    tensor = torch.from_numpy(patch[None, None]).to(device)
                    probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                    region = np.s_[i : i + patch_shape[0], j : j + patch_shape[1], k : k + patch_shape[2]]
                    probability_sum[region] += probability * weight
                    weight_sum[region] += weight
                    completed += 1
                    print(f"patch {completed}/{patch_count}", flush=True)

    return probability_sum / np.maximum(weight_sum, 1e-8), [list(v) for v in starts]


def prediction_statistics(probability, threshold):
    binary = probability >= threshold
    structure = ndimage.generate_binary_structure(3, 2)
    component_labels, component_count = ndimage.label(binary, structure=structure)
    sizes = np.bincount(component_labels.ravel())
    foreground_sizes = sizes[1:]
    foreground_count = int(binary.sum())
    largest = int(foreground_sizes.max()) if foreground_sizes.size else 0
    return {
        "probability_min": float(probability.min()),
        "probability_max": float(probability.max()),
        "probability_mean": float(probability.mean()),
        "probability_p95": float(np.percentile(probability, 95)),
        "probability_p99": float(np.percentile(probability, 99)),
        "threshold": float(threshold),
        "predicted_voxel_fraction": float(binary.mean()),
        "connected_component_count_18_neighbor": int(component_count),
        "largest_component_voxels": largest,
        "largest_component_fraction_of_prediction": float(largest / max(foreground_count, 1)),
    }


def plot_qc(amplitude, probability, threshold, output):
    mids = tuple(v // 2 for v in amplitude.shape)
    views = [
        ("Inline", amplitude[mids[0], :, :].T, probability[mids[0], :, :].T),
        ("Crossline", amplitude[:, mids[1], :].T, probability[:, mids[1], :].T),
        ("Time slice", amplitude[:, :, mids[2]], probability[:, :, mids[2]]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), dpi=180)
    for row, (name, amp, prob) in enumerate(views):
        axes[row, 0].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name}: amplitude")
        axes[row, 1].imshow(prob, cmap="inferno", aspect="auto", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{name}: probability")
        axes[row, 2].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        overlay = np.ma.masked_where(prob < threshold, prob)
        axes[row, 2].imshow(overlay, cmap="autumn", aspect="auto", vmin=threshold, vmax=1, alpha=0.75)
        axes[row, 2].set_title(f"{name}: overlay (p >= {threshold:.2f})")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def overlay_colors(amplitude, probability, threshold):
    rgba = cm.gray(np.clip(amplitude, 0, 1))
    fault = cm.autumn(np.clip(probability, 0, 1))
    alpha = np.where(probability >= threshold, 0.72, 0.0)[..., None]
    rgba[..., :3] = (1 - alpha) * rgba[..., :3] + alpha * fault[..., :3]
    rgba[..., 3] = 1.0
    return rgba


def plot_3d_slices(amplitude, probability, threshold, output, title):
    ni, nx, nt = amplitude.shape
    im, xm, tm = ni // 2, nx // 2, nt // 2
    fig = plt.figure(figsize=(11, 8), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ni))
    ax.plot_surface(xx, yy, np.full_like(xx, tm), rstride=4, cstride=4,
                    facecolors=overlay_colors(amplitude[:, :, tm], probability[:, :, tm], threshold), shade=False)
    xx, zz = np.meshgrid(np.arange(nx), np.arange(nt))
    ax.plot_surface(xx, np.full_like(xx, im), zz, rstride=2, cstride=4,
                    facecolors=overlay_colors(amplitude[im, :, :].T, probability[im, :, :].T, threshold), shade=False)
    yy, zz = np.meshgrid(np.arange(ni), np.arange(nt))
    ax.plot_surface(np.full_like(yy, xm), yy, zz, rstride=2, cstride=4,
                    facecolors=overlay_colors(amplitude[:, xm, :].T, probability[:, xm, :].T, threshold), shade=False)

    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ni - 1)
    ax.set_zlim(nt - 1, 0)
    ax.set_xlabel("Crossline / axis 1")
    ax.set_ylabel("Inline / axis 0")
    ax.set_zlabel("Time / axis 2")
    ax.set_box_aspect((nx, ni, nt * 1.4))
    ax.view_init(elev=24, azim=-55)
    ax.set_title(title, pad=16)
    mappable = cm.ScalarMappable(norm=colors.Normalize(threshold, 1), cmap="autumn")
    fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08, label="Predicted fault probability")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sliding-window inference on a normalized real seismic volume.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--volume-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--threshold-source", default="synthetic validation Dice optimum; fixed before field-data inference")
    parser.add_argument("--patch-shape", type=parse_triplet, default=(128, 128, 128))
    parser.add_argument("--stride", type=parse_triplet, default=(64, 64, 128))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    volume_dir = Path(args.volume_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((volume_dir / "metadata.json").read_text(encoding="utf-8"))
    amplitude = np.load(volume_dir / "amplitude_norm.npy", mmap_mode="r")
    if tuple(amplitude.shape) != tuple(metadata["shape"]):
        raise ValueError(f"Amplitude shape {amplitude.shape} does not match metadata {metadata['shape']}")
    if float(np.nanmin(amplitude)) < -1e-3 or float(np.nanmax(amplitude)) > 1.001:
        raise ValueError("Real amplitude must be normalized to [0, 1] before inference")

    device_name = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model, checkpoint = load_model(checkpoint_path, device)
    started = time.perf_counter()
    probability, starts = infer_volume(model, amplitude, args.patch_shape, args.stride, device)
    elapsed = time.perf_counter() - started
    binary = (probability >= args.threshold).astype(np.uint8)
    np.save(output_dir / "fault_probability.npy", probability.astype(np.float16))
    np.save(output_dir / "fault_binary.npy", binary)

    train_args = checkpoint.get("args", {})
    stats = prediction_statistics(probability, args.threshold)
    result = {
        "data_role": "field-data model prediction; not ground truth",
        "volume": metadata,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_arguments": train_args,
        "input_transform": "amplitude_norm.npy [0,1] converted to [-1,1] as 2*x-1",
        "patch_shape": list(args.patch_shape),
        "stride": list(args.stride),
        "window_starts": starts,
        "blending": "separable Hann window clipped to minimum weight 0.05",
        "threshold_source": args.threshold_source,
        "runtime_seconds": elapsed,
        "device": str(device),
        "statistics": stats,
        "interpretation_warning": "Connected-component and occupancy values are descriptive proxies, not accuracy metrics. Dice/IoU require independent interpreter labels.",
    }
    (output_dir / "inference_metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_qc(amplitude, probability, args.threshold, output_dir / "prediction_qc.png")
    plot_3d_slices(
        amplitude,
        probability,
        args.threshold,
        output_dir / "prediction_3d_slices.png",
        f"{metadata['id']} - {train_args.get('model', 'model')}",
    )
    print(json.dumps(stats, indent=2))
    print(f"Wrote {output_dir} in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
