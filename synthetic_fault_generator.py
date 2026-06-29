import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def ricker_wavelet(length=31, frequency=0.18):
    half = length // 2
    t = np.arange(-half, half + 1, dtype=np.float32)
    a = (np.pi * frequency * t) ** 2
    wavelet = (1.0 - 2.0 * a) * np.exp(-a)
    wavelet -= wavelet.mean()
    wavelet /= np.max(np.abs(wavelet)) + 1e-8
    return wavelet.astype(np.float32)


def make_reflectivity_trace(nt, rng):
    n_events = rng.integers(18, 34)
    trace = np.zeros(nt, dtype=np.float32)
    event_positions = rng.choice(np.arange(4, nt - 4), size=n_events, replace=False)
    event_positions.sort()
    amplitudes = rng.normal(0.0, 1.0, size=n_events).astype(np.float32)
    amplitudes *= rng.uniform(0.6, 1.4, size=n_events).astype(np.float32)
    trace[event_positions] = amplitudes
    trace = gaussian_filter(trace, sigma=rng.uniform(0.3, 0.8))
    return trace


def build_layer_model(shape, rng):
    ni, nx, nt = shape
    trace = make_reflectivity_trace(nt + 64, rng)

    ii, xx, tt = np.meshgrid(
        np.linspace(-1.0, 1.0, ni, dtype=np.float32),
        np.linspace(-1.0, 1.0, nx, dtype=np.float32),
        np.arange(nt, dtype=np.float32),
        indexing="ij",
    )

    fold = np.zeros((ni, nx), dtype=np.float32)
    for _ in range(rng.integers(2, 5)):
        amp = rng.uniform(-8.0, 12.0)
        sx = rng.uniform(0.5, 1.4)
        sy = rng.uniform(0.5, 1.4)
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        fold += amp * np.sin(np.pi * sx * xx[:, :, 0] + phase_x) * np.cos(
            np.pi * sy * ii[:, :, 0] + phase_y
        )
    fold = gaussian_filter(fold, sigma=rng.uniform(8.0, 18.0))

    regional_dip = rng.uniform(-12.0, 12.0) * ii[:, :, 0] + rng.uniform(-12.0, 12.0) * xx[:, :, 0]
    strat_coord = tt + fold[:, :, None] + regional_dip[:, :, None] + 32.0
    model = map_coordinates(trace, [strat_coord], order=1, mode="nearest").reshape(shape)
    return model.astype(np.float32), strat_coord.astype(np.float32)


def add_faults(strat_coord, shape, rng):
    ni, nx, nt = shape
    ii, xx, _ = np.meshgrid(
        np.linspace(-1.0, 1.0, ni, dtype=np.float32),
        np.linspace(-1.0, 1.0, nx, dtype=np.float32),
        np.arange(nt, dtype=np.float32),
        indexing="ij",
    )

    coord = strat_coord.copy()
    label = np.zeros(shape, dtype=np.uint8)
    fault_specs = []

    n_faults = int(rng.integers(2, 7))
    for fault_id in range(n_faults):
        angle = rng.uniform(0, np.pi)
        normal_i = np.cos(angle)
        normal_x = np.sin(angle)
        center_i = rng.uniform(-0.55, 0.55)
        center_x = rng.uniform(-0.55, 0.55)
        curvature = rng.uniform(-0.18, 0.18)
        dip_term = rng.uniform(-0.25, 0.25)
        throw = rng.uniform(4.0, 18.0) * rng.choice([-1.0, 1.0])
        transition = rng.uniform(0.015, 0.035)
        thickness = rng.uniform(0.008, 0.020)

        distance_2d = (
            normal_i * (ii[:, :, 0] - center_i)
            + normal_x * (xx[:, :, 0] - center_x)
            + curvature * (ii[:, :, 0] ** 2 - xx[:, :, 0] ** 2)
        )
        # Low-angle/listric-like behavior with depth-varying apparent position.
        depth_shift = dip_term * np.linspace(-1.0, 1.0, nt, dtype=np.float32)[None, None, :]
        distance_3d = distance_2d[:, :, None] + depth_shift
        displacement = throw * np.tanh(distance_3d / transition)
        coord += displacement

        vertical_gate = rng.uniform(0.15, 0.95)
        active_depth = np.linspace(0.0, 1.0, nt, dtype=np.float32)[None, None, :]
        active = active_depth < vertical_gate
        fault_mask = (np.abs(distance_3d) < thickness) & active
        label[fault_mask] = 1

        fault_specs.append(
            {
                "fault_id": fault_id,
                "angle_rad": float(angle),
                "center_i": float(center_i),
                "center_x": float(center_x),
                "curvature": float(curvature),
                "dip_term": float(dip_term),
                "throw_samples": float(throw),
                "transition": float(transition),
                "thickness": float(thickness),
                "active_depth_fraction": float(vertical_gate),
            }
        )

    return coord.astype(np.float32), label, fault_specs


def synthesize_sample(shape, seed):
    rng = np.random.default_rng(seed)
    base, strat_coord = build_layer_model(shape, rng)
    faulted_coord, label, fault_specs = add_faults(strat_coord, shape, rng)

    ni, nx, nt = shape
    ii, xx, _ = np.meshgrid(
        np.arange(ni, dtype=np.float32),
        np.arange(nx, dtype=np.float32),
        np.arange(nt, dtype=np.float32),
        indexing="ij",
    )
    amplitude = map_coordinates(
        base,
        [ii, xx, np.clip(faulted_coord, 0, nt - 1)],
        order=1,
        mode="nearest",
    ).reshape(shape)

    wavelet_lengths = [length for length in (25, 31, 37) if length <= nt]
    if not wavelet_lengths:
        wavelet_lengths = [max(3, nt if nt % 2 else nt - 1)]
    wavelet = ricker_wavelet(
        length=int(rng.choice(wavelet_lengths)), frequency=rng.uniform(0.12, 0.22)
    )
    amplitude = np.apply_along_axis(lambda tr: np.convolve(tr, wavelet, mode="same"), 2, amplitude)

    lateral_texture = gaussian_filter(rng.normal(0, 1, size=shape).astype(np.float32), sigma=(5.0, 5.0, 1.5))
    amplitude += rng.uniform(0.02, 0.07) * lateral_texture
    amplitude += rng.normal(0.0, rng.uniform(0.01, 0.04), size=shape).astype(np.float32)

    amplitude = gaussian_filter(amplitude, sigma=(0.35, 0.35, 0.15))
    p1, p99 = np.percentile(amplitude, [1, 99])
    amplitude = np.clip((amplitude - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)
    amplitude = (2.0 * amplitude - 1.0).astype(np.float32)

    return amplitude, label.astype(np.uint8), fault_specs


def plot_qc(amplitude, label, out_file, title):
    mids = [s // 2 for s in amplitude.shape]
    panels = [
        ("Amplitude inline", amplitude[mids[0], :, :].T, "gray", (-1, 1)),
        ("Label inline", label[mids[0], :, :].T, "Reds", (0, 1)),
        ("Amplitude crossline", amplitude[:, mids[1], :].T, "gray", (-1, 1)),
        ("Label crossline", label[:, mids[1], :].T, "Reds", (0, 1)),
        ("Amplitude time", amplitude[:, :, mids[2]], "gray", (-1, 1)),
        ("Label time", label[:, :, mids[2]], "Reds", (0, 1)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=180)
    for ax, (name, image, cmap, clim) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap, aspect="auto", vmin=clim[0], vmax=clim[1])
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def split_for_index(index, n_train, n_val):
    if index < n_train:
        return "train"
    if index < n_train + n_val:
        return "val"
    return "test"


def generate_dataset(output_dir, n_samples, shape, seed, n_train, n_val, qc_count):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    for idx in range(n_samples):
        sample_seed = int(seed + idx * 7919)
        split = split_for_index(idx, n_train, n_val)
        sample_id = f"syn_{idx:05d}"
        sample_dir = output_dir / split / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        amplitude, label, fault_specs = synthesize_sample(shape, sample_seed)
        np.save(sample_dir / "amplitude.npy", amplitude.astype(np.float16))
        np.save(sample_dir / "fault_label.npy", label)

        metadata = {
            "sample_id": sample_id,
            "split": split,
            "seed": sample_seed,
            "shape": list(shape),
            "amplitude_file": "amplitude.npy",
            "fault_label_file": "fault_label.npy",
            "label_source": "analytic fault geometry, not gradient thresholding",
            "fault_voxel_fraction": float(label.mean()),
            "faults": fault_specs,
        }
        (sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if idx < qc_count:
            plot_qc(amplitude, label, sample_dir / "qc.png", sample_id)

        samples.append(
            {
                "sample_id": sample_id,
                "split": split,
                "relative_path": str(sample_dir.relative_to(output_dir)),
                "seed": sample_seed,
                "fault_voxel_fraction": float(label.mean()),
            }
        )
        print(sample_id, split, "fault_fraction", round(float(label.mean()), 5))

    manifest = {
        "description": "Synthetic 3D seismic fault dataset generated from analytic fault geometry.",
        "generator": "synthetic_fault_generator.py",
        "seed": seed,
        "shape": list(shape),
        "n_samples": n_samples,
        "splits": {
            "train": n_train,
            "val": n_val,
            "test": n_samples - n_train - n_val,
        },
        "scientific_note": (
            "Fault labels are generated directly from analytic fault planes with finite thickness. "
            "They are not obtained by thresholding seismic amplitude gradients."
        ),
        "samples": samples,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_shape(value):
    parts = [int(v) for v in value.lower().replace("x", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must contain three integers, e.g. 128,128,128")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic 3D seismic amplitude/fault-label pairs.")
    parser.add_argument("--output", default="processed_data/synthetic_fault_v2")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--shape", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--train", type=int, default=8)
    parser.add_argument("--val", type=int, default=2)
    parser.add_argument("--qc-count", type=int, default=6)
    args = parser.parse_args()

    if args.train + args.val > args.n_samples:
        raise ValueError("--train + --val must be <= --n-samples")

    generate_dataset(
        output_dir=args.output,
        n_samples=args.n_samples,
        shape=args.shape,
        seed=args.seed,
        n_train=args.train,
        n_val=args.val,
        qc_count=args.qc_count,
    )


if __name__ == "__main__":
    main()
