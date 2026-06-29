import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/real_subvolumes/force_field_mid_384x512x128"
RUN_ROOT = ROOT / "runs/force_frozen_external"
OUTPUT_DIR = RUN_ROOT / "figures"
MODELS = [
    ("U-Net", "unet", 0.50),
    ("Hybrid DSA", "hybrid_dsa", 0.15),
    ("SwinUNETR", "swinunetr_f3chain", 0.40),
]
TOKENS = {"surface": "#FCFCFD", "ink": "#1F2430", "muted": "#6F768A"}


def load_inputs():
    amplitude = np.load(DATA_DIR / "amplitude_norm.npy", mmap_mode="r")
    probabilities = {
        name: np.load(RUN_ROOT / directory / "fault_probability.npy", mmap_mode="r")
        for name, directory, _ in MODELS
    }
    metadata = {
        name: json.loads((RUN_ROOT / directory / "inference_metadata.json").read_text(encoding="utf-8"))
        for name, directory, _ in MODELS
    }
    return amplitude, probabilities, metadata


def overlay(ax, seismic, probability, threshold):
    ax.imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
    prediction = probability >= threshold
    rgba = np.zeros((*prediction.shape, 4), dtype=np.float32)
    rgba[prediction] = np.array([0.94, 0.44, 0.22, 0.76], dtype=np.float32)
    ax.imshow(rgba, aspect="auto")


def plot_sections(amplitude, probabilities, metadata):
    mids = tuple(size // 2 for size in amplitude.shape)
    views = [
        ("Central inline", np.asarray(amplitude[mids[0], :, :]).T, lambda p: np.asarray(p[mids[0], :, :]).T),
        ("Central crossline", np.asarray(amplitude[:, mids[1], :]).T, lambda p: np.asarray(p[:, mids[1], :]).T),
        ("Central time slice", np.asarray(amplitude[:, :, mids[2]]), lambda p: np.asarray(p[:, :, mids[2]])),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14.8, 10.4), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    titles = ["Seismic amplitude", *[name for name, _, _ in MODELS]]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10.5, color=TOKENS["ink"], pad=7)
    for row, (view_name, seismic, extract) in enumerate(views):
        axes[row, 0].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(view_name, fontsize=9, color=TOKENS["ink"])
        for column, (name, _, threshold) in enumerate(MODELS, start=1):
            overlay(axes[row, column], seismic, extract(probabilities[name]), threshold)
            occupancy = metadata[name]["statistics"]["predicted_voxel_fraction"] * 100.0
            if row == 0:
                axes[row, column].text(
                    0.02,
                    0.02,
                    f"threshold={threshold:.2f}; volume={occupancy:.4f}%",
                    transform=axes[row, column].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.2,
                    color="white",
                    bbox={"facecolor": "#1F2430", "alpha": 0.78, "pad": 2, "edgecolor": "none"},
                )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.text(
        0.055,
        0.985,
        "Frozen F3-adapted models do not transfer consistently to the FORCE survey",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.055,
        0.956,
        "Independent FORCE field volume; fixed central slices and Thebe-validation thresholds. "
        "Orange = predicted fault. No FORCE interpretation was used, so this is qualitative external-survey evidence.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.035, top=0.91, wspace=0.035, hspace=0.08)
    fig.savefig(OUTPUT_DIR / "force_frozen_central_sections.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / "force_frozen_central_sections.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plane_rgba(seismic, probability=None, threshold=None):
    gray = np.clip(seismic, 0, 1)
    rgba = np.empty((*gray.shape, 4), dtype=np.float32)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 1.0
    if probability is not None:
        mask = probability >= threshold
        rgba[mask, :3] = np.array([0.94, 0.44, 0.22], dtype=np.float32)
    return rgba


def add_orthogonal_planes(ax, amplitude, probability=None, threshold=None):
    ni, nx, nt = amplitude.shape
    im, xm, tm = ni // 2, nx // 2, nt // 2
    step_i, step_x, step_t = 4, 4, 2

    ii = np.arange(0, ni, step_i)
    xx = np.arange(0, nx, step_x)
    tt = np.arange(0, nt, step_t)
    x_grid, i_grid = np.meshgrid(xx, ii)
    p = None if probability is None else np.asarray(probability[np.ix_(ii, xx, [tm])])[:, :, 0]
    ax.plot_surface(
        x_grid, i_grid, np.full_like(x_grid, tm), facecolors=plane_rgba(np.asarray(amplitude[np.ix_(ii, xx, [tm])])[:, :, 0], p, threshold),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    x_grid, t_grid = np.meshgrid(xx, tt)
    p = None if probability is None else np.asarray(probability[np.ix_([im], xx, tt)])[0].T
    ax.plot_surface(
        x_grid, np.full_like(x_grid, im), t_grid, facecolors=plane_rgba(np.asarray(amplitude[np.ix_([im], xx, tt)])[0].T, p, threshold),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    i_grid, t_grid = np.meshgrid(ii, tt)
    p = None if probability is None else np.asarray(probability[np.ix_(ii, [xm], tt)])[:, 0, :].T
    ax.plot_surface(
        np.full_like(i_grid, xm), i_grid, t_grid, facecolors=plane_rgba(np.asarray(amplitude[np.ix_(ii, [xm], tt)])[:, 0, :].T, p, threshold),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ni - 1)
    ax.set_zlim(nt - 1, 0)
    ax.set_box_aspect((nx, ni, nt * 1.5))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("Axis 1", fontsize=7)
    ax.set_ylabel("Axis 0", fontsize=7)
    ax.set_zlabel("Sample", fontsize=7)
    ax.tick_params(labelsize=6, pad=0)


def plot_3d(amplitude, probabilities, metadata):
    fig = plt.figure(figsize=(16, 5.3), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    panels = [("Seismic amplitude", None, None), *[(name, probabilities[name], threshold) for name, _, threshold in MODELS]]
    for index, (name, probability, threshold) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 4, index, projection="3d")
        add_orthogonal_planes(ax, amplitude, probability, threshold)
        title = name
        if probability is not None:
            occupancy = metadata[name]["statistics"]["predicted_voxel_fraction"] * 100.0
            title += f"\nPredicted volume {occupancy:.4f}%"
        ax.set_title(title, fontsize=9.5, color=TOKENS["ink"], pad=8)
    fig.text(
        0.035,
        0.985,
        "Independent-survey stress test exposes strong model-dependent domain shift",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.035,
        0.943,
        "FORCE field subvolume; identical central orthogonal planes, camera, frozen weights, and validation-only thresholds.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.88, wspace=0.02)
    fig.savefig(OUTPUT_DIR / "force_frozen_orthogonal_3d.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / "force_frozen_orthogonal_3d.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    amplitude, probabilities, metadata = load_inputs()
    plot_sections(amplitude, probabilities, metadata)
    plot_3d(amplitude, probabilities, metadata)
    (OUTPUT_DIR / "visualization_metadata.json").write_text(
        json.dumps(
            {
                "slice_policy": "fixed geometric center on all three axes",
                "comparison_policy": "identical slices, camera, frozen weights, and Thebe-validation thresholds",
                "evidence_role": "qualitative independent-survey stress test; no FORCE expert labels",
                "occupancy_percent": {
                    name: metadata[name]["statistics"]["predicted_voxel_fraction"] * 100.0
                    for name, _, _ in MODELS
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
