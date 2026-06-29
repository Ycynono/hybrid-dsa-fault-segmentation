import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/delft_external_center"
RUN_ROOT = ROOT / "runs/delft_frozen_external"
OUTPUT_DIR = RUN_ROOT / "figures"
TFL_THRESHOLD = 0.20
MODELS = [
    ("U-Net", "unet", 0.50),
    ("Hybrid DSA", "hybrid_dsa", 0.15),
    ("SwinUNETR", "swinunetr_f3chain", 0.40),
]
TOKENS = {"surface": "#FCFCFD", "ink": "#1F2430", "muted": "#6F768A"}


def load_inputs():
    amplitude = np.load(DATA_DIR / "amplitude_norm.npy", mmap_mode="r")
    tfl = np.load(DATA_DIR / "tfl_thinned.npy", mmap_mode="r")
    probabilities = {
        name: np.load(RUN_ROOT / directory / "fault_probability.npy", mmap_mode="r")
        for name, directory, _ in MODELS
    }
    metadata = {
        name: json.loads((RUN_ROOT / directory / "inference_metadata.json").read_text(encoding="utf-8"))
        for name, directory, _ in MODELS
    }
    return amplitude, tfl, probabilities, metadata


def agreement_metrics(prediction, reference):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    tp = np.logical_and(prediction, reference).sum()
    exact_dice = 2.0 * tp / max(prediction.sum() + reference.sum(), 1)
    structure = ndimage.generate_binary_structure(3, 1)
    ref_dilated = ndimage.binary_dilation(reference, structure=structure, iterations=3)
    pred_dilated = ndimage.binary_dilation(prediction, structure=structure, iterations=3)
    tolerant_precision = np.logical_and(prediction, ref_dilated).sum() / max(prediction.sum(), 1)
    tolerant_recall = np.logical_and(reference, pred_dilated).sum() / max(reference.sum(), 1)
    tolerant_dice = (
        2.0 * tolerant_precision * tolerant_recall
        / max(tolerant_precision + tolerant_recall, 1.0e-12)
    )
    return {
        "exact_dice_to_tfl": float(exact_dice),
        "tolerant_precision_to_tfl": float(tolerant_precision),
        "tolerant_recall_to_tfl": float(tolerant_recall),
        "tolerant_dice_to_tfl": float(tolerant_dice),
    }


def rgba_overlay(seismic, mask, color):
    gray = np.clip(seismic, 0, 1)
    rgba = np.empty((*gray.shape, 4), dtype=np.float32)
    rgba[..., :3] = gray[..., None]
    rgba[..., 3] = 1.0
    rgba[mask, :3] = np.asarray(color, dtype=np.float32)
    return rgba


def plot_sections(amplitude, tfl, probabilities, metadata, agreements):
    mids = tuple(size // 2 for size in amplitude.shape)
    views = [
        ("Central inline", np.asarray(amplitude[mids[0]]).T, lambda p: np.asarray(p[mids[0]]).T),
        ("Central crossline", np.asarray(amplitude[:, mids[1], :]).T, lambda p: np.asarray(p[:, mids[1], :]).T),
        ("Central time slice", np.asarray(amplitude[:, :, mids[2]]), lambda p: np.asarray(p[:, :, mids[2]])),
    ]
    fig, axes = plt.subplots(3, 5, figsize=(16.2, 10.2), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    titles = ["Seismic amplitude", "Traditional TFL", *[name for name, _, _ in MODELS]]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10.5, color=TOKENS["ink"], pad=7)
    for row, (view_name, seismic, extract) in enumerate(views):
        axes[row, 0].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(view_name, fontsize=9, color=TOKENS["ink"])
        tfl_mask = extract(tfl) >= TFL_THRESHOLD
        axes[row, 1].imshow(rgba_overlay(seismic, tfl_mask, (0.74, 0.34, 0.61)), aspect="auto")
        for column, (name, _, threshold) in enumerate(MODELS, start=2):
            prediction = extract(probabilities[name]) >= threshold
            axes[row, column].imshow(
                rgba_overlay(seismic, prediction, (0.94, 0.44, 0.22)), aspect="auto"
            )
            if row == 0:
                occupancy = metadata[name]["statistics"]["predicted_voxel_fraction"] * 100.0
                axes[row, column].text(
                    0.02,
                    0.02,
                    f"volume={occupancy:.2f}%\nTFL TD={agreements[name]['tolerant_dice_to_tfl']:.3f}",
                    transform=axes[row, column].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.2,
                    color="white",
                    bbox={"facecolor": "#1F2430", "alpha": 0.80, "pad": 2, "edgecolor": "none"},
                )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.text(
        0.045,
        0.985,
        "Frozen networks over-activate differently on the independent Delft survey",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.045,
        0.956,
        "Fixed central ROI and Thebe-validation thresholds. Pink = OpendTect thinned fault likelihood; "
        "orange = network prediction. TFL agreement is diagnostic only, not expert-labeled accuracy.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.035, top=0.91, wspace=0.035, hspace=0.08)
    fig.savefig(OUTPUT_DIR / "delft_tfl_frozen_central_sections.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / "delft_tfl_frozen_central_sections.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def add_orthogonal_planes(ax, amplitude, mask=None, color=(0.94, 0.44, 0.22)):
    ni, nx, nt = amplitude.shape
    im, xm, tm = ni // 2, nx // 2, nt // 2
    ii, xx, tt = np.arange(0, ni, 4), np.arange(0, nx, 4), np.arange(0, nt, 2)
    x_grid, i_grid = np.meshgrid(xx, ii)
    plane_mask = None if mask is None else np.asarray(mask[np.ix_(ii, xx, [tm])])[:, :, 0]
    ax.plot_surface(
        x_grid, i_grid, np.full_like(x_grid, tm),
        facecolors=rgba_overlay(np.asarray(amplitude[np.ix_(ii, xx, [tm])])[:, :, 0], plane_mask if plane_mask is not None else np.zeros_like(x_grid, dtype=bool), color),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    x_grid, t_grid = np.meshgrid(xx, tt)
    plane_mask = None if mask is None else np.asarray(mask[np.ix_([im], xx, tt)])[0].T
    ax.plot_surface(
        x_grid, np.full_like(x_grid, im), t_grid,
        facecolors=rgba_overlay(np.asarray(amplitude[np.ix_([im], xx, tt)])[0].T, plane_mask if plane_mask is not None else np.zeros_like(x_grid, dtype=bool), color),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    i_grid, t_grid = np.meshgrid(ii, tt)
    plane_mask = None if mask is None else np.asarray(mask[np.ix_(ii, [xm], tt)])[:, 0, :].T
    ax.plot_surface(
        np.full_like(i_grid, xm), i_grid, t_grid,
        facecolors=rgba_overlay(np.asarray(amplitude[np.ix_(ii, [xm], tt)])[:, 0, :].T, plane_mask if plane_mask is not None else np.zeros_like(i_grid, dtype=bool), color),
        rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False,
    )
    ax.set_xlim(0, nx - 1)
    ax.set_ylim(0, ni - 1)
    ax.set_zlim(nt - 1, 0)
    ax.set_box_aspect((nx, ni, nt * 1.5))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("Crossline", fontsize=6.5)
    ax.set_ylabel("Inline", fontsize=6.5)
    ax.set_zlabel("Sample", fontsize=6.5)
    ax.tick_params(labelsize=5.5, pad=0)


def plot_3d(amplitude, tfl, probabilities, metadata):
    fig = plt.figure(figsize=(17.5, 5.2), dpi=220)
    fig.patch.set_facecolor(TOKENS["surface"])
    panels = [
        ("Seismic amplitude", None, (0.94, 0.44, 0.22)),
        ("Traditional TFL", tfl >= TFL_THRESHOLD, (0.74, 0.34, 0.61)),
        *[(name, probabilities[name] >= threshold, (0.94, 0.44, 0.22)) for name, _, threshold in MODELS],
    ]
    for index, (name, mask, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 5, index, projection="3d")
        add_orthogonal_planes(ax, amplitude, mask, color)
        title = name
        if name in metadata:
            title += f"\nPredicted volume {metadata[name]['statistics']['predicted_voxel_fraction'] * 100:.2f}%"
        ax.set_title(title, fontsize=9, color=TOKENS["ink"], pad=8)
    fig.text(
        0.025,
        0.985,
        "Delft field stress test reveals survey-dependent over-segmentation",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.025,
        0.943,
        "Identical central orthogonal planes, camera, frozen weights, and Thebe-validation thresholds; TFL is not ground truth.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.87, wspace=0.01)
    fig.savefig(OUTPUT_DIR / "delft_tfl_frozen_orthogonal_3d.png", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(OUTPUT_DIR / "delft_tfl_frozen_orthogonal_3d.svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    amplitude, tfl, probabilities, metadata = load_inputs()
    reference = np.asarray(tfl) >= TFL_THRESHOLD
    agreements = {
        name: agreement_metrics(np.asarray(probabilities[name]) >= threshold, reference)
        for name, _, threshold in MODELS
    }
    plot_sections(amplitude, tfl, probabilities, metadata, agreements)
    plot_3d(amplitude, tfl, probabilities, metadata)
    result = {
        "evidence_role": "qualitative independent-survey stress test with traditional TFL comparator; no expert labels",
        "slice_policy": "fixed geometric center on all three axes",
        "tfl_threshold": TFL_THRESHOLD,
        "tfl_fraction": float(reference.mean()),
        "models": {
            name: {
                "threshold": threshold,
                "predicted_fraction": metadata[name]["statistics"]["predicted_voxel_fraction"],
                "largest_component_fraction": metadata[name]["statistics"]["largest_component_fraction_of_prediction"],
                **agreements[name],
            }
            for name, _, threshold in MODELS
        },
        "claim_warning": "Agreement with TFL is not accuracy because TFL is an algorithmic attribute, not an independent interpretation.",
    }
    (RUN_ROOT / "delft_attribute_agreement.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
