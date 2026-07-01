import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fault_experiments.visualize_true_3d_fault_surfaces import render_surface_panels


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "processed_data" / "smeaheia" / "expert_roi_384x512x640"
RUN = ROOT / "runs" / "smeaheia_frozen_external"
OUTPUT = RUN / "figures"

METHODS = [
    ("Expert interpretation", None, None, "#E73B3B"),
    ("Dip-steered coherence", "dip_steered_coherence", 0.02, "#E08A2E"),
    ("U-Net", "unet", 0.50, "#4C78A8"),
    ("Hybrid DSA", "hybrid_dsa", 0.15, "#F2C14E"),
    ("SwinUNETR", "swinunetr_f3chain", 0.40, "#59A14F"),
]


def load_results():
    result = json.loads((RUN / "results.json").read_text(encoding="utf-8"))
    return {row["model"]: row for row in result["summaries"]}


def load_masks():
    expert = np.load(DATA / "expert_centreline.npy", mmap_mode="r").astype(bool)
    masks = {"Expert interpretation": expert}
    for name, directory, _, _ in METHODS[1:]:
        masks[name] = np.load(RUN / directory / "fault_binary.npy", mmap_mode="r").astype(bool)
    return masks


def overlay(axis, amplitude, mask, color, title):
    axis.imshow(amplitude, cmap="gray", aspect="auto", vmin=0, vmax=1)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgb = matplotlib.colors.to_rgb(color)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.float32) * 0.72
    axis.imshow(rgba, aspect="auto")
    axis.set_title(title, fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])


def make_section_figure(amplitude, masks, summaries):
    expert = masks["Expert interpretation"]
    inline = int(np.argmax(expert.sum(axis=(1, 2))))
    sample = int(np.argmax(expert.sum(axis=(0, 1))))
    figure, axes = plt.subplots(2, len(METHODS), figsize=(17, 7.6), dpi=200)
    for column, (name, directory, threshold, color) in enumerate(METHODS):
        if directory is None:
            subtitle = f"{name}\nreleased expert sticks"
        else:
            summary = summaries[directory]
            subtitle = (
                f"{name} | t={threshold:.2f}\n"
                f"Dice {summary['dice']:.3f} | tol. {summary['tolerant_dice_3px']:.3f}"
            )
        overlay(
            axes[0, column],
            np.asarray(amplitude[inline]).T,
            np.asarray(masks[name][inline]).T,
            color,
            subtitle,
        )
        overlay(
            axes[1, column],
            np.asarray(amplitude[:, :, sample]),
            np.asarray(masks[name][:, :, sample]),
            color,
            f"Time slice {sample}",
        )
    axes[0, 0].set_ylabel(f"Expert-rich inline {inline}", fontsize=10)
    axes[1, 0].set_ylabel(f"Expert-rich time sample {sample}", fontsize=10)
    figure.suptitle(
        "Smeaheia GN1101: independent sparse expert interpretation and frozen comparisons",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output = OUTPUT / "smeaheia_expert_method_sections.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return {
        "figure": str(output.relative_to(ROOT)),
        "inline_local_index": inline,
        "time_sample_local_index": sample,
        "selection": "maximum expert-centreline voxel count; selected without model predictions",
    }


def make_surface_figure(amplitude, masks, summaries, validity):
    panels = []
    for name, directory, threshold, color in METHODS:
        if directory is None:
            subtitle = "Released 3D expert fault sticks"
        else:
            summary = summaries[directory]
            subtitle = f"Frozen t={threshold:.2f} | Dice={summary['dice']:.3f}"
        display_mask = masks[name] if directory is None else np.logical_and(masks[name], validity)
        if directory is not None:
            subtitle += " | expert validity domain"
        panels.append((name, display_mask, color, subtitle))
    output = OUTPUT / "smeaheia_true_3d_fault_surfaces.png"
    panel_audit = render_surface_panels(
        np.asarray(amplitude, dtype=np.float32),
        panels,
        output,
        OUTPUT / "surface_meshes",
        factors=(2, 2, 2),
        minimum_component_voxels=8,
        crossline_exaggeration=1.0,
        maximum_faces=160_000,
    )
    return {
        "figure": str(output.relative_to(ROOT)),
        "display_processing_only": (
            "Traditional and network masks are first restricted to the same released-expert validity "
            "domain used by the metrics. Each displayed mask is then thinned on original expert-inline "
            "sections, max-pooled by 2x2x2, filtered below eight display voxels and mesh-decimated. "
            "Metrics use unmodified arrays. Full prediction extent remains visible in the section figure."
        ),
        "panels": panel_audit,
    }


def make_metric_figure(summaries):
    order = ["dip_steered_coherence", "unet", "hybrid_dsa", "swinunetr_f3chain"]
    labels = ["Dip-steered\ncoherence", "U-Net", "Hybrid DSA", "SwinUNETR"]
    colors = ["#E08A2E", "#4C78A8", "#F2C14E", "#59A14F"]
    metrics = [
        ("dice", "Exact Dice"),
        ("tolerant_dice_3px", "3-pixel tolerant Dice"),
        ("histogram_auprc_400_bins", "AUPRC"),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), dpi=200)
    x = np.arange(len(order))
    for axis, (field, title) in zip(axes, metrics):
        values = [summaries[name][field] for name in order]
        bars = axis.bar(x, values, color=colors, width=0.68, edgecolor="#30343B", linewidth=0.5)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, max(values) * 1.25)
        axis.set_title(title)
        axis.grid(axis="y", color="#D8DADF", linewidth=0.6)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.035,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle("Smeaheia GN1101 independent sparse-expert validation")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = OUTPUT / "smeaheia_metric_comparison.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    return str(output.relative_to(ROOT))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    amplitude = np.load(DATA / "amplitude_norm.npy", mmap_mode="r")
    validity = np.load(DATA / "validity_mask.npy", mmap_mode="r").astype(bool)
    masks = load_masks()
    summaries = load_results()
    result = {
        "survey": "Smeaheia GN1101",
        "evidence_role": "independent sparse expert 3D validation",
        "section_comparison": make_section_figure(amplitude, masks, summaries),
        "surface_comparison": make_surface_figure(amplitude, masks, summaries, validity),
        "metric_comparison": make_metric_figure(summaries),
    }
    (OUTPUT / "smeaheia_comparison_audit.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
