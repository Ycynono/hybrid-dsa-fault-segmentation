import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "processed_data/cracks_external_v1"
RUN_ROOT = ROOT / "runs/cracks_audit_frozen"
MODELS = [
    ("U-Net", "unet", 0.50),
    ("Hybrid DSA", "hybrid_dsa", 0.15),
    ("SwinUNETR F3-chain", "swinunetr_f3chain", 0.40),
]


def main():
    split = json.loads((DATA_DIR / "expert_split.json").read_text(encoding="utf-8"))
    sections = split["audit_sections"]
    amplitude = np.load(DATA_DIR / "amplitude_01.npy", mmap_mode="r")
    masks = np.load(DATA_DIR / "audit_expert_fault_masks.npy", mmap_mode="r")
    densities = masks.mean(axis=(1, 2))
    position = int(np.argmin(np.abs(densities - np.median(densities))))
    section = sections[position]
    seismic = np.asarray(amplitude[section - 1]).T
    truth = np.asarray(masks[position]).T
    probabilities = {
        name: np.load(RUN_ROOT / directory / "fault_probability_float16.npy", mmap_mode="r")
        for name, directory, _ in MODELS
    }
    fig, axes = plt.subplots(1, 5, figsize=(17, 6), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("#FCFCFD")
    axes[0].imshow(seismic, cmap="seismic", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Decoded seismic")
    axes[1].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(~truth, truth), cmap="spring", alpha=0.82, aspect="auto")
    axes[1].set_title("Independent expert")
    for ax, (name, _, threshold) in zip(axes[2:], MODELS):
        prediction = np.asarray(probabilities[name][section - 1]) >= threshold
        ax.imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.imshow(
            np.ma.masked_where(~prediction.T, prediction.T),
            cmap="autumn",
            alpha=0.72,
            aspect="auto",
        )
        ax.set_title(name)
    for ax in axes:
        ax.set_xlabel("Trace index")
        ax.set_ylabel("Depth pixel")
    fig.suptitle(
        f"CRACKS audit section {section}: prespecified median expert-label density\nPink = expert; orange = frozen model prediction",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output_dir = RUN_ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"cracks_audit_section_{section:03d}_comparison.png"
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_policy": "audit section closest to median expert fault-pixel density",
                "section": section,
                "fault_fraction": float(densities[position]),
                "reserve_accessed": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
