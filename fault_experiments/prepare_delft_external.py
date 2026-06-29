import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fault_experiments.cbvs import clean_component, open_regular_cbvs


ROOT = Path(__file__).resolve().parents[1]
SURVEY_ROOT = ROOT / "Delft"
OUTPUT = ROOT / "processed_data/delft_external_center"
SURVEY_SHAPE = (451, 469, 951)
ROI_SHAPE = (384, 384, 128)
ROI_START = tuple((full - roi) // 2 for full, roi in zip(SURVEY_SHAPE, ROI_SHAPE))


def robust_normalize(volume):
    low, high = np.percentile(volume[np.isfinite(volume)], (1.0, 99.0))
    normalized = np.clip((volume - low) / max(high - low, 1.0e-8), 0.0, 1.0)
    return normalized.astype(np.float32), float(low), float(high)


def plot_qc(amplitude, tfl, output):
    mids = tuple(size // 2 for size in amplitude.shape)
    views = [
        ("Central inline", amplitude[mids[0]].T, tfl[mids[0]].T),
        ("Central crossline", amplitude[:, mids[1], :].T, tfl[:, mids[1], :].T),
        ("Central time slice", amplitude[:, :, mids[2]], tfl[:, :, mids[2]]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10), dpi=180)
    fig.patch.set_facecolor("#FCFCFD")
    for row, (name, seismic, attribute) in enumerate(views):
        axes[row, 0].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name}: amplitude")
        axes[row, 1].imshow(attribute, cmap="inferno", aspect="auto", vmin=0, vmax=1)
        axes[row, 1].set_title("Thinned fault likelihood")
        axes[row, 2].imshow(seismic, cmap="gray", aspect="auto", vmin=0, vmax=1)
        overlay = np.ma.masked_where(attribute < 0.2, attribute)
        axes[row, 2].imshow(overlay, cmap="autumn", aspect="auto", vmin=0.2, vmax=1, alpha=0.75)
        axes[row, 2].set_title("Attribute overlay (>= 0.20)")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Delft fixed-center ROI geometry check\nTraditional TFL is a comparator, not expert ground truth",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    amplitude_source, amplitude_header = open_regular_cbvs(
        SURVEY_ROOT / "Seismics/Seismic.cbvs", SURVEY_SHAPE, components=1
    )
    tfl_source, tfl_header = open_regular_cbvs(
        SURVEY_ROOT / "Seismics/TFL.cbvs", SURVEY_SHAPE, components=3
    )
    slices = tuple(slice(start, start + size) for start, size in zip(ROI_START, ROI_SHAPE))
    amplitude_raw = clean_component(amplitude_source[slices[0], slices[1], 0, slices[2]])
    if not np.isfinite(amplitude_raw).all():
        raise ValueError("The fixed Delft ROI contains undefined seismic samples")
    amplitude_norm, clip_low, clip_high = robust_normalize(amplitude_raw)
    tfl = clean_component(tfl_source[slices[0], slices[1], 0, slices[2]], undefined_fill=0.0)
    tfl = np.clip(tfl, 0.0, 1.0).astype(np.float32)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT / "amplitude_norm.npy", amplitude_norm)
    np.save(OUTPUT / "amplitude_raw.npy", amplitude_raw.astype(np.float32))
    np.save(OUTPUT / "tfl_thinned.npy", tfl)
    metadata = {
        "id": "delft_external_center_384x384x128",
        "dataset": "Delft 3D survey",
        "license_note": "Survey metadata states CC BY-SA 3.0 via Terranubis",
        "source_shape": list(SURVEY_SHAPE),
        "starts": list(ROI_START),
        "shape": list(ROI_SHAPE),
        "ends": [start + size for start, size in zip(ROI_START, ROI_SHAPE)],
        "coordinates": {
            "inline": [2500 + ROI_START[0], 2500 + ROI_START[0] + ROI_SHAPE[0] - 1],
            "crossline": [3139 + ROI_START[1], 3139 + ROI_START[1] + ROI_SHAPE[1] - 1],
            "time_seconds": [ROI_START[2] * 0.004, (ROI_START[2] + ROI_SHAPE[2] - 1) * 0.004],
        },
        "selection_policy": "fixed geometric center on all three survey axes; selected before model inference",
        "normalization": {
            "method": "ROI 1st/99th percentile clipping mapped to [0,1]",
            "clip_low": clip_low,
            "clip_high": clip_high,
        },
        "cbvs_validation": {
            "amplitude_header_bytes": amplitude_header,
            "tfl_header_bytes": tfl_header,
            "record_layout": "20-byte trace header followed by little-endian float32 samples",
        },
        "tfl_role": "traditional thinned fault-likelihood comparator; not expert ground truth",
        "amplitude_summary": {
            "min": float(amplitude_raw.min()),
            "max": float(amplitude_raw.max()),
            "mean": float(amplitude_raw.mean()),
            "std": float(amplitude_raw.std()),
        },
        "tfl_summary": {
            "nonzero_fraction": float((tfl > 0).mean()),
            "fraction_ge_0_2": float((tfl >= 0.2).mean()),
            "max": float(tfl.max()),
        },
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_qc(amplitude_norm, tfl, OUTPUT / "geometry_qc.png")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
