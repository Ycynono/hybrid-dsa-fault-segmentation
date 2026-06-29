import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
SEGY_PATH = ROOT / "F3" / "Export" / "amplitude.sgy"
AMPLITUDE_PATH = ROOT / "F3" / "Export" / "amplitude.npy"
FAULT_PATH = ROOT / "F3" / "Rawdata" / "Faults" / "FaultA.txt"
OUTPUT_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"


def trace_xy(header):
    scalar = header[segyio.TraceField.SourceGroupScalar]
    scale = 1.0 / abs(scalar) if scalar < 0 else float(scalar or 1)
    return np.array(
        [header[segyio.TraceField.CDP_X] * scale, header[segyio.TraceField.CDP_Y] * scale],
        dtype=np.float64,
    )


def geometry_transform():
    with segyio.open(str(SEGY_PATH), "r", ignore_geometry=True) as seismic:
        xline_count = 951
        origin = trace_xy(seismic.header[0])
        inline_step = trace_xy(seismic.header[xline_count]) - origin
        xline_step = trace_xy(seismic.header[xline_count - 1]) - origin
        xline_step /= xline_count - 1
        matrix = np.column_stack([inline_step, xline_step])
        return origin, matrix, float(segyio.tools.dt(seismic) / 1000.0), float(seismic.samples[0])


def normalize01(section):
    finite = section[np.isfinite(section)]
    lo, hi = np.percentile(finite, [1, 99])
    return np.clip((section - lo) / max(hi - lo, 1e-8), 0, 1).astype(np.float32)


def rasterize_stick(xline_indices, sample_indices, shape, line_radius=1, corridor_radius=16):
    order = np.argsort(sample_indices)
    samples = sample_indices[order]
    xlines = xline_indices[order]
    sample_start = int(np.ceil(samples.min()))
    sample_end = int(np.floor(samples.max()))
    sample_grid = np.arange(sample_start, sample_end + 1)
    xline_grid = np.interp(sample_grid, samples, xlines)
    label = np.zeros(shape, dtype=np.uint8)
    validity = np.zeros(shape, dtype=np.uint8)
    for sample, xline in zip(sample_grid, xline_grid):
        center = int(round(xline))
        lo = max(0, center - line_radius)
        hi = min(shape[0], center + line_radius + 1)
        label[lo:hi, sample] = 1
        vlo = max(0, center - corridor_radius)
        vhi = min(shape[0], center + corridor_radius + 1)
        validity[vlo:vhi, sample] = 1
    return label, validity


def plot_case(amplitude, label, validity, output, title):
    amp = normalize01(amplitude).T
    lbl = label.T
    valid = validity.T
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), dpi=180)
    axes[0].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Amplitude")
    axes[1].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[1].imshow(np.ma.masked_where(lbl == 0, lbl), cmap="autumn", aspect="auto", vmin=0, vmax=1)
    axes[1].set_title("Independent FaultA picks")
    axes[2].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[2].imshow(np.ma.masked_where(valid == 0, valid), cmap="Blues", alpha=0.25, aspect="auto")
    axes[2].imshow(np.ma.masked_where(lbl == 0, lbl), cmap="autumn", aspect="auto", vmin=0, vmax=1)
    axes[2].set_title("Evaluation corridor")
    for ax in axes:
        ax.set_xlabel("Crossline index")
        ax.set_ylabel("Time sample")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def split_for_stick(stick_id):
    if stick_id <= 5:
        return "train"
    if stick_id == 6:
        return "buffer_train_val"
    if stick_id <= 8:
        return "val"
    if stick_id == 9:
        return "buffer_val_test"
    return "test"


def prepare_inference_volume(amplitude):
    starts = [0, 512, 0]
    shape = [256, 256, amplitude.shape[2]]
    raw = np.asarray(amplitude[0:256, 512:768, :], dtype=np.float32)
    finite = raw[np.isfinite(raw)]
    lo, hi = np.percentile(finite, [1, 99])
    normalized = np.clip((raw - lo) / max(hi - lo, 1e-8), 0, 1)
    volume_dir = OUTPUT_ROOT / "inference_volume"
    volume_dir.mkdir(parents=True, exist_ok=True)
    np.save(volume_dir / "amplitude_norm.npy", normalized.astype(np.float16))
    metadata = {
        "id": "f3_faulta_256x256x463",
        "dataset": "F3",
        "source": str(AMPLITUDE_PATH.relative_to(ROOT)),
        "source_shape": list(amplitude.shape),
        "starts": starts,
        "shape": shape,
        "ends": [start + size for start, size in zip(starts, shape)],
        "axis_names": ["inline_index", "crossline_index", "time_index"],
        "coordinate_note": "inline=index+100, crossline=index+300, time_ms=index*4",
        "normalization": {
            "method": "robust percentile clipping to [0,1]",
            "low_percentile": 1.0,
            "high_percentile": 99.0,
            "clip_low_value": float(lo),
            "clip_high_value": float(hi),
        },
        "purpose": "Inference volume covering the independently interpreted F3 FaultA surface.",
        "publication_warning": "FaultA labels are stored separately and are never model inputs.",
    }
    (volume_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    origin, matrix, sample_step_ms, sample_start_ms = geometry_transform()
    inverse = np.linalg.inv(matrix)
    points = np.loadtxt(FAULT_PATH)
    grid_offsets = (points[:, :2] - origin) @ inverse.T
    inline_indices = grid_offsets[:, 0]
    xline_indices = grid_offsets[:, 1]
    sample_indices = (points[:, 2] - sample_start_ms) / sample_step_ms
    amplitude = np.load(AMPLITUDE_PATH, mmap_mode="r")
    cases = []

    for stick_value in sorted(np.unique(points[:, 3])):
        stick_id = int(stick_value)
        mask = points[:, 3] == stick_value
        inline_index = int(round(float(np.median(inline_indices[mask]))))
        label, validity = rasterize_stick(
            xline_indices[mask], sample_indices[mask], (amplitude.shape[1], amplitude.shape[2])
        )
        section = np.asarray(amplitude[inline_index], dtype=np.float32)
        split = split_for_stick(stick_id)
        case_id = f"faulta_stick_{stick_id:02d}_il_{inline_index + 100}"
        case_dir = OUTPUT_ROOT / "sticks" / split / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "amplitude.npy", normalize01(section).astype(np.float16))
        np.save(case_dir / "fault_label.npy", label)
        np.save(case_dir / "validity_mask.npy", validity)
        metadata = {
            "case_id": case_id,
            "stick_id": stick_id,
            "split": split,
            "inline_index": inline_index,
            "inline_coordinate": inline_index + 100,
            "picked_point_count": int(mask.sum()),
            "crossline_coordinate_range": [
                float(xline_indices[mask].min() + 300),
                float(xline_indices[mask].max() + 300),
            ],
            "time_ms_range": [float(points[mask, 2].min()), float(points[mask, 2].max())],
            "label_line_radius_samples": 1,
            "evaluation_corridor_radius_crosslines": 16,
            "label_source": "F3/Rawdata/Faults/FaultA.txt; independent interpretation",
            "warning": "Only the validity corridor is evaluated because other faults may be uninterpreted.",
        }
        (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        plot_case(section, label, validity, case_dir / "reference_qc.png", case_id)
        cases.append(metadata)

    inference_metadata = prepare_inference_volume(amplitude)
    manifest = {
        "purpose": "Quantitative F3 benchmark from independently interpreted FaultA sticks.",
        "source": str(FAULT_PATH.relative_to(ROOT)),
        "coordinate_transform": {
            "xy_origin_at_inline100_crossline300": origin.tolist(),
            "columns_inline_crossline_step_xy": matrix.tolist(),
            "sample_start_ms": sample_start_ms,
            "sample_step_ms": sample_step_ms,
        },
        "split_policy": {
            "train": "sticks 0-5",
            "train_val_buffer": "stick 6 excluded",
            "val": "sticks 7-8",
            "val_test_buffer": "stick 9 excluded",
            "test": "sticks 10-12",
        },
        "evaluation_policy": "Evaluate only within each stick's validity corridor.",
        "inference_volume": inference_metadata,
        "cases": cases,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["split_policy"], indent=2))
    print(f"Wrote {len(cases)} interpreted sticks to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
