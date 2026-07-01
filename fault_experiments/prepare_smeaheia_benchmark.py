import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import segyio
from numpy.lib.format import open_memmap
from scipy import ndimage

from fault_experiments.audit_smeaheia_dataset import (
    DEFAULT_STICKS,
    MISSING_LINE_NUMBER,
    parse_sticks,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEGY = (
    ROOT
    / "external_data"
    / "smeaheia"
    / "seismic_3d"
    / "Seismic_3D_Surveys"
    / "data"
    / "GN1101_Scaled(Realized)"
)
DEFAULT_OUTPUT = ROOT / "processed_data" / "smeaheia" / "expert_roi_384x512x640"
ROI_SHAPE = (384, 512, 640)
SELECTION_STRIDE = 32
LABEL_RADIUS = 1
VALIDITY_RADIUS = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare the frozen Smeaheia sparse expert benchmark.")
    parser.add_argument("--segy", type=Path, default=DEFAULT_SEGY)
    parser.add_argument("--sticks", type=Path, default=DEFAULT_STICKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def starts(length, window, stride):
    values = list(range(0, length - window + 1, stride))
    if values[-1] != length - window:
        values.append(length - window)
    return values


def nearest_value(values, target):
    position = int(np.searchsorted(values, target))
    candidates = []
    if position < len(values):
        candidates.append(values[position])
    if position > 0:
        candidates.append(values[position - 1])
    return int(min(candidates, key=lambda value: abs(value - target)))


def resolve_grid_coordinates(records, inline_values, crossline_values):
    valid = np.array(
        [
            record["inline"] != MISSING_LINE_NUMBER
            and record["crossline"] != MISSING_LINE_NUMBER
            for record in records
        ],
        dtype=bool,
    )
    design = np.array([[record["x"], record["y"], 1.0] for record in records], dtype=np.float64)
    targets = np.array([[record["inline"], record["crossline"]] for record in records], dtype=np.float64)
    coefficients = np.linalg.lstsq(design[valid], targets[valid], rcond=None)[0]
    predicted = design @ coefficients
    resolved = []
    for index, record in enumerate(records):
        item = dict(record)
        if valid[index]:
            item["resolved_inline"] = record["inline"]
            item["resolved_crossline"] = record["crossline"]
            item["grid_source"] = "provider_inline_crossline"
        else:
            item["resolved_inline"] = nearest_value(inline_values, predicted[index, 0])
            item["resolved_crossline"] = nearest_value(crossline_values, predicted[index, 1])
            item["grid_source"] = "provider_xy_affine_to_nearest_trace"
        resolved.append(item)
    fitted = design[valid] @ coefficients
    fit_residual = targets[valid] - fitted
    return resolved, {
        "method": "least-squares affine X/Y-to-inline/crossline fit using provider-indexed picks",
        "coefficients_columns_inline_crossline": coefficients.tolist(),
        "fit_residual_median_line_units": np.median(np.abs(fit_residual), axis=0).tolist(),
        "fit_residual_p95_line_units": np.percentile(np.abs(fit_residual), 95, axis=0).tolist(),
        "xy_only_point_count": int((~valid).sum()),
    }


def map_records(records, inline_values, crossline_values, samples):
    inline_lookup = {int(value): index for index, value in enumerate(inline_values)}
    crossline_lookup = {int(value): index for index, value in enumerate(crossline_values)}
    sample_step = float(np.median(np.diff(samples)))
    mapped = []
    for record in records:
        if record["resolved_inline"] not in inline_lookup:
            continue
        if record["resolved_crossline"] not in crossline_lookup:
            continue
        sample_index = int(round((record["z"] - samples[0]) / sample_step))
        if sample_index < 0 or sample_index >= len(samples):
            continue
        item = dict(record)
        item["inline_index"] = inline_lookup[record["resolved_inline"]]
        item["crossline_index"] = crossline_lookup[record["resolved_crossline"]]
        item["sample_index"] = sample_index
        mapped.append(item)
    return mapped


def select_roi(records, volume_shape):
    arrays = np.array(
        [[record["inline_index"], record["crossline_index"], record["sample_index"]] for record in records],
        dtype=np.int64,
    )
    best = None
    evaluated = 0
    for inline_start in starts(volume_shape[0], ROI_SHAPE[0], SELECTION_STRIDE):
        inline_mask = (arrays[:, 0] >= inline_start) & (arrays[:, 0] < inline_start + ROI_SHAPE[0])
        for crossline_start in starts(volume_shape[1], ROI_SHAPE[1], SELECTION_STRIDE):
            planar_mask = (
                inline_mask
                & (arrays[:, 1] >= crossline_start)
                & (arrays[:, 1] < crossline_start + ROI_SHAPE[1])
            )
            if not planar_mask.any():
                continue
            for sample_start in starts(volume_shape[2], ROI_SHAPE[2], SELECTION_STRIDE):
                mask = (
                    planar_mask
                    & (arrays[:, 2] >= sample_start)
                    & (arrays[:, 2] < sample_start + ROI_SHAPE[2])
                )
                evaluated += 1
                point_count = int(mask.sum())
                if point_count == 0:
                    continue
                fault_count = len({records[index]["fault"] for index in np.flatnonzero(mask)})
                stick_count = len(
                    {
                        (records[index]["fault"], records[index]["stick"])
                        for index in np.flatnonzero(mask)
                    }
                )
                score = (point_count, fault_count, stick_count)
                candidate_start = (inline_start, crossline_start, sample_start)
                if best is None or score > best["score"] or (
                    score == best["score"] and candidate_start < best["start"]
                ):
                    best = {"start": candidate_start, "score": score, "mask": mask.copy()}
    if best is None:
        raise RuntimeError("No expert picks overlap a valid ROI.")
    best["evaluated_window_count"] = evaluated
    return best


def line_voxels(point_a, point_b):
    point_a = np.asarray(point_a, dtype=np.float64)
    point_b = np.asarray(point_b, dtype=np.float64)
    steps = int(np.ceil(np.max(np.abs(point_b - point_a)))) + 1
    return np.rint(np.linspace(point_a, point_b, steps)).astype(np.int64)


def build_sparse_reference(records, roi_start, roi_shape):
    centreline = np.zeros(roi_shape, dtype=bool)
    groups = defaultdict(list)
    for record in records:
        groups[(record["fault"], record["stick"])].append(record)
    included_groups = Counter()
    included_points = Counter()
    start = np.asarray(roi_start, dtype=np.int64)
    end = start + np.asarray(roi_shape, dtype=np.int64)

    for (fault, stick), group in groups.items():
        group.sort(key=lambda record: record["line_number"])
        points = np.array(
            [[record["inline_index"], record["crossline_index"], record["sample_index"]] for record in group],
            dtype=np.int64,
        )
        group_contributed = False
        if len(points) == 1:
            segments = [points]
        else:
            segments = [line_voxels(points[index], points[index + 1]) for index in range(len(points) - 1)]
        for segment in segments:
            inside = np.all((segment >= start) & (segment < end), axis=1)
            local = segment[inside] - start
            if local.size:
                centreline[tuple(local.T)] = True
                included_points[fault] += int(local.shape[0])
                group_contributed = True
        if group_contributed:
            included_groups[fault] += 1

    label = np.zeros_like(centreline)
    validity = np.zeros_like(centreline)
    structure = ndimage.generate_binary_structure(2, 2)
    active_inlines = np.flatnonzero(centreline.any(axis=(1, 2)))
    for inline in active_inlines:
        label[inline] = ndimage.binary_dilation(
            centreline[inline], structure=structure, iterations=LABEL_RADIUS
        )
        validity[inline] = ndimage.binary_dilation(
            centreline[inline], structure=structure, iterations=VALIDITY_RADIUS
        )
    return centreline, label, validity, included_groups, included_points


def trace_index_rows(ilines, xlines, inline_values, crossline_values, roi_start):
    ni, nx, _ = ROI_SHAPE
    regular_order = np.array_equal(ilines, np.repeat(inline_values, len(crossline_values))) and np.array_equal(
        xlines, np.tile(crossline_values, len(inline_values))
    )
    rows = []
    if regular_order:
        for inline_index in range(roi_start[0], roi_start[0] + ni):
            first = inline_index * len(crossline_values) + roi_start[1]
            rows.append(np.arange(first, first + nx, dtype=np.int64))
    else:
        lookup = {(int(il), int(xl)): index for index, (il, xl) in enumerate(zip(ilines, xlines))}
        for inline_index in range(roi_start[0], roi_start[0] + ni):
            il = int(inline_values[inline_index])
            rows.append(
                np.array(
                    [
                        lookup[(il, int(crossline_values[crossline_index]))]
                        for crossline_index in range(roi_start[1], roi_start[1] + nx)
                    ],
                    dtype=np.int64,
                )
            )
    return rows, regular_order


def read_trace_block(seismic, indices, sample_start, sample_count):
    if np.all(np.diff(indices) == 1):
        block = np.asarray(seismic.trace.raw[int(indices[0]) : int(indices[-1]) + 1], dtype=np.float32)
    else:
        block = np.stack([np.asarray(seismic.trace.raw[int(index)], dtype=np.float32) for index in indices])
    return block[:, sample_start : sample_start + sample_count]


def extract_amplitude(seismic, trace_rows, roi_start, output_path):
    sample_start = roi_start[2]
    sample_count = ROI_SHAPE[2]
    percentile_samples = []
    for local_inline in range(0, ROI_SHAPE[0], 16):
        block = read_trace_block(seismic, trace_rows[local_inline], sample_start, sample_count)
        percentile_samples.append(block[::8, ::4].ravel())
    percentile_samples = np.concatenate(percentile_samples)
    finite = percentile_samples[np.isfinite(percentile_samples)]
    low, high = np.percentile(finite, [1, 99])

    amplitude = open_memmap(output_path, mode="w+", dtype=np.float16, shape=ROI_SHAPE)
    for local_inline, indices in enumerate(trace_rows):
        block = read_trace_block(seismic, indices, sample_start, sample_count)
        normalized = np.clip((block - low) / max(float(high - low), 1e-8), 0, 1)
        amplitude[local_inline] = normalized.astype(np.float16)
        if (local_inline + 1) % 32 == 0 or local_inline + 1 == ROI_SHAPE[0]:
            amplitude.flush()
            print(f"amplitude inline {local_inline + 1}/{ROI_SHAPE[0]}", flush=True)
    return float(low), float(high), int(finite.size)


def save_uint8(path, array):
    output = open_memmap(path, mode="w+", dtype=np.uint8, shape=array.shape)
    output[:] = array
    output.flush()


def plot_qc(amplitude, label, validity, output):
    active = np.flatnonzero(label.any(axis=(1, 2)))
    inline = int(active[len(active) // 2])
    time = int(np.argmax(label.sum(axis=(0, 1))))
    views = [
        ("Expert inline section", amplitude[inline].T, label[inline].T, validity[inline].T),
        ("Expert time slice", amplitude[:, :, time], label[:, :, time], validity[:, :, time]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    for row, (title, amp, truth, valid) in enumerate(views):
        axes[row, 0].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_title(title.replace("Expert ", "Amplitude "))
        axes[row, 1].imshow(amp, cmap="gray", aspect="auto", vmin=0, vmax=1)
        axes[row, 1].imshow(np.ma.masked_where(valid == 0, valid), cmap="Blues", alpha=0.20, aspect="auto")
        axes[row, 1].imshow(np.ma.masked_where(truth == 0, truth), cmap="autumn", aspect="auto")
        axes[row, 1].set_title(title)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, malformed = parse_sticks(args.sticks)
    if malformed:
        raise ValueError(f"Malformed expert-stick lines: {malformed[:5]}")

    with segyio.open(str(args.segy), "r", ignore_geometry=True) as seismic:
        ilines = np.asarray(seismic.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int64)
        xlines = np.asarray(seismic.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int64)
        inline_values = np.unique(ilines)
        crossline_values = np.unique(xlines)
        samples = np.asarray(seismic.samples, dtype=np.float64)
        volume_shape = (len(inline_values), len(crossline_values), len(samples))
        if seismic.tracecount != volume_shape[0] * volume_shape[1]:
            raise ValueError("GN1101 is not a complete regular inline/crossline grid.")

        resolved, xy_resolution = resolve_grid_coordinates(records, inline_values, crossline_values)
        mapped = map_records(resolved, inline_values, crossline_values, samples)
        selected = select_roi(mapped, volume_shape)
        roi_start = selected["start"]
        roi_end = tuple(start + size for start, size in zip(roi_start, ROI_SHAPE))
        trace_rows, regular_order = trace_index_rows(
            ilines, xlines, inline_values, crossline_values, roi_start
        )
        low, high, percentile_count = extract_amplitude(
            seismic, trace_rows, roi_start, args.output_dir / "amplitude_norm.npy"
        )

    centreline, label, validity, fault_sticks, fault_voxels = build_sparse_reference(
        mapped, roi_start, ROI_SHAPE
    )
    save_uint8(args.output_dir / "expert_centreline.npy", centreline)
    save_uint8(args.output_dir / "fault_label.npy", label)
    save_uint8(args.output_dir / "validity_mask.npy", validity)
    plot_qc(
        np.load(args.output_dir / "amplitude_norm.npy", mmap_mode="r"),
        label,
        validity,
        args.output_dir / "reference_qc.png",
    )

    roi_records = [record for record, included in zip(mapped, selected["mask"]) if included]
    fault_counts = Counter(record["fault"] for record in roi_records)
    sample_step = float(np.median(np.diff(samples)))
    metadata = {
        "id": "smeaheia_gn1101_expert_roi_384x512x640",
        "dataset": "Smeaheia GN1101",
        "data_role": "independent sparse 3D expert validation; no training or threshold selection",
        "doi": "10.11582/2021.00012",
        "source": {
            "segy": str(args.segy.relative_to(ROOT)),
            "segy_sha256": sha256(args.segy),
            "fault_sticks": str(args.sticks.relative_to(ROOT)),
            "fault_sticks_sha256": sha256(args.sticks),
        },
        "source_volume_shape": list(volume_shape),
        "shape": list(ROI_SHAPE),
        "starts": list(roi_start),
        "ends_exclusive": list(roi_end),
        "coordinate_ranges": {
            "inline": [int(inline_values[roi_start[0]]), int(inline_values[roi_end[0] - 1])],
            "crossline": [
                int(crossline_values[roi_start[1]]),
                int(crossline_values[roi_end[1] - 1]),
            ],
            "time_ms": [float(samples[roi_start[2]]), float(samples[roi_end[2] - 1])],
            "sample_interval_ms": sample_step,
        },
        "selection_policy": {
            "locked_before_prediction": True,
            "window_shape": list(ROI_SHAPE),
            "candidate_stride": SELECTION_STRIDE,
            "primary_score": "maximum released expert-point coverage",
            "secondary_scores": ["fault-object count", "fault-stick count"],
            "tie_break": "lexicographically earliest inline/crossline/sample start",
            "evaluated_window_count": selected["evaluated_window_count"],
            "winning_score_point_fault_stick": list(selected["score"]),
        },
        "expert_reference": {
            "released_point_count_in_roi": len(roi_records),
            "fault_object_count_in_roi": len(fault_counts),
            "fault_stick_count_in_roi": int(sum(fault_sticks.values())),
            "centreline_voxels": int(centreline.sum()),
            "label_voxels_radius_1": int(label.sum()),
            "validity_voxels_radius_8": int(validity.sum()),
            "active_inline_sections": int(label.any(axis=(1, 2)).sum()),
            "label_radius_2d_pixels": LABEL_RADIUS,
            "validity_radius_2d_pixels": VALIDITY_RADIUS,
            "faults": [
                {
                    "name": fault,
                    "released_points_in_roi": count,
                    "contributing_sticks": int(fault_sticks[fault]),
                    "rasterized_centreline_samples": int(fault_voxels[fault]),
                }
                for fault, count in fault_counts.most_common()
            ],
        },
        "xy_only_resolution": xy_resolution,
        "trace_order_regular": regular_order,
        "normalization": {
            "method": "1st/99th percentile clipping to [0,1]",
            "percentile_sample": "every 16th inline, 8th crossline and 4th time sample",
            "sampled_finite_voxels": percentile_count,
            "clip_low": low,
            "clip_high": high,
        },
        "evaluation_policy": (
            "Evaluate only on released expert inline sections and their 2D validity corridors. "
            "All other voxels are ignored because released sticks are not an exhaustive fault inventory."
        ),
        "threshold_policy": "Use Thebe val1-val2 thresholds unchanged for all models.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata["selection_policy"], indent=2))
    print(json.dumps(metadata["expert_reference"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
