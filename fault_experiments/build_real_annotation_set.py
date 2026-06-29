import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "processed_data" / "real_fault_annotation_v1"
BLOCK_SHAPE = (128, 128, 128)

SURVEYS = {
    "F3": {
        "path": ROOT / "F3" / "Export" / "amplitude.npy",
        "inline0": 100,
        "crossline0": 300,
        "sample0": 0.0,
        "sample_step": 4.0,
        "sample_unit": "ms",
    },
    "FORCE": {
        "path": ROOT / "FORCE_ML_Competition_2020" / "Export" / "amplitude.npy",
        "inline0": 1001,
        "crossline0": 2040,
        "sample0": 1500.0,
        "sample_step": 4.0,
        "sample_unit": "ms",
    },
}

TARGETS = {"train": 6, "val": 4, "test": 5}


def robust_normalize(section):
    finite = section[np.isfinite(section)]
    if finite.size == 0:
        return np.zeros_like(section, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((section - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(normalized, nan=0.5, posinf=1.0, neginf=0.0).astype(np.float32)


def section_score(section):
    image = robust_normalize(section)
    smooth = ndimage.gaussian_filter(image, sigma=(0.8, 0.8))
    lateral_gradient = np.abs(np.gradient(smooth, axis=0))
    temporal_gradient = np.abs(np.gradient(smooth, axis=1))
    lateral_edge = float(np.percentile(lateral_gradient, 95))
    reflector_energy = float(np.percentile(temporal_gradient, 75))

    traces = smooth - smooth.mean(axis=1, keepdims=True)
    numerator = np.sum(traces[:-1] * traces[1:], axis=1)
    denominator = np.sqrt(np.sum(traces[:-1] ** 2, axis=1) * np.sum(traces[1:] ** 2, axis=1)) + 1e-8
    adjacent_correlation = np.clip(numerator / denominator, -1, 1)
    continuity = float(np.median(np.maximum(adjacent_correlation, 0)))

    residual = image - smooth
    noise_mad = float(np.median(np.abs(residual - np.median(residual))))
    signal_to_noise_proxy = reflector_energy / max(noise_mad, 1e-4)
    discontinuity = lateral_edge * max(continuity, 0.05)
    score = discontinuity * reflector_energy * np.clip(signal_to_noise_proxy, 0.25, 4.0)
    return {
        "score": float(score),
        "lateral_edge_p95": lateral_edge,
        "reflector_energy_p75": reflector_energy,
        "adjacent_trace_correlation_median": continuity,
        "noise_mad": noise_mad,
        "signal_to_noise_proxy": float(signal_to_noise_proxy),
    }


def split_ranges(axis_length):
    # Spatially disjoint crossline regions with approximately 10% survey-width buffers.
    return {
        "train": (0, int(0.40 * axis_length)),
        "val": (int(0.50 * axis_length), int(0.70 * axis_length)),
        "test": (int(0.80 * axis_length), axis_length),
    }


def starts_within(low, high, patch, stride=64):
    last = high - patch
    if last < low:
        return []
    starts = list(range(low, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def scan_candidates(volume, split_name, crossline_range):
    ni, _, nt = volume.shape
    bi, bx, bt = BLOCK_SHAPE
    inline_starts = starts_within(0, ni, bi, stride=128)
    crossline_starts = starts_within(*crossline_range, bx, stride=64)
    time_starts = starts_within(0, nt, bt, stride=128)
    candidates = []
    for i in inline_starts:
        for x in crossline_starts:
            for t in time_starts:
                inline_section = np.asarray(volume[i + bi // 2, x : x + bx, t : t + bt], dtype=np.float32)
                crossline_section = np.asarray(volume[i : i + bi, x + bx // 2, t : t + bt], dtype=np.float32)
                finite_fraction = float(
                    0.5 * (np.isfinite(inline_section).mean() + np.isfinite(crossline_section).mean())
                )
                valid_values = np.concatenate(
                    [inline_section[np.isfinite(inline_section)], crossline_section[np.isfinite(crossline_section)]]
                )
                zero_fraction = float((valid_values == 0).mean()) if valid_values.size else 1.0
                if finite_fraction < 0.995 or zero_fraction > 0.05:
                    continue
                inline_metrics = section_score(inline_section)
                crossline_metrics = section_score(crossline_section)
                score = float(np.sqrt(inline_metrics["score"] * crossline_metrics["score"]))
                candidates.append(
                    {
                        "split": split_name,
                        "inline_start": i,
                        "crossline_start": x,
                        "sample_start": t,
                        "score": score,
                        "finite_fraction": finite_fraction,
                        "zero_fraction": zero_fraction,
                        "inline_metrics": inline_metrics,
                        "crossline_metrics": crossline_metrics,
                    }
                )
    return candidates


def separated(candidate, selected):
    for other in selected:
        di = abs(candidate["inline_start"] - other["inline_start"])
        dx = abs(candidate["crossline_start"] - other["crossline_start"])
        dt = abs(candidate["sample_start"] - other["sample_start"])
        if di < BLOCK_SHAPE[0] and dx < BLOCK_SHAPE[1] and dt < BLOCK_SHAPE[2]:
            return False
    return True


def stratified_select(candidates, count):
    ordered = sorted(candidates, key=lambda row: row["score"])
    bins = {
        "low": ordered[: max(1, len(ordered) // 3)],
        "medium": ordered[len(ordered) // 3 : 2 * len(ordered) // 3],
        "high": ordered[2 * len(ordered) // 3 :],
    }
    allocation = ["high", "high", "medium", "low", "high", "medium", "low"]
    selected = []
    for stratum in allocation:
        if len(selected) >= count:
            break
        pool = sorted(bins[stratum], key=lambda row: row["score"], reverse=stratum == "high")
        choice = next((row for row in pool if row not in selected and separated(row, selected)), None)
        if choice is None:
            choice = next((row for row in pool if row not in selected), None)
        if choice is not None:
            item = dict(choice)
            item["selection_stratum"] = stratum
            selected.append(item)
    if len(selected) < count:
        for row in sorted(candidates, key=lambda item: item["score"], reverse=True):
            if row not in selected:
                item = dict(row)
                item["selection_stratum"] = "fallback"
                selected.append(item)
            if len(selected) >= count:
                break
    return selected


def save_grayscale(image, output):
    plt.imsave(output, robust_normalize(image).T, cmap="gray", vmin=0, vmax=1)


def save_context(inline_section, crossline_section, time_slice, metadata, output):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), dpi=180)
    panels = [
        ("Inline amplitude", inline_section.T),
        ("Crossline amplitude", crossline_section.T),
        ("Time-slice amplitude", time_slice),
    ]
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(robust_normalize(image), cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Blind case {metadata['case_id']} | split={metadata['split']} | stratum={metadata['selection_stratum']}")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def materialize_case(survey_name, config, volume, candidate, index):
    bi, bx, bt = BLOCK_SHAPE
    i, x, t = candidate["inline_start"], candidate["crossline_start"], candidate["sample_start"]
    case_id = f"{survey_name.lower()}_{candidate['split']}_{index:02d}"
    case_dir = OUTPUT_ROOT / survey_name / candidate["split"] / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    inline_section = np.asarray(volume[i + bi // 2, x : x + bx, t : t + bt], dtype=np.float32)
    crossline_section = np.asarray(volume[i : i + bi, x + bx // 2, t : t + bt], dtype=np.float32)
    time_slice = np.asarray(volume[i : i + bi, x : x + bx, t + bt // 2], dtype=np.float32)
    np.save(case_dir / "amplitude_inline.npy", robust_normalize(inline_section).astype(np.float16))
    np.save(case_dir / "amplitude_crossline.npy", robust_normalize(crossline_section).astype(np.float16))
    np.save(case_dir / "label_inline.npy", np.full((bx, bt), 255, dtype=np.uint8))
    np.save(case_dir / "label_crossline.npy", np.full((bi, bt), 255, dtype=np.uint8))
    save_grayscale(inline_section, case_dir / "amplitude_inline.png")
    save_grayscale(crossline_section, case_dir / "amplitude_crossline.png")

    metadata = {
        "case_id": case_id,
        "survey": survey_name,
        "split": candidate["split"],
        "selection_stratum": candidate["selection_stratum"],
        "selection_uses_model_predictions": False,
        "selection_score": candidate["score"],
        "finite_fraction": candidate["finite_fraction"],
        "zero_fraction": candidate["zero_fraction"],
        "inline_metrics": candidate["inline_metrics"],
        "crossline_metrics": candidate["crossline_metrics"],
        "block_start_indices": [i, x, t],
        "block_shape": list(BLOCK_SHAPE),
        "inline_coordinate_range": [config["inline0"] + i, config["inline0"] + i + bi - 1],
        "crossline_coordinate_range": [config["crossline0"] + x, config["crossline0"] + x + bx - 1],
        "sample_coordinate_range": [
            config["sample0"] + t * config["sample_step"],
            config["sample0"] + (t + bt - 1) * config["sample_step"],
        ],
        "sample_unit": config["sample_unit"],
        "label_encoding": {"0": "background", "1": "fault", "2": "uncertain", "255": "unannotated"},
        "annotation_status": "pending",
    }
    save_context(inline_section, crossline_section, time_slice, metadata, case_dir / "blind_context.png")
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_metadata = []
    screening_rows = []
    split_policies = {}
    for survey_name, config in SURVEYS.items():
        print(f"Screening {survey_name}", flush=True)
        volume = np.load(config["path"], mmap_mode="r")
        ranges = split_ranges(volume.shape[1])
        split_policies[survey_name] = {
            "axis": "crossline_index",
            "ranges": {key: list(value) for key, value in ranges.items()},
            "buffer_regions": [[ranges["train"][1], ranges["val"][0]], [ranges["val"][1], ranges["test"][0]]],
            "source_shape": list(volume.shape),
        }
        for split_name, crossline_range in ranges.items():
            candidates = scan_candidates(volume, split_name, crossline_range)
            for row in candidates:
                screening_rows.append({"survey": survey_name, **row})
            selected = stratified_select(candidates, TARGETS[split_name])
            for index, candidate in enumerate(selected, start=1):
                metadata = materialize_case(survey_name, config, volume, candidate, index)
                all_metadata.append(metadata)
                print(metadata["case_id"], round(metadata["selection_score"], 6), flush=True)

    manifest_fields = [
        "case_id", "survey", "split", "selection_stratum", "selection_score",
        "block_start_indices", "inline_coordinate_range", "crossline_coordinate_range",
        "sample_coordinate_range", "annotation_status",
    ]
    with (OUTPUT_ROOT / "annotation_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for row in all_metadata:
            writer.writerow({key: row[key] for key in manifest_fields})

    screening_flat = []
    for row in screening_rows:
        screening_flat.append(
            {
                "survey": row["survey"],
                "split": row["split"],
                "inline_start": row["inline_start"],
                "crossline_start": row["crossline_start"],
                "sample_start": row["sample_start"],
                "score": row["score"],
                "finite_fraction": row["finite_fraction"],
                "zero_fraction": row["zero_fraction"],
            }
        )
    with (OUTPUT_ROOT / "screening_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(screening_flat[0]))
        writer.writeheader()
        writer.writerows(screening_flat)

    manifest = {
        "purpose": "Blind field-seismic annotation set for zero-shot and few-shot evaluation.",
        "case_count": len(all_metadata),
        "section_count": 2 * len(all_metadata),
        "targets_per_survey": TARGETS,
        "block_shape": list(BLOCK_SHAPE),
        "selection": "Amplitude-only structural screening; no neural-network prediction was read.",
        "split_policies": split_policies,
        "label_encoding": {"0": "background", "1": "fault", "2": "uncertain", "255": "unannotated"},
        "cases": all_metadata,
    }
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_metadata)} blocks / {2 * len(all_metadata)} sections to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
