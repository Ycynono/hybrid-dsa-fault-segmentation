import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import segyio


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STICKS = (
    ROOT
    / "external_data"
    / "smeaheia"
    / "fault_sticks"
    / "Fault_Sticks"
    / "data"
    / "fault_Sticks_GN1101_2012"
)
DEFAULT_SEGY_ROOT = ROOT / "external_data" / "smeaheia" / "seismic_3d"
DEFAULT_OUTPUT = ROOT / "processed_data" / "smeaheia" / "data_quality_audit.json"
MISSING_LINE_NUMBER = np.iinfo(np.int32).max


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Smeaheia expert sticks and GN1101 geometry.")
    parser.add_argument("--sticks", type=Path, default=DEFAULT_STICKS)
    parser.add_argument(
        "--segy",
        type=Path,
        help="GN1101 SEG-Y. If omitted, the script searches external_data/smeaheia/seismic_3d.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-segy", action="store_true")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sticks(path):
    records = []
    malformed = []
    for line_number, line in enumerate(path.read_text(encoding="cp1252").splitlines(), 1):
        fields = line.split()
        if len(fields) != 8:
            malformed.append({"line_number": line_number, "field_count": len(fields)})
            continue
        records.append(
            {
                "orientation": fields[0],
                "inline": int(fields[1]),
                "crossline": int(fields[2]),
                "x": float(fields[3]),
                "y": float(fields[4]),
                "z": float(fields[5]),
                "fault": fields[6],
                "stick": int(fields[7]),
                "line_number": line_number,
            }
        )
    return records, malformed


def range_summary(values):
    values = np.asarray(values)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "unique": int(np.unique(values).size),
    }


def summarize_sticks(path, records, malformed):
    has_grid = np.array(
        [
            record["inline"] != MISSING_LINE_NUMBER
            and record["crossline"] != MISSING_LINE_NUMBER
            for record in records
        ],
        dtype=bool,
    )
    valid_grid_records = [record for record, valid in zip(records, has_grid) if valid]
    fault_counts = Counter(record["fault"] for record in records)
    fault_sticks = defaultdict(set)
    for record in records:
        fault_sticks[record["fault"]].add(record["stick"])
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256(path),
        "encoding": "cp1252",
        "point_count": len(records),
        "malformed_line_count": len(malformed),
        "points_with_inline_crossline": int(has_grid.sum()),
        "points_with_xy_only": int((~has_grid).sum()),
        "fault_object_count": len(fault_counts),
        "fault_stick_count": int(sum(len(sticks) for sticks in fault_sticks.values())),
        "ranges": {
            "inline": range_summary([record["inline"] for record in valid_grid_records]),
            "crossline": range_summary([record["crossline"] for record in valid_grid_records]),
            "x_m": range_summary([record["x"] for record in records]),
            "y_m": range_summary([record["y"] for record in records]),
            "z": range_summary([record["z"] for record in records]),
        },
        "faults": [
            {
                "name": name,
                "point_count": int(count),
                "stick_count": len(fault_sticks[name]),
            }
            for name, count in fault_counts.most_common()
        ],
        "interpretation_note": (
            "The provider file contains sparse expert fault sticks. Missing inline/crossline sentinels "
            "retain valid X/Y/Z coordinates and must be registered spatially rather than discarded."
        ),
    }


def coordinate_scale(scalars):
    scalars = np.asarray(scalars, dtype=np.float64)
    return np.where(scalars < 0, 1.0 / np.maximum(np.abs(scalars), 1.0), np.maximum(scalars, 1.0))


def read_coordinates(seismic, x_field, y_field):
    scalars = np.asarray(seismic.attributes(segyio.TraceField.SourceGroupScalar)[:], dtype=np.float64)
    scale = coordinate_scale(scalars)
    x = np.asarray(seismic.attributes(x_field)[:], dtype=np.float64) * scale
    y = np.asarray(seismic.attributes(y_field)[:], dtype=np.float64) * scale
    return x, y


def grid_spacing(ilines, xlines, x, y):
    order = np.lexsort((xlines, ilines))
    il_sorted = ilines[order]
    xl_sorted = xlines[order]
    xy_sorted = np.column_stack([x[order], y[order]])
    same_inline = il_sorted[1:] == il_sorted[:-1]
    changed_crossline = xl_sorted[1:] != xl_sorted[:-1]
    distances = np.linalg.norm(xy_sorted[1:] - xy_sorted[:-1], axis=1)
    candidates = distances[same_inline & changed_crossline & (distances > 0)]
    if candidates.size == 0:
        return None
    return float(np.median(candidates))


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def audit_segy(path, records):
    with segyio.open(str(path), "r", ignore_geometry=True) as seismic:
        ilines = np.asarray(seismic.attributes(segyio.TraceField.INLINE_3D)[:], dtype=np.int64)
        xlines = np.asarray(seismic.attributes(segyio.TraceField.CROSSLINE_3D)[:], dtype=np.int64)
        samples = np.asarray(seismic.samples, dtype=np.float64)
        sample_interval = float(segyio.tools.dt(seismic) / 1000.0)

        grid_to_trace = {(int(il), int(xl)): index for index, (il, xl) in enumerate(zip(ilines, xlines))}
        indexed_records = [
            record
            for record in records
            if record["inline"] != MISSING_LINE_NUMBER
            and record["crossline"] != MISSING_LINE_NUMBER
        ]
        matched_records = [
            record
            for record in indexed_records
            if (record["inline"], record["crossline"]) in grid_to_trace
        ]
        matched_indices = np.array(
            [grid_to_trace[(record["inline"], record["crossline"])] for record in matched_records],
            dtype=np.int64,
        )

        coordinate_candidates = {}
        for name, x_field, y_field in (
            ("CDP_X_CDP_Y", segyio.TraceField.CDP_X, segyio.TraceField.CDP_Y),
            ("SourceX_SourceY", segyio.TraceField.SourceX, segyio.TraceField.SourceY),
            ("GroupX_GroupY", segyio.TraceField.GroupX, segyio.TraceField.GroupY),
        ):
            x, y = read_coordinates(seismic, x_field, y_field)
            if matched_indices.size:
                expert_xy = np.array([[record["x"], record["y"]] for record in matched_records])
                trace_xy = np.column_stack([x[matched_indices], y[matched_indices]])
                residual = np.linalg.norm(expert_xy - trace_xy, axis=1)
                residual_summary = percentile_summary(residual)
            else:
                residual_summary = None
            coordinate_candidates[name] = {
                "nonzero_coordinate_fraction": float(np.mean((x != 0) | (y != 0))),
                "expert_trace_xy_residual_m": residual_summary,
                "median_crossline_spacing_m": grid_spacing(ilines, xlines, x, y),
            }

        usable_candidates = {
            name: value
            for name, value in coordinate_candidates.items()
            if value["expert_trace_xy_residual_m"] is not None
            and value["nonzero_coordinate_fraction"] > 0.5
        }
        best_name = min(
            usable_candidates,
            key=lambda name: usable_candidates[name]["expert_trace_xy_residual_m"]["p95"],
            default=None,
        )
        z = np.asarray([record["z"] for record in records], dtype=np.float64)
        nearest_samples = samples[0] + np.rint((z - samples[0]) / sample_interval) * sample_interval
        z_residual = np.abs(z - nearest_samples)
        vertical_overlap = float(np.mean((z >= samples.min()) & (z <= samples.max())))
        grid_match_fraction = len(matched_records) / max(len(indexed_records), 1)

        best = coordinate_candidates.get(best_name)
        horizontal_pass = False
        if best and best["median_crossline_spacing_m"]:
            horizontal_pass = (
                best["expert_trace_xy_residual_m"]["p95"]
                <= 0.5 * best["median_crossline_spacing_m"]
            )
        vertical_pass = vertical_overlap >= 0.99 and np.percentile(z_residual, 95) <= sample_interval / 2
        return {
            "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "sha256": sha256(path),
            "trace_count": int(seismic.tracecount),
            "sample_count": int(samples.size),
            "sample_start": float(samples.min()),
            "sample_end": float(samples.max()),
            "sample_interval": sample_interval,
            "inline_range": [int(ilines.min()), int(ilines.max())],
            "crossline_range": [int(xlines.min()), int(xlines.max())],
            "expert_grid_match_fraction": grid_match_fraction,
            "coordinate_candidates": coordinate_candidates,
            "selected_coordinate_headers": best_name,
            "vertical": {
                "expert_z_overlap_fraction": vertical_overlap,
                "expert_z_to_nearest_sample_residual": percentile_summary(z_residual),
            },
            "quality_gates": {
                "inline_crossline_match_at_least_95_percent": bool(grid_match_fraction >= 0.95),
                "xy_p95_below_half_trace_spacing": bool(horizontal_pass),
                "z_within_sample_axis_and_half_sample": bool(vertical_pass),
            },
        }


def discover_segy(root):
    candidates = sorted(root.rglob("*.sgy")) + sorted(root.rglob("*.segy"))
    candidates += sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "gn1101" in path.name.lower()
        and path.suffix.lower() not in {".xml", ".txt", ".json"}
    )
    candidates = list(dict.fromkeys(candidates))
    gn1101 = [path for path in candidates if "gn1101" in path.name.lower()]
    if len(gn1101) == 1:
        return gn1101[0]
    if not candidates:
        return None
    raise RuntimeError(f"Could not select a unique GN1101 SEG-Y from: {candidates}")


def main():
    args = parse_args()
    records, malformed = parse_sticks(args.sticks)
    result = {
        "dataset": "Smeaheia GN1101",
        "provider": "Gassnova and Equinor via CO2DataShare",
        "doi": "10.11582/2021.00012",
        "fault_sticks": summarize_sticks(args.sticks, records, malformed),
    }
    segy_path = args.segy or discover_segy(DEFAULT_SEGY_ROOT)
    if segy_path is not None:
        result["seismic"] = audit_segy(segy_path, records)
        gates = result["seismic"]["quality_gates"]
        result["decision"] = "pass" if all(gates.values()) else "fail_registration_gates"
    else:
        result["decision"] = "pending_segy_download"
        if args.require_segy:
            raise FileNotFoundError("GN1101 SEG-Y was not found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
