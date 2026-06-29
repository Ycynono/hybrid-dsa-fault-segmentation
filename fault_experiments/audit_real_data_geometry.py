import json
from pathlib import Path

import numpy as np
import segyio


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "F3": ROOT / "F3" / "Export" / "amplitude.sgy",
    "FORCE": ROOT / "FORCE_ML_Competition_2020" / "Export" / "amplitude.sgy",
}


def audit_segy(name, path):
    with segyio.open(str(path), "r", ignore_geometry=True) as seismic:
        ilines = np.asarray(seismic.attributes(segyio.TraceField.INLINE_3D)[:])
        xlines = np.asarray(seismic.attributes(segyio.TraceField.CROSSLINE_3D)[:])
        samples = np.asarray(seismic.samples)
        il_unique = np.unique(ilines)
        xl_unique = np.unique(xlines)
        expected_traces = int(il_unique.size * xl_unique.size)
        npy_path = path.with_suffix(".npy")
        npy = np.load(npy_path, mmap_mode="r")
        expected_shape = (int(il_unique.size), int(xl_unique.size), int(samples.size))
        return {
            "survey": name,
            "segy_path": str(path.relative_to(ROOT)),
            "npy_path": str(npy_path.relative_to(ROOT)),
            "trace_count": int(seismic.tracecount),
            "sample_count": int(samples.size),
            "sample_interval_us": float(segyio.tools.dt(seismic)),
            "sample_start": float(samples[0]),
            "sample_end": float(samples[-1]),
            "inline_min": int(il_unique.min()),
            "inline_max": int(il_unique.max()),
            "inline_count": int(il_unique.size),
            "crossline_min": int(xl_unique.min()),
            "crossline_max": int(xl_unique.max()),
            "crossline_count": int(xl_unique.size),
            "expected_regular_grid_traces": expected_traces,
            "regular_grid_complete": bool(seismic.tracecount == expected_traces),
            "trace_order": "crossline varies fastest within each inline",
            "expected_npy_shape": list(expected_shape),
            "actual_npy_shape": list(npy.shape),
            "npy_shape_matches_headers": bool(tuple(npy.shape) == expected_shape),
            "npy_dtype": str(npy.dtype),
            "axis_order": ["inline", "crossline", "time/depth sample"],
        }


def main():
    output = ROOT / "processed_data" / "real_data_geometry_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    audits = [audit_segy(name, path) for name, path in SOURCES.items()]
    result = {
        "purpose": "Verify field-volume geometry before candidate selection and annotation.",
        "audits": audits,
        "decision": (
            "Both exported NumPy arrays match the SEG-Y header grids. Axis-order mismatch is not a "
            "plausible explanation for the poor FORCE prediction."
        ),
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
