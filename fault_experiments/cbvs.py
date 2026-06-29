from pathlib import Path

import numpy as np


UNDEFINED_LIMIT = 1.0e20


def open_regular_cbvs(path, shape, components=1):
    """Open a regular little-endian CBVS volume after validating its record layout."""
    path = Path(path)
    ni, nx, ns = (int(value) for value in shape)
    with path.open("rb") as handle:
        signature = handle.read(8)
        header_size = int.from_bytes(handle.read(4), "little")
    if signature[:3] != b"dGB":
        raise ValueError(f"Not an OpendTect CBVS file: {path}")
    record_size = 20 + components * ns * 4
    expected_size = header_size + ni * nx * record_size + 8
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"Unexpected CBVS size for {path}: {path.stat().st_size}, expected {expected_size}"
        )
    dtype = np.dtype(
        [
            ("trace_header", "V20"),
            ("samples", "<f4", (components, ns)),
        ]
    )
    records = np.memmap(path, mode="r", dtype=dtype, offset=header_size, shape=(ni * nx,))
    return records["samples"].reshape(ni, nx, components, ns), header_size


def clean_component(array, undefined_fill=np.nan):
    result = np.asarray(array, dtype=np.float32).copy()
    invalid = ~np.isfinite(result) | (np.abs(result) >= UNDEFINED_LIMIT)
    result[invalid] = undefined_fill
    return result
