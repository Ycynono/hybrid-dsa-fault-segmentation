import argparse
import gc
import json
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


def main():
    parser = argparse.ArgumentParser(description="Convert a compressed Thebe block to mmap-friendly arrays.")
    parser.add_argument("--seismic", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading compressed seismic block", flush=True)
    with np.load(args.seismic) as archive:
        amplitude = archive[archive.files[0]]
    sample = amplitude[:, ::16, ::8]
    low, high = np.percentile(sample[np.isfinite(sample)], [1, 99])
    output_amplitude = open_memmap(
        output_dir / "amplitude_norm.npy", mode="w+", dtype=np.float16, shape=amplitude.shape
    )
    for start in range(0, amplitude.shape[0], 5):
        end = min(start + 5, amplitude.shape[0])
        chunk = np.asarray(amplitude[start:end], dtype=np.float32)
        chunk = np.clip((chunk - low) / max(high - low, 1e-8), 0, 1)
        output_amplitude[start:end] = (2.0 * chunk - 1.0).astype(np.float16)
        output_amplitude.flush()
        print(f"amplitude {end}/{amplitude.shape[0]}", flush=True)
    del output_amplitude, amplitude, sample
    gc.collect()

    print("Loading expert fault labels", flush=True)
    with np.load(args.fault) as archive:
        label = archive[archive.files[0]].astype(bool, copy=False)
    output_label = open_memmap(
        output_dir / "fault_label.npy", mode="w+", dtype=bool, shape=label.shape
    )
    for start in range(0, label.shape[0], 10):
        end = min(start + 10, label.shape[0])
        output_label[start:end] = label[start:end]
        output_label.flush()
    metadata = {
        "dataset": "Thebe Gas Field expert-interpreted fault dataset",
        "source_seismic": str(Path(args.seismic)),
        "source_fault": str(Path(args.fault)),
        "shape": list(label.shape),
        "axis_order_from_dataset": ["crossline_chunk", "inline", "sample"],
        "normalization": {
            "method": "sampled 1st/99th percentile clipping mapped to [-1,1]",
            "sample_stride": [1, 16, 8],
            "clip_low": float(low),
            "clip_high": float(high),
        },
        "fault_fraction": float(label.mean()),
        "label_role": "independent expert interpretation; external test only",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
