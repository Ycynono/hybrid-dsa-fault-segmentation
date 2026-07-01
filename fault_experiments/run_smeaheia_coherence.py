import json
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from fault_experiments.evaluate_coherence_baseline import iter_block_scores


ROOT = Path(__file__).resolve().parents[1]
VOLUME = ROOT / "processed_data" / "smeaheia" / "expert_roi_384x512x640"
CALIBRATION = ROOT / "runs" / "dip_steered_coherence_baseline" / "calibration.json"
OUTPUT = ROOT / "runs" / "smeaheia_frozen_external" / "dip_steered_coherence"


def main():
    volume_metadata = json.loads((VOLUME / "metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    if calibration["selection_data"] != "Thebe val1-val2 only":
        raise RuntimeError("The coherence threshold was not selected only on Thebe validation data.")
    threshold = float(calibration["selected_threshold"])
    sample_window = int(calibration["sample_window"])
    smoothing_sigma = float(calibration["smoothing_sigma_inline_sample"])
    max_dip_shift = int(calibration["maximum_sample_shift_for_local_dip_steering"])
    amplitude = np.load(VOLUME / "amplitude_norm.npy", mmap_mode="r")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    probability = open_memmap(
        OUTPUT / "fault_probability.npy", mode="w+", dtype=np.float16, shape=amplitude.shape
    )
    binary = open_memmap(
        OUTPUT / "fault_binary.npy", mode="w+", dtype=np.uint8, shape=amplitude.shape
    )
    started = time.perf_counter()
    for inline, score in iter_block_scores(amplitude, sample_window, smoothing_sigma, max_dip_shift):
        probability[inline] = score.astype(np.float16)
        binary[inline] = score >= threshold
        if (inline + 1) % 32 == 0 or inline + 1 == amplitude.shape[0]:
            probability.flush()
            binary.flush()
            print(f"coherence inline {inline + 1}/{amplitude.shape[0]}", flush=True)
    runtime = time.perf_counter() - started
    metadata = {
        "data_role": "traditional attribute baseline on independent sparse expert validation",
        "volume": volume_metadata,
        "method": calibration["method"],
        "score_definition": calibration["score_definition"],
        "sample_window": sample_window,
        "smoothing_sigma_inline_sample": smoothing_sigma,
        "maximum_sample_shift_for_local_dip_steering": max_dip_shift,
        "threshold_source": "Thebe val1-val2 only; transferred unchanged to Smeaheia",
        "runtime_seconds": runtime,
        "statistics": {
            "threshold": threshold,
            "predicted_voxel_fraction": float(np.mean(binary)),
            "probability_min": float(np.min(probability)),
            "probability_max": float(np.max(probability)),
            "probability_mean": float(np.mean(probability, dtype=np.float64)),
        },
        "interpretation_warning": (
            "This is a reproducible local discontinuity baseline, not expert ground truth."
        ),
    }
    (OUTPUT / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["statistics"], indent=2))
    print(f"Wrote {OUTPUT} in {runtime:.1f} s")


if __name__ == "__main__":
    main()
