import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "processed_data" / "f3_faulta_benchmark"
XLINE_START = 512
XLINE_END = 768


def main():
    parser = argparse.ArgumentParser(description="Plot FaultA predictions against independent picks.")
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    prediction_root = Path(args.prediction_root)
    if args.model is not None and args.threshold is not None:
        thresholds = {args.model: args.threshold}
    else:
        calibrated = json.loads(
            (prediction_root / "faulta_calibrated_results.json").read_text(encoding="utf-8")
        )
        thresholds = {row["model"]: row["calibrated_threshold"] for row in calibrated["summary"]}
    output_root = prediction_root / "faulta_test_qc"
    output_root.mkdir(parents=True, exist_ok=True)

    for model, threshold in thresholds.items():
        model_dir = prediction_root / model if (prediction_root / model).is_dir() else prediction_root
        probability = np.load(model_dir / "fault_probability.npy", mmap_mode="r")
        for case_dir in sorted((BENCHMARK_ROOT / "sticks" / "test").glob("faulta_stick_*")):
            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            amplitude = np.load(case_dir / "amplitude.npy")[XLINE_START:XLINE_END].astype(np.float32)
            label = np.load(case_dir / "fault_label.npy")[XLINE_START:XLINE_END].astype(bool)
            valid = np.load(case_dir / "validity_mask.npy")[XLINE_START:XLINE_END].astype(bool)
            prob = np.asarray(probability[metadata["inline_index"]], dtype=np.float32)
            binary = prob >= threshold
            panels = [
                ("Amplitude", amplitude.T, "gray", 0, 1),
                ("Independent label", amplitude.T, "gray", 0, 1),
                ("Fault probability", prob.T, "inferno", 0, 1),
                (f"Prediction p >= {threshold:.2f}", amplitude.T, "gray", 0, 1),
            ]
            fig, axes = plt.subplots(1, 4, figsize=(14, 5), dpi=180)
            for ax, (title, image, cmap, vmin, vmax) in zip(axes, panels):
                ax.imshow(image, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                ax.set_title(title)
                ax.set_xlabel("Local crossline")
                ax.set_ylabel("Time sample")
            axes[1].imshow(np.ma.masked_where(~label.T, label.T), cmap="spring", aspect="auto", vmin=0, vmax=1)
            shown_prediction = np.logical_and(binary, valid)
            axes[3].imshow(
                np.ma.masked_where(~shown_prediction.T, shown_prediction.T), cmap="autumn", alpha=0.7, aspect="auto"
            )
            axes[3].imshow(np.ma.masked_where(~label.T, label.T), cmap="spring", aspect="auto", vmin=0, vmax=1)
            fig.suptitle(f"{model} | {metadata['case_id']} | cyan/pink=reference, orange=prediction")
            fig.tight_layout()
            fig.savefig(output_root / f"{model}_{metadata['case_id']}.png", bbox_inches="tight")
            plt.close(fig)
    print("Wrote", output_root)


if __name__ == "__main__":
    main()
