import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs/cracks_audit_frozen"
RESERVE = ROOT / "runs/cracks_reserve_final"
OUTPUT = ROOT / "runs/cracks_final_statistics.json"
MODELS = {
    "U-Net": "unet",
    "Hybrid DSA": "hybrid_dsa",
    "SwinUNETR F3-chain": "swinunetr_f3chain",
}


def read_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {"section": int(row["section"])}
            converted.update({key: float(value) for key, value in row.items() if key != "section"})
            converted["dice"] = 2 * converted["tp"] / max(
                2 * converted["tp"] + converted["fp"] + converted["fn"], 1
            )
            rows.append(converted)
    return rows


def aggregate(rows):
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    return {
        "n_sections": len(rows),
        "micro_precision": tp / max(tp + fp, 1),
        "micro_recall": tp / max(tp + fn, 1),
        "micro_dice": 2 * tp / max(2 * tp + fp + fn, 1),
        "macro_exact_dice": float(np.mean([row["dice"] for row in rows])),
        "macro_tolerant_dice_3px": float(np.mean([row["tolerant_dice"] for row in rows])),
    }


def compare(baseline, proposed, metric, rng, iterations=100000):
    base = np.array([row[metric] for row in baseline])
    prop = np.array([row[metric] for row in proposed])
    difference = prop - base
    indices = rng.integers(0, len(base), size=(iterations, len(base)))
    bootstrap = difference[indices].mean(axis=1)
    nonzero = difference[difference != 0]
    return {
        "paired_mean_difference": float(difference.mean()),
        "bootstrap_95_ci_difference": [
            float(value) for value in np.percentile(bootstrap, [2.5, 97.5])
        ],
        "sections_proposed_better": int((difference > 0).sum()),
        "wilcoxon_two_sided_p": float(wilcoxon(prop, base, method="auto").pvalue),
        "sign_test_two_sided_p": float(
            binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue
        ),
    }


def split_statistics(rows, rng):
    return {
        "aggregate": {model: aggregate(values) for model, values in rows.items()},
        "hybrid_vs_unet": {
            metric: compare(rows["U-Net"], rows["Hybrid DSA"], metric, rng)
            for metric in ("dice", "tolerant_dice")
        },
        "hybrid_vs_swin": {
            metric: compare(
                rows["SwinUNETR F3-chain"], rows["Hybrid DSA"], metric, rng
            )
            for metric in ("dice", "tolerant_dice")
        },
    }


def main():
    audit_rows = {
        model: read_rows(AUDIT / directory / "per_section_metrics.csv")
        for model, directory in MODELS.items()
    }
    reserve_rows = {
        model: read_rows(RESERVE / directory / "per_section_metrics.csv")
        for model, directory in MODELS.items()
    }
    combined_rows = {
        model: audit_rows[model] + reserve_rows[model] for model in MODELS
    }
    rng = np.random.default_rng(20260627)
    result = {
        "protocol": "audit 20 sections used for confirmation; reserve 20 sections opened once after FINAL_PROTOCOL_LOCK.json",
        "threshold_source": "Thebe val1-val2 only; no CRACKS calibration",
        "inference_caveat": "all sections belong to one F3 survey",
        "audit": split_statistics(audit_rows, rng),
        "reserve_primary": split_statistics(reserve_rows, rng),
        "combined_descriptive": split_statistics(combined_rows, rng),
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["reserve_primary"], indent=2))


if __name__ == "__main__":
    main()
