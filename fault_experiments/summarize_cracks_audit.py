import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/cracks_audit_frozen"
MODELS = {
    "U-Net": "unet",
    "Hybrid DSA": "hybrid_dsa",
    "SwinUNETR F3-chain": "swinunetr_f3chain",
}


def load_rows(path):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {"section": int(row["section"])}
            for key, value in row.items():
                if key != "section":
                    converted[key] = float(value)
            denominator = 2 * converted["tp"] + converted["fp"] + converted["fn"]
            converted["dice"] = 2 * converted["tp"] / max(denominator, 1)
            rows.append(converted)
    return rows


def paired_statistics(baseline, proposed, metric, rng, iterations=100000):
    base = np.array([row[metric] for row in baseline])
    prop = np.array([row[metric] for row in proposed])
    differences = prop - base
    indices = rng.integers(0, len(base), size=(iterations, len(base)))
    samples = differences[indices].mean(axis=1)
    nonzero = differences[differences != 0]
    return {
        "metric": metric,
        "n_paired_sections": len(base),
        "baseline_macro_mean": float(base.mean()),
        "proposed_macro_mean": float(prop.mean()),
        "paired_mean_difference": float(differences.mean()),
        "bootstrap_95_ci_difference": [
            float(value) for value in np.percentile(samples, [2.5, 97.5])
        ],
        "sections_proposed_better": int((differences > 0).sum()),
        "wilcoxon_two_sided_p": float(
            wilcoxon(prop, base, alternative="two-sided", method="auto").pvalue
        ),
        "sign_test_two_sided_p": float(
            binomtest(int((nonzero > 0).sum()), len(nonzero), alternative="two-sided").pvalue
        ),
    }


def main():
    rng = np.random.default_rng(20260627)
    rows = {
        name: load_rows(RUN_ROOT / directory / "per_section_metrics.csv")
        for name, directory in MODELS.items()
    }
    summaries = {
        name: json.loads((RUN_ROOT / directory / "summary.json").read_text(encoding="utf-8"))
        for name, directory in MODELS.items()
    }
    section_table = []
    for index, section in enumerate(rows["U-Net"]):
        record = {"section": section["section"]}
        for name in MODELS:
            prefix = name.lower().replace(" ", "_").replace("-", "")
            record[f"{prefix}_dice"] = rows[name][index]["dice"]
            record[f"{prefix}_tolerant_dice_3px"] = rows[name][index]["tolerant_dice"]
        section_table.append(record)
    with (RUN_ROOT / "section_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(section_table[0]))
        writer.writeheader()
        writer.writerows(section_table)

    comparisons = {}
    for baseline_name, proposed_name in [
        ("U-Net", "Hybrid DSA"),
        ("SwinUNETR F3-chain", "Hybrid DSA"),
    ]:
        key = f"{proposed_name}_vs_{baseline_name}"
        comparisons[key] = {
            metric: paired_statistics(
                rows[baseline_name], rows[proposed_name], metric, rng
            )
            for metric in ("dice", "tolerant_dice")
        }
    result = {
        "scope": "20 preregistered CRACKS audit sections; sealed reserve not accessed",
        "independence_caveat": "sections are paired observations from one F3 survey and are not independent surveys",
        "threshold_policy": "all thresholds selected on Thebe val1-val2; no CRACKS calibration",
        "aggregate": {
            name: summary["combined_certain_and_uncertain_faults"]
            for name, summary in summaries.items()
        },
        "paired_section_statistics": comparisons,
    }
    (RUN_ROOT / "audit_statistics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
