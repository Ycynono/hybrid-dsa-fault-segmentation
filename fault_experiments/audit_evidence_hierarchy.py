from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/evidence_hierarchy"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    thebe = read_csv(ROOT / "runs/thebe_final_test2_7/statistics/block_metrics.csv")
    cracks = {
        "U-Net": read_csv(ROOT / "runs/cracks_reserve_final/unet/per_section_metrics.csv"),
        "Hybrid DSA": read_csv(
            ROOT / "runs/cracks_reserve_final/hybrid_dsa/per_section_metrics.csv"
        ),
        "SwinUNETR": read_csv(
            ROOT / "runs/cracks_reserve_final/swinunetr_f3chain/per_section_metrics.csv"
        ),
    }
    evidence_units = [
        {
            "dataset": "Thebe",
            "survey_cluster": "Exmouth Plateau, northwest Australia",
            "role": "dense field adaptation, validation selection, and spatially held-out frozen test",
            "label_status": "expert-labelled 3D blocks",
            "reported_units": len(thebe),
            "unit_type": "spatial test block",
            "independent_survey_clusters": 1,
            "supports_within_survey_inference": True,
            "supports_between_survey_generalization": False,
        },
        {
            "dataset": "CRACKS reserve",
            "survey_cluster": "F3, Dutch North Sea",
            "role": "sealed external annotation-domain reserve",
            "label_status": "independent expert section labels",
            "reported_units": len(cracks["U-Net"]),
            "unit_type": "2D section",
            "independent_survey_clusters": 1,
            "supports_within_survey_inference": True,
            "supports_between_survey_generalization": False,
        },
        {
            "dataset": "FORCE",
            "survey_cluster": "Ichthys, northwest Australia",
            "role": "frozen survey-domain stress test",
            "label_status": "no independent expert labels",
            "reported_units": 1,
            "unit_type": "fixed ROI",
            "independent_survey_clusters": 1,
            "supports_within_survey_inference": False,
            "supports_between_survey_generalization": False,
        },
        {
            "dataset": "Delft",
            "survey_cluster": "West Netherlands Basin",
            "role": "frozen survey-domain stress test with algorithmic TFL comparator",
            "label_status": "no independent expert labels; TFL is not ground truth",
            "reported_units": 1,
            "unit_type": "fixed ROI",
            "independent_survey_clusters": 1,
            "supports_within_survey_inference": False,
            "supports_between_survey_generalization": False,
        },
    ]

    thebe_differences = np.asarray(
        [float(row["hybrid_dice"]) - float(row["unet_dice"]) for row in thebe]
    )
    cracks_exact = {}
    for name, rows in cracks.items():
        cracks_exact[name] = np.asarray(
            [
                2 * int(row["tp"])
                / max(2 * int(row["tp"]) + int(row["fp"]) + int(row["fn"]), 1)
                for row in rows
            ],
            dtype=np.float64,
        )
    result = {
        "purpose": "Distinguish reported observational units from independent survey clusters",
        "expert_labelled_survey_clusters": 2,
        "unlabelled_external_stress_test_clusters": 2,
        "survey_level_confidence_interval_estimable": False,
        "reason": "Only two expert-labelled survey clusters are available and they have different adaptation and annotation roles; block/section resampling cannot estimate broad between-survey variance.",
        "permitted_claims": [
            "within-Thebe paired block comparison",
            "within-CRACKS paired section comparison under the sealed protocol",
            "descriptive model failure on fixed FORCE and Delft ROIs",
        ],
        "claims_not_supported": [
            "population-level geological generalization",
            "survey-independent accuracy",
            "accuracy on FORCE or Delft",
        ],
        "evidence_units": evidence_units,
        "within_survey_descriptive_checks": {
            "Thebe_Hybrid_minus_UNet_exact_Dice": {
                "mean": float(thebe_differences.mean()),
                "minimum": float(thebe_differences.min()),
                "maximum": float(thebe_differences.max()),
                "positive_blocks": int((thebe_differences > 0).sum()),
                "n_blocks": int(thebe_differences.size),
            },
            "CRACKS_reserve_macro_exact_Dice": {
                name: float(values.mean()) for name, values in cracks_exact.items()
            },
        },
    }
    (OUTPUT / "evidence_hierarchy.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "evidence_units.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_units[0]))
        writer.writeheader()
        writer.writerows(evidence_units)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

