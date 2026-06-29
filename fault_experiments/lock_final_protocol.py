import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "FINAL_PROTOCOL_LOCK.json"


def main():
    if LOCK.exists():
        raise RuntimeError(f"Protocol lock already exists: {LOCK}")
    reproducibility = json.loads(
        (ROOT / "reproducibility_manifest.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (ROOT / "processed_data/cracks_external_v1/expert_split.json").read_text(
            encoding="utf-8"
        )
    )
    hashes = {row["path"]: row["sha256"] for row in reproducibility["files"]}
    protocol = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One-time final evaluation on the sealed CRACKS reserve",
        "no_further_model_or_threshold_tuning_after_lock": True,
        "models": [
            {
                "name": "U-Net",
                "checkpoint": "runs/thebe_adaptation/unet3d_e8/best.pt",
                "sha256": hashes["runs/thebe_adaptation/unet3d_e8/best.pt"],
                "threshold": 0.50,
            },
            {
                "name": "Hybrid DSA",
                "checkpoint": "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt",
                "sha256": hashes[
                    "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt"
                ],
                "threshold": 0.15,
            },
            {
                "name": "SwinUNETR F3-chain",
                "checkpoint": "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt",
                "sha256": hashes[
                    "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt"
                ],
                "threshold": 0.40,
            },
        ],
        "threshold_source": "Thebe val1-val2 only; no CRACKS calibration",
        "reserve_sections": split["reserve_sections"],
        "primary_metrics": ["micro exact Dice", "macro exact Dice", "macro 3-pixel tolerant Dice"],
        "label_policy": "combine expert certain and uncertain fault labels",
        "paired_inference_unit": "CRACKS section, with single-survey dependence caveat",
    }
    LOCK.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(LOCK)


if __name__ == "__main__":
    main()
