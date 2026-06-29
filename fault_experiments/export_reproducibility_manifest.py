import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import einops
import matplotlib
import monai
import numpy
import PIL
import pyvista
import scipy
import skimage
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    "runs/thebe_adaptation/unet3d_e8/best.pt",
    "runs/thebe_adaptation/dsa_hybrid_replay_e8/best.pt",
    "runs/thebe_adaptation/swin_unetr_f3chain_e8/best.pt",
    "runs/baselines/swin_unetr_synthetic_e50/best.pt",
    "runs/thebe_final_test2_7/statistics/summary_statistics.json",
    "runs/cracks_audit_frozen/audit_statistics.json",
    "runs/cracks_reserve_final/reserve_results.json",
    "runs/cracks_final_statistics.json",
    "FINAL_PROTOCOL_LOCK.json",
    "processed_data/thebe_official/manifest.json",
    "processed_data/cracks_external_v1/expert_split.json",
    "external_data/CRACKS/Fault segmentations.zip",
    "external_data/CRACKS/images.zip",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    files = []
    for relative in ARTIFACTS:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "monai": monai.__version__,
            "einops": einops.__version__,
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
            "scikit_image": skimage.__version__,
            "pyvista": pyvista.__version__,
        },
        "fixed_thresholds": {
            "U-Net": 0.50,
            "Hybrid DSA": 0.15,
            "SwinUNETR F3-chain": 0.40,
            "selection_source": "Thebe val1-val2 only",
        },
        "data_policy": {
            "thebe": "train1-train9 train; val1-val2 selection; test1 exposed pilot; test2-test7 frozen evaluation",
            "cracks": "20 preregistered audit sections evaluated; 20 reserve sections opened once after final protocol lock",
            "cracks_reserve_opened": True,
        },
        "files": files,
    }
    output = ROOT / "reproducibility_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
