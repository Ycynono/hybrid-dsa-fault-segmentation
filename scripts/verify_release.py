import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "U-Net": ROOT / "checkpoints/unet3d_thebe_e8.pt",
    "Hybrid DSA": ROOT / "checkpoints/hybrid_dsa_thebe_e8.pt",
    "SwinUNETR F3-chain": ROOT / "checkpoints/swinunetr_f3chain_thebe_e8.pt",
}
REQUIRED = [
    ROOT / "protocol/thebe_manifest.json",
    ROOT / "protocol/cracks_expert_split.json",
    ROOT / "results/thebe/summary_statistics.json",
    ROOT / "results/cracks/reserve_results.json",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    protocol = json.loads((ROOT / "protocol/FINAL_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    expected = {item["name"]: item["sha256"] for item in protocol["models"]}
    failures = []
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            failures.append(f"missing checkpoint: {path.relative_to(ROOT)}")
            continue
        actual = sha256(path)
        ok = actual == expected[name]
        print(f"{'OK' if ok else 'FAIL'} {name}: {actual}")
        if not ok:
            failures.append(f"hash mismatch: {name}")
    for path in REQUIRED:
        ok = path.exists()
        print(f"{'OK' if ok else 'FAIL'} required: {path.relative_to(ROOT)}")
        if not ok:
            failures.append(f"missing required file: {path.relative_to(ROOT)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Release verification passed.")


if __name__ == "__main__":
    main()

