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
    ROOT / "results/smeaheia/data_quality_audit.json",
    ROOT / "results/smeaheia/benchmark_metadata.json",
    ROOT / "results/smeaheia/results.json",
    ROOT / "results/smeaheia/fault_object_results.json",
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
    if not failures:
        audit = json.loads(
            (ROOT / "results/smeaheia/data_quality_audit.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (ROOT / "results/smeaheia/benchmark_metadata.json").read_text(encoding="utf-8")
        )
        results = json.loads(
            (ROOT / "results/smeaheia/results.json").read_text(encoding="utf-8")
        )
        objects = json.loads(
            (ROOT / "results/smeaheia/fault_object_results.json").read_text(encoding="utf-8")
        )
        smeaheia_checks = {
            "data-quality decision": audit.get("decision") == "pass",
            "ROI locked before prediction": metadata["selection_policy"].get(
                "locked_before_prediction"
            )
            is True,
            "36 fault objects in ROI": metadata["expert_reference"].get(
                "fault_object_count_in_roi"
            )
            == 36,
            "four frozen methods": len(results.get("summaries", [])) == 4,
            "36 object metrics": objects.get("fault_object_count") == 36,
        }
        for label, ok in smeaheia_checks.items():
            print(f"{'OK' if ok else 'FAIL'} Smeaheia: {label}")
            if not ok:
                failures.append(f"Smeaheia verification failed: {label}")
    if failures:
        raise SystemExit("\n".join(failures))
    print("Release verification passed.")


if __name__ == "__main__":
    main()
