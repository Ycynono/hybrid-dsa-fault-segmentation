import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare all downloaded Thebe train/val/test blocks.")
    parser.add_argument("--download-root", default="external_data/Thebe/complete")
    parser.add_argument("--output-root", default="processed_data/thebe_official")
    parser.add_argument("--include-test1", action="store_true")
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    download_root = Path(args.download_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    requested_splits = {value.strip() for value in args.splits.split(",") if value.strip()}
    groups = [
        (split, indices)
        for split, indices in [("train", range(1, 10)), ("val", range(1, 3)), ("test", range(1, 8))]
        if split in requested_splits
    ]
    for split, indices in groups:
        for index in indices:
            if split == "test" and index == 1 and not args.include_test1:
                continue
            seismic = download_root / f"seis{split}{index}.npz"
            fault = download_root / f"fault{split}{index}.npz"
            if not seismic.exists() or not fault.exists():
                raise FileNotFoundError(f"Missing pair: {seismic}, {fault}")
            block_id = f"{split}{index}"
            block_dir = output_root / split / block_id
            metadata_file = block_dir / "metadata.json"
            if metadata_file.exists():
                print(f"SKIP prepared {block_id}", flush=True)
            else:
                command = [
                    sys.executable,
                    "-m",
                    "fault_experiments.prepare_thebe_block",
                    "--seismic",
                    str(seismic),
                    "--fault",
                    str(fault),
                    "--output-dir",
                    str(block_dir),
                ]
                print(f"PREPARE {block_id}", flush=True)
                subprocess.run(command, check=True)
    prepared = []
    for split in ("train", "val", "test"):
        split_dir = output_root / split
        if not split_dir.exists():
            continue
        for block_dir in sorted(split_dir.iterdir(), key=lambda path: path.name):
            metadata_file = block_dir / "metadata.json"
            if metadata_file.exists():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                prepared.append({"id": block_dir.name, "split": split, **metadata})

    manifest = {
        "dataset": "Thebe Gas Field expert-interpreted fault dataset",
        "test_policy": "test1 exposed for development; test2-test7 reserved for final frozen evaluation",
        "blocks": prepared,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(prepared)} blocks")


if __name__ == "__main__":
    main()
