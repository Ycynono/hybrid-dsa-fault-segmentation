import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path


PATTERN = re.compile(r"^(seis|fault)(train[1-9]|val[1-2]|test[1-7])\.npz$")
PRINT_LOCK = threading.Lock()


def md5sum(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(item, output_dir, retries=8):
    target = output_dir / item["name"]
    expected_size = item["bytes"]
    expected_md5 = item["md5"]
    if target.exists() and target.stat().st_size == expected_size:
        if md5sum(target) == expected_md5:
            return {"name": item["name"], "status": "verified", "bytes": expected_size}
        raise RuntimeError(f"Existing full-size file has wrong MD5: {target}")

    url = f"https://dataverse.harvard.edu/api/access/datafile/{item['id']}"
    for attempt in range(1, retries + 1):
        existing = target.stat().st_size if target.exists() else 0
        headers = {"User-Agent": "MS-SGA-Net-Thebe-Downloader/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if existing and status != 206:
                    raise RuntimeError(
                        f"Server ignored Range for {item['name']}: status={status}, existing={existing}"
                    )
                mode = "ab" if existing else "wb"
                with target.open(mode) as handle:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if target.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch {item['name']}: {target.stat().st_size}/{expected_size}"
                )
            actual_md5 = md5sum(target)
            if actual_md5 != expected_md5:
                raise RuntimeError(f"MD5 mismatch {item['name']}: {actual_md5}/{expected_md5}")
            return {"name": item["name"], "status": "downloaded", "bytes": expected_size}
        except Exception as error:
            with PRINT_LOCK:
                print(f"RETRY {item['name']} attempt={attempt}/{retries}: {error}", flush=True)
            if attempt == retries:
                raise
            time.sleep(min(5 * attempt, 30))


def progress_monitor(items, output_dir, stop_event):
    expected = sum(item["bytes"] for item in items)
    while not stop_event.wait(20):
        present = sum(
            min((output_dir / item["name"]).stat().st_size, item["bytes"])
            if (output_dir / item["name"]).exists()
            else 0
            for item in items
        )
        with PRINT_LOCK:
            print(f"PROGRESS {present / 1e9:.2f}/{expected / 1e9:.2f} GB ({100 * present / expected:.1f}%)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Resumable Harvard Dataverse Thebe downloader.")
    parser.add_argument("--metadata", default="thebe_dataverse_metadata.json")
    parser.add_argument("--output-dir", default="external_data/Thebe/complete")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    items = []
    for entry in metadata["data"]["latestVersion"]["files"]:
        name = entry["label"]
        if PATTERN.match(name):
            items.append(
                {
                    "name": name,
                    "id": entry["dataFile"]["id"],
                    "bytes": entry["dataFile"]["filesize"],
                    "md5": entry["dataFile"]["md5"].lower(),
                }
            )
    items.sort(key=lambda item: (not item["name"].startswith("fault"), item["name"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse already downloaded files without copying partial files.
    legacy_dir = Path("external_data/Thebe/test")
    for item in items:
        legacy = legacy_dir / item["name"]
        target = output_dir / item["name"]
        if not target.exists() and legacy.exists() and legacy.stat().st_size == item["bytes"]:
            if md5sum(legacy) == item["md5"]:
                try:
                    os.link(legacy, target)
                except OSError:
                    target.write_bytes(legacy.read_bytes())

    stop_event = threading.Event()
    monitor = threading.Thread(target=progress_monitor, args=(items, output_dir, stop_event), daemon=True)
    monitor.start()
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_one, item, output_dir): item for item in items}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                with PRINT_LOCK:
                    print(f"COMPLETE {result['name']} {result['status']}", flush=True)
    finally:
        stop_event.set()
        monitor.join(timeout=2)

    manifest = {
        "dataset": "A gigabyte interpreted seismic dataset for automatic fault recognition",
        "persistent_id": "doi:10.7910/DVN/YBYGBK",
        "license": "CC BY 4.0",
        "file_count": len(results),
        "total_bytes": sum(result["bytes"] for result in results),
        "files": sorted(results, key=lambda row: row["name"]),
    }
    (output_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"file_count": len(results), "total_gb": manifest["total_bytes"] / 1e9}, indent=2))


if __name__ == "__main__":
    main()
