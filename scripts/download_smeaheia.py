import argparse
import hashlib
import http.cookiejar
import re
import urllib.parse
import urllib.request
from pathlib import Path


BASE_URL = "https://co2datashare.org"
DATASET_DOI = "10.11582/2021.00012"
RESOURCES = {
    "fault_sticks": {
        "id": "8e6889c8-14d8-47f2-97ee-04c68e896e42",
        "filename": "fault_sticks.zip",
        "expected_size": 565_769,
    },
    "reports": {
        "id": "1285a81c-a0a2-417d-9805-01e3d3f14fc2",
        "filename": "reports.zip",
        "expected_size": 17_749_528,
    },
    "seismic_3d": {
        "id": "705d84fe-3054-4ab4-951b-c045782078fb",
        "filename": "seismic_3d_surveys.zip",
        "expected_size": 1_889_055_556,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download licensed Smeaheia resources through the official "
            "CO2DataShare two-stage form. By running this command, the user "
            "confirms acceptance of the Smeaheia Dataset License."
        )
    )
    parser.add_argument(
        "resources",
        nargs="+",
        choices=sorted(RESOURCES),
        help="Dataset components to download.",
    )
    parser.add_argument("--country", required=True, help="Country/territory submitted to CO2DataShare.")
    parser.add_argument(
        "--affiliation",
        required=True,
        help="Company or institution name submitted to CO2DataShare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("external_data/smeaheia"),
        help="Destination directory (default: external_data/smeaheia).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract(pattern, text, label):
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find {label} in the CO2DataShare response.")
    return match.group(1)


def build_opener():
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [("User-Agent", "Hybrid-DSA-reproducibility/1.1")]
    return opener


def request_text(opener, request):
    with opener.open(request, timeout=120) as response:
        return response.read().decode("utf-8", errors="replace")


def download_resource(opener, resource_name, country, affiliation, output_dir, overwrite):
    resource = RESOURCES[resource_name]
    resource_id = resource["id"]
    page_url = f"{BASE_URL}/dataset/smeaheia-dataset/resource/{resource_id}"

    landing_html = request_text(opener, urllib.request.Request(page_url))
    first_token = extract(
        r'name="_csrf_token"\s+value="([^"]+)"', landing_html, "initial CSRF token"
    )
    first_payload = urllib.parse.urlencode({"_csrf_token": first_token}).encode("utf-8")
    download_form_html = request_text(
        opener,
        urllib.request.Request(
            page_url,
            data=first_payload,
            headers={"Referer": page_url},
            method="POST",
        ),
    )

    form_match = re.search(
        r'<form[^>]+id="resource-download"[^>]+action=([^\s>]+)[^>]*>(.*?)</form>',
        download_form_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not form_match:
        raise RuntimeError("CO2DataShare did not return the download registration form.")
    action = form_match.group(1).strip('"\'')
    form_html = form_match.group(2)
    second_token = extract(
        r'name="_csrf_token"\s+value="([^"]+)"', form_html, "download CSRF token"
    )
    returned_id = extract(r'name="res_id"[^>]+value=([a-f0-9-]+)', form_html, "resource id")
    if returned_id != resource_id:
        raise RuntimeError(f"Resource-id mismatch: expected {resource_id}, received {returned_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / resource["filename"]
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --overwrite to replace it.")

    final_payload = urllib.parse.urlencode(
        {
            "_csrf_token": second_token,
            "res_id": resource_id,
            "affiliation": affiliation,
            "country": country,
        }
    ).encode("utf-8")
    final_url = urllib.parse.urljoin(BASE_URL, action)
    request = urllib.request.Request(
        final_url,
        data=final_payload,
        headers={"Referer": page_url},
        method="POST",
    )

    digest = hashlib.sha256()
    bytes_written = 0
    temporary = destination.with_suffix(destination.suffix + ".part")
    with opener.open(request, timeout=300) as response, temporary.open("wb") as output:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type.lower():
            preview = response.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Expected ZIP data but received HTML: {preview[:240]!r}")
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            bytes_written += len(chunk)
            print(f"{resource_name}: {bytes_written / (1024 ** 2):.1f} MiB", end="\r", flush=True)

    if bytes_written < 4:
        raise RuntimeError(f"Downloaded file is empty: {temporary}")
    with temporary.open("rb") as stream:
        if stream.read(4) != b"PK\x03\x04":
            raise RuntimeError(f"Downloaded file is not a ZIP archive: {temporary}")
    temporary.replace(destination)
    print(
        f"{resource_name}: wrote {bytes_written} bytes to {destination} "
        f"(sha256={digest.hexdigest()})"
    )
    if bytes_written != resource["expected_size"]:
        print(
            f"WARNING: provider metadata reports {resource['expected_size']} bytes; "
            f"downloaded {bytes_written} bytes. Preserve the checksum and inspect the archive."
        )


def main():
    args = parse_args()
    opener = build_opener()
    print(f"Smeaheia dataset DOI: {DATASET_DOI}")
    for resource_name in args.resources:
        download_resource(
            opener,
            resource_name,
            args.country,
            args.affiliation,
            args.output_dir,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
