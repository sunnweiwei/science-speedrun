"""Apply the installed official S2 callback to a saved raw pixel probe."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import planetary_computer
import requests
from rslearn.data_sources.copernicus import get_harmonize_callback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.probe / "report.json").read_text())
    item_id = report["modalities"]["sentinel2"]["item_id"]
    response = requests.get(
        "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/"
        + item_id,
        timeout=60,
    )
    response.raise_for_status()
    url = planetary_computer.sign(response.json()["assets"]["product-metadata"]["href"])
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    tree = ET.fromstring(response.content)
    callback = get_harmonize_callback(tree)
    raw = np.load(args.probe / "sentinel2.npz")
    corrected = np.concatenate(
        [callback(b[None]) if callback else b[None] for b in raw["pixels"]], axis=0
    )
    # Verify official baseline04 handling, including nodata and clamped dark pixels.
    fixture = ET.fromstring("<root><BOA_ADD_OFFSET>-1000</BOA_ADD_OFFSET></root>")
    baseline04 = get_harmonize_callback(fixture)
    actual = baseline04(np.array([[[0, 1, 999, 1000, 1001, 2000]]], dtype=np.uint16))
    np.testing.assert_array_equal(
        actual, np.array([[[0, 1, 1, 1, 1, 1000]]], dtype=np.uint16)
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "product-metadata.xml").write_bytes(response.content)
    np.savez_compressed(
        args.output / "sentinel2-harmonized.npz", pixels=corrected, valid=raw["valid"]
    )
    result = {
        "item_id": item_id,
        "probe_report_sha256": hashlib.sha256(
            (args.probe / "report.json").read_bytes()
        ).hexdigest(),
        "product_metadata_sha256": hashlib.sha256(response.content).hexdigest(),
        "official_callback_source_sha256": hashlib.sha256(
            inspect.getsource(get_harmonize_callback).encode()
        ).hexdigest(),
        "offset_callback_present": callback is not None,
        "changed_pixels": int(np.count_nonzero(corrected != raw["pixels"])),
        "synthetic_baseline04_nodata_and_clamp_check": "passed",
        "target_used": False,
        "model_used": False,
        "scope": "One item and synthetic offset edge cases; full scene mosaics not yet verified",
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
