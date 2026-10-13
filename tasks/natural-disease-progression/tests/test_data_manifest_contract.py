from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


DATA_DIR = Path("/task/environment/data")
sys.path.insert(0, str(DATA_DIR))
from validate_longitudinal_v2 import validate_manifest_contract  # noqa: E402


def write_contract(root: Path) -> None:
    horizons = list(range(30, 361, 30)) + [365]
    diseases = ["A00", "B00"]
    (root / "public/validation").mkdir(parents=True)
    (root / "public/test").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "natural-disease-progression/v2",
        "protocol": "longitudinal-v2",
        "horizons_days": horizons,
        "scored_diseases": diseases,
        "num_scored_diseases": len(diseases),
        "primary_metric": "macro_time_averaged_auroc",
    }))
    for split in ("validation", "test"):
        (root / f"public/{split}/manifest.json").write_text(json.dumps({
            "protocol": "longitudinal-v2",
            "horizons_days": horizons,
            "scored_diseases": diseases,
            "split": split,
            "primary_metric": "macro_time_averaged_auroc",
        }))


def test_manifest_contract_accepts_active_protocol(tmp_path: Path) -> None:
    write_contract(tmp_path)
    _, diseases = validate_manifest_contract(tmp_path)
    assert diseases == ["A00", "B00"]


@pytest.mark.parametrize("field,value", [
    ("protocol", "legacy"),
    ("horizons_days", [365]),
    ("primary_metric", "age_sex_normalized_disease_time_brier"),
    ("scored_diseases", ["B00", "A00"]),
])
def test_manifest_contract_rejects_stale_or_misaligned_split(
    tmp_path: Path, field: str, value: object,
) -> None:
    write_contract(tmp_path)
    path = tmp_path / "public/validation/manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        validate_manifest_contract(tmp_path)
