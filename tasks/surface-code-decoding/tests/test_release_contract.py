"""Executable release, grading, corruption, and label-isolation contracts."""
from __future__ import annotations
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import zipfile
import numpy as np
import pytest

TASK = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit" if (Path(__file__).resolve().parents[1] / "environment" / "baseline" / "alphaqubit").is_dir() else Path(__file__).resolve().parents[1]  # environment/baseline/alphaqubit here, /opt/task in the verifier image
sys.path.insert(0, str(PROJECT))
from qecbench.data import SCHEMA, DevelopmentDataset, load_condition, load_manifest, read_npz, sha256, write_json
from qecbench.scoring import InvalidPrediction, normalize, probabilities, score
from qecbench.stage_validation import stage
from baselines.alphaqubit_artifact import EXPECTED_GROUPS


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


materializer = module("qec_materializer", TASK / "environment/data/materialize.py")
supervisor = module("qec_supervisor", TASK / "tests/main_verifier.py")
grader = module("qec_grader", TASK / "tests/grader.py")


@pytest.fixture
def suite(tmp_path: Path):
    inputs, labels, predictions = (tmp_path / name for name in ("inputs", "labels", "predictions"))
    for folder in (inputs, labels, predictions):
        folder.mkdir()
    rows, targets = [], []
    for group in EXPECTED_GROUPS:
        distance, basis = int(group[1]), group[4]
        center = group.split("center_")[1]
        for rounds in range(1, 26, 2):
            name = f"surface_code_b{basis}_d{distance}_r{rounds:02}_center_{center}"
            folder = inputs / name
            folder.mkdir()
            ids = np.array([991, 27, 64, 118], dtype=np.uint64)
            detectors = np.zeros((4, (rounds * (distance**2 - 1) + 7) // 8), dtype=np.uint8)
            np.savez(folder / "arrays.npz", sample_ids=ids, detectors=detectors)
            (folder / "circuit_ideal.stim").write_text("fixture circuit\n")
            target = np.array([0, 0, 0, 1], dtype=np.uint8)
            np.savez(labels / f"{name}.npz", sample_ids=ids, labels=target)
            np.savez(predictions / f"{name}.npz", sample_ids=ids[::-1], probabilities=np.array([1, 0, 0, 0], dtype=np.float32))
            rows.append({"name": name, "group": group, "distance": distance, "basis": basis, "rounds": rounds,
                "shots": 4, "num_detectors": rounds * (distance**2 - 1), "arrays": f"{name}/arrays.npz",
                "arrays_sha256": sha256(folder / "arrays.npz"), "circuit": f"{name}/circuit_ideal.stim",
                "circuit_sha256": sha256(folder / "circuit_ideal.stim")})
            targets.append({"name": name, "arrays": f"{name}.npz", "arrays_sha256": sha256(labels / f"{name}.npz")})
    write_json(inputs / "manifest.json", {"schema": SCHEMA, "split": "test", "conditions": rows})
    write_json(labels / "manifest.json", {"schema": "surface-code-decoding-labels/v1", "input_manifest_sha256": sha256(inputs / "manifest.json"), "conditions": targets})
    return inputs, labels, predictions, rows


def test_frozen_normalization_anchors():
    config = json.loads((PROJECT / "score_config.json").read_text())
    assert normalize(config["lower_bound"], config)["score"] == 0
    assert normalize(config["upper_bound"], config)["score"] == 1
    assert normalize(1, config)["score_unclipped"] > 1
    assert normalize(1, config)["score"] == normalize(1, config)["score_unclipped"]   # no upper clip (v1.1.0)
    assert normalize(0, config)["score_unclipped"] < 0
    assert normalize(0, config)["score"] == 0


def test_full_suite_reordering_and_perfect_score(suite):
    inputs, labels, predictions, _ = suite
    result = score(inputs, predictions, labels, PROJECT / "score_config.json")
    assert result["shots"] == 520
    assert result["performance"] == 1
    assert result["score"] == result["score_unclipped"] > 1   # perfect decoding is above the TN anchor; no upper clip
    assert result["failure_by_distance"] == {"d3": 0, "d5": 0}


def test_missing_prediction_is_invalid_not_infrastructure(suite):
    inputs, labels, predictions, rows = suite
    (predictions / f"{rows[0]['name']}.npz").unlink()
    with pytest.raises(InvalidPrediction):
        score(inputs, predictions, labels, PROJECT / "score_config.json")


@pytest.mark.parametrize("failure", ["nan", "float64", "duplicate", "extra-id", "zip-garbage", "encrypted", "compression"])
def test_bad_outputs_fail_closed(suite, failure):
    inputs, _, predictions, rows = suite
    row = rows[0]
    ids = load_condition(inputs, row)["sample_ids"]
    values = np.zeros(4, dtype=np.float32)
    if failure == "nan":
        values[0] = np.nan
    elif failure == "float64":
        values = values.astype(np.float64)
    elif failure == "duplicate":
        ids[1] = ids[0]
    elif failure == "extra-id":
        ids[1] += 1
    path = predictions / f"{row['name']}.npz"
    np.savez(path, sample_ids=ids, probabilities=values)
    if failure == "zip-garbage":
        path.write_bytes(b"not-a-zip")
    if failure in ("encrypted", "compression"):
        payload = bytearray(path.read_bytes())
        for marker, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
            cursor = 0
            while (cursor := payload.find(marker, cursor)) >= 0:
                payload[cursor + offset + (2 if failure == "compression" else 0)] = 99 if failure == "compression" else 1
                cursor += 4
        path.write_bytes(payload)
    with pytest.raises(InvalidPrediction):
        probabilities(predictions, row, load_condition(inputs, row)["sample_ids"])


def test_trusted_label_corruption_is_infrastructure(suite):
    inputs, labels, predictions, rows = suite
    (labels / f"{rows[0]['name']}.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Trusted label checksum"):
        score(inputs, predictions, labels, PROJECT / "score_config.json")


def test_label_free_schema_and_development_split_guard(suite):
    inputs, _, _, rows = suite
    row = rows[0]
    path = inputs / row["arrays"]
    arrays = load_condition(inputs, row)
    np.savez(path, **arrays, labels=np.zeros(4, dtype=np.uint8))
    with pytest.raises(ValueError, match="Wrong array fields"):
        load_condition(inputs, row)
    with pytest.raises(ValueError, match="train/validation only"):
        DevelopmentDataset(inputs, "test")


def test_symlink_prediction_rejected(suite, tmp_path):
    inputs, _, predictions, rows = suite
    row = rows[0]
    path = predictions / f"{row['name']}.npz"
    other = tmp_path / "outside.npz"
    path.replace(other)
    path.symlink_to(other)
    with pytest.raises(InvalidPrediction):
        probabilities(predictions, row, load_condition(inputs, row)["sample_ids"])


def test_npz_bomb_and_forged_shape_rejected(tmp_path):
    path = tmp_path / "bomb.npz"
    np.savez_compressed(path, probabilities=np.zeros(100000, dtype=np.float32))
    with pytest.raises(ValueError, match="expanded content"):
        read_npz(path, 48)
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False, "shape": (2**60,)})
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("probabilities.npy", stream.getvalue())
    with pytest.raises(ValueError, match="shape disagrees"):
        read_npz(path, 48)


def test_materializer_rejects_unsafe_tar_before_write(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        item = tarfile.TarInfo("development/../../escaped")
        item.size = 1
        output.addfile(item, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="Unsafe archive"):
        materializer.extract(archive, tmp_path / "dest", "development", 10)
    assert not (tmp_path / "escaped").exists()


def test_labeled_validation_staging(suite, tmp_path):
    inputs, labels, _, rows = suite
    manifest = json.loads((inputs / "manifest.json").read_text())
    manifest["split"] = "validation"
    for row in rows:
        path = inputs / row["arrays"]
        arrays = load_condition(inputs, row)
        np.savez(path, **arrays, labels=np.zeros(4, dtype=np.uint8))
        row["arrays_sha256"] = sha256(path)
    manifest["conditions"] = rows
    write_json(inputs / "manifest.json", manifest)
    with pytest.raises(ValueError, match="physically label-free"):
        load_manifest(inputs)
    output = tmp_path / "stage"
    stage(inputs, output, limit=3, seed=1, group=rows[0]["group"])
    staged = load_manifest(output)
    assert len(staged["conditions"]) == 13
    assert {row["shots"] for row in staged["conditions"]} == {3}
    assert set(load_condition(output, staged["conditions"][0])) == {"sample_ids", "detectors"}
    predictions = tmp_path / "subset-predictions"
    predictions.mkdir()
    for row in staged["conditions"]:
        arrays = load_condition(output, row)
        np.savez(predictions / f"{row['name']}.npz", sample_ids=arrays["sample_ids"],
                 probabilities=np.zeros(row["shots"], dtype=np.float32))
    report = score(output, predictions, inputs, PROJECT / "score_config.json", require_full_suite=False, validation=True)
    assert report["shots"] == 39 and report["performance"] == 1


def test_root_grader_isolation_topology():
    import tomllib
    import yaml
    compose = yaml.safe_load((TASK / "tests/docker-compose.yaml").read_text())
    for service in compose["services"].values():
        assert service["network_mode"] == "none"
    main = compose["services"]["main"]
    assert main["deploy"]["resources"]["reservations"]["devices"] == [
        {"driver": "nvidia", "count": 1, "capabilities": ["gpu"]}
    ]
    development = yaml.safe_load((TASK / "environment/docker-compose.yaml").read_text())
    # A Compose network override prevents Harbor's model API allowlist sidecar
    # from attaching; the runner must apply the default-deny task policy.
    assert "network_mode" not in development["services"]["main"]
    assert "networks" not in development["services"]["main"]
    task = tomllib.loads((TASK / "task.toml").read_text())
    assert task["environment"]["network_mode"] == "no-network"
    assert task["verifier"]["environment"]["network_mode"] == "no-network"
    local = yaml.safe_load((TASK / "tests/docker-compose.local.yaml").read_text())
    assert "gpus" not in local["services"]["main"]
    assert all(not isinstance(volume, dict) or volume["target"] != "/private" for volume in main["volumes"])
    trusted = compose["services"]["grader"]
    assert any(isinstance(volume, dict) and volume["target"] == "/private" and volume["read_only"] for volume in trusted["volumes"])
    assert all(not isinstance(volume, dict) or volume["target"] != "/workspace/submission" for volume in trusted["volumes"])


def test_infrastructure_failure_emits_no_reward(tmp_path):
    sealed, out = tmp_path / "sealed", tmp_path / "results"
    sealed.mkdir()
    token = "a" * 32
    grader.grade(sealed, tmp_path / "absent", tmp_path / "private", PROJECT / "score_config.json", out,
                 {"run_id": token, "snapshot_sha256": "0" * 64})
    assert json.loads((out / token / "report.json").read_text())["status"] == "infrastructure_error"
    assert not (out / token / "reward.json").exists()


def test_invalid_submission_has_zero_not_missing_reward(tmp_path):
    supervisor.invalid(tmp_path, "invalid_submission", ValueError("fixture"))
    assert json.loads((tmp_path / "reward.json").read_text()) == {"reward": 0.0}
    assert json.loads((tmp_path / "report.json").read_text())["model_quality_measured"] is False


def test_submission_manifest_nonobject_rejected(tmp_path):
    for name in ("infer.sh", "README.md", "requirements.lock"):
        (tmp_path / name).write_text("fixture")
    (tmp_path / "model_manifest.json").write_text("[]")
    with pytest.raises(ValueError, match="Unsupported submission manifest"):
        supervisor.validate_submission(tmp_path)


@pytest.mark.parametrize("case", ["nested-inventory", "symlink-root", "fifo"])
def test_materializer_exact_existing_inventory(tmp_path, case):
    root = tmp_path / "development"
    root.mkdir()
    write_json(root / "inventory.json", {"schema": "surface-code-decoding-inventory/v1", "files": {}})
    materializer.verify_role(root)
    if case == "nested-inventory":
        (root / "unexpected").mkdir()
        (root / "unexpected/inventory.json").write_text("undeclared")
    elif case == "symlink-root":
        link = tmp_path / "linked"
        link.symlink_to(root, target_is_directory=True)
        root = link
    else:
        import os
        os.mkfifo(root / "special")
    with pytest.raises(ValueError):
        materializer.verify_role(root)


def test_invalid_prediction_does_not_hide_late_infrastructure_corruption(suite):
    inputs, labels, predictions, rows = suite
    (predictions / f"{rows[0]['name']}.npz").unlink()
    (labels / f"{rows[-1]['name']}.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Trusted label checksum"):
        score(inputs, predictions, labels, PROJECT / "score_config.json")


def test_primary_uses_equal_distance_group_duration_weights(suite):
    inputs, labels, predictions, rows = suite
    chosen = [next(row for row in rows if row["distance"] == distance) for distance in (3, 5)]
    for row in chosen:
        ids = load_condition(inputs, row)["sample_ids"]
        with np.load(labels / f"{row['name']}.npz", allow_pickle=False) as truth:
            wrong = 1 - truth["labels"]
        np.savez(predictions / f"{row['name']}.npz", sample_ids=ids, probabilities=wrong.astype(np.float32))
    report = score(inputs, predictions, labels, PROJECT / "score_config.json")
    # One failed duration in one group per distance: d3 has eight groups; d5 two.
    expected = 0.5 * (1 / (8 * 13) + 1 / (2 * 13))
    assert report["balanced_logical_failure"] == pytest.approx(expected)
    assert report["balanced_logical_failure"] != pytest.approx(2 / 130)


@pytest.fixture
def reusable_materialization(tmp_path):
    import argparse
    source = tmp_path / "source"
    role = source / "development"
    role.mkdir(parents=True)
    (role / "payload.bin").write_bytes(b"verified data")
    write_json(role / "inventory.json", {"schema": "surface-code-decoding-inventory/v1", "files": {
        "payload.bin": {"bytes": 13, "sha256": sha256(role / "payload.bin")}}})
    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    archive = archive_root / "development.tar.gz"
    lock = tmp_path / "lock.json"

    def publish():
        with tarfile.open(archive, "w:gz") as tar:
            for path in sorted(role.iterdir()):
                # Avoid NSS lookups in gettarinfo on cluster hosts.
                item = tarfile.TarInfo(f"development/{path.name}")
                item.size = path.stat().st_size
                with path.open("rb") as stream:
                    tar.addfile(item, stream)
        write_json(lock, {"schema": "surface-code-decoding-hf-release/v1", "release": "fixture",
            "artifacts": {"development": {"repo_id": "local/fixture", "revision": "0" * 40,
                "filename": archive.name, "bytes": archive.stat().st_size, "sha256": sha256(archive),
                "expanded_bytes": sum(path.stat().st_size for path in role.iterdir())}}})

    publish()
    args = argparse.Namespace(from_sources=False, lock=lock, output=tmp_path / "restored",
                              roles=["development"], archive_dir=archive_root, cache_dir=None)
    materializer.run(args)
    return args, role, publish


def test_reuse_skips_payload_extraction(reusable_materialization, monkeypatch):
    args, _, _ = reusable_materialization

    def forbidden_extract(*args, **kwargs):
        raise AssertionError("Reused roles must not extract a second payload tree")

    monkeypatch.setattr(materializer, "extract", forbidden_extract)
    monkeypatch.setattr(materializer.tempfile, "TemporaryDirectory", forbidden_extract)
    assert materializer.run(args)["status"] == "verified"
    assert (args.output / "development/payload.bin").read_bytes() == b"verified data"
    assert not list(args.output.glob(".qec-materialize-*"))


def test_reuse_rejects_corrupt_existing_payload(reusable_materialization, monkeypatch):
    args, _, _ = reusable_materialization
    (args.output / "development/payload.bin").write_bytes(b"corrupt data!")
    monkeypatch.setattr(materializer, "extract", lambda *a, **k: pytest.fail("Unexpected extraction"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        materializer.run(args)


def test_reuse_rejects_different_hash_verified_inventory(reusable_materialization, monkeypatch):
    args, source, publish = reusable_materialization
    (source / "payload.bin").write_bytes(b"different release")
    write_json(source / "inventory.json", {"schema": "surface-code-decoding-inventory/v1", "files": {
        "payload.bin": {"bytes": 17, "sha256": sha256(source / "payload.bin")}}})
    publish()  # Lock authenticates this different archive, but the target is old.
    monkeypatch.setattr(materializer, "extract", lambda *a, **k: pytest.fail("Unexpected extraction"))
    with pytest.raises(ValueError, match="another release"):
        materializer.run(args)
    assert (args.output / "development/payload.bin").read_bytes() == b"verified data"


@pytest.mark.parametrize("case", ["symlink", "directory", "oversized", "unsafe-path"])
def test_archive_inventory_validates_paths_types_and_size(tmp_path, case):
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        item = tarfile.TarInfo("development/inventory.json")
        payload = b"{}"
        if case == "symlink":
            item.type = tarfile.SYMTYPE
            item.linkname = "outside"
        elif case == "directory":
            item.type = tarfile.DIRTYPE
        else:
            item.size = len(payload)
        tar.addfile(item, io.BytesIO(payload) if item.isfile() else None)
        if case == "unsafe-path":
            other = tarfile.TarInfo("development/../outside")
            other.size = 1
            tar.addfile(other, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        materializer.archive_inventory(archive, "development", 1 if case == "oversized" else 100)
