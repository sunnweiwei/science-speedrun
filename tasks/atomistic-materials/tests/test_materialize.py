import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/data"))
import materialize as m


def item(source, content, **extra):
    return dict(
        bytes=source.stat().st_size,
        sha256=m.sha256(source),
        unpacked_bytes=len(content),
        unpacked_sha256=hashlib.sha256(content).hexdigest(),
        encoding="gzip",
        **extra,
    )


def test_verified_unpack_is_repeatable_and_rejects_corruption(tmp_path):
    content = b'{"frame":1}\n'
    source = tmp_path / "data.gz"
    source.write_bytes(gzip.compress(content))
    metadata = item(source, content)
    destination = tmp_path / "out/data.jsonl"
    m.unpack(source, destination, metadata)
    m.unpack(source, destination, metadata)
    assert destination.read_bytes() == content
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        m.unpack(source, destination, metadata)
    destination.unlink()
    source.write_bytes(source.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        m.unpack(source, destination, metadata)
    assert not destination.exists()


def test_unpack_enforces_expansion_limit_without_partial_output(tmp_path):
    source = tmp_path / "data.gz"
    source.write_bytes(gzip.compress(b"x" * 100))
    metadata = item(source, b"x" * 100)
    metadata["unpacked_bytes"] = 10
    destination = tmp_path / "out/data"
    with pytest.raises(ValueError, match="declared size"):
        m.unpack(source, destination, metadata)
    assert list(destination.parent.iterdir()) == []


def test_destinations_cannot_escape_through_traversal_or_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    for name in ("../secret", "/absolute", "escape/secret"):
        with pytest.raises(ValueError, match="path|Path"):
            m.within(root, name)


def test_materialization_fetches_only_explicit_components(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    files = []
    for group in m.GROUPS:
        content = (group + "\n").encode()
        source = remote / f"{group}.gz"
        source.write_bytes(gzip.compress(content))
        files.append(
            item(
                source,
                content,
                path=source.name,
                destination=f"{group}/data",
                group=group,
            )
        )
    manifest = remote / "manifest.json"
    manifest.write_text(json.dumps({"files": files, "required_groups": ["licenses"]}))
    pin = tmp_path / "release.json"
    pin.write_text(
        json.dumps(
            dict(
                repo_id="owner/data",
                revision="a" * 40,
                manifest_sha256=m.sha256(manifest),
            )
        )
    )
    monkeypatch.setattr(m, "PIN", pin)
    calls = []

    def fetch(url, target, expected_sha, expected_bytes=None, token=None):
        name = url.rsplit("/", 1)[1]
        calls.append(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(remote / name, target)
        m.checked(target, expected_bytes or target.stat().st_size, expected_sha)
        return target

    monkeypatch.setattr(m, "fetch", fetch)
    root = tmp_path / "data"
    m.materialize(root, ["training", "protocol"])
    assert set(calls) == {
        "manifest.json", "training.gz", "protocol.gz", "licenses.gz"
    }
    assert not (root / "records").exists()
    assert not (root / "eval-test").exists()
    assert not (root / "trained-models").exists()
    m.materialize(root, ["records"])
    assert (root / "records/data").read_text() == "records\n"
    m.materialize(root, ["initial-model", "eval-val"])
    assert (root / "initial-model/data").read_text() == "initial-model\n"
    assert (root / "eval-val/data").read_text() == "eval-val\n"
    assert not (root / "eval-test").exists()
    m.materialize(root, ["eval-test"])
    assert (root / "eval-test/data").read_text() == "eval-test\n"

    manifest.write_text(json.dumps({"files": files, "required_groups": ["eval-test"]}))
    pin_data = json.loads(pin.read_text())
    pin_data["manifest_sha256"] = m.sha256(manifest)
    pin.write_text(json.dumps(pin_data))
    with pytest.raises(ValueError, match="implicit components"):
        m.materialize(tmp_path / "fresh", ["training"])
