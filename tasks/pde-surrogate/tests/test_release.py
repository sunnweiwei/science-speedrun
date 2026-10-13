"""Offline tests for the public training release: what it may contain, and what it must not."""
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(ROOT / "environment" / "evaluation"))
import build_release  # noqa: E402
import pde_io  # noqa: E402

SIMS_ATTRS = {"PDE": "Burgers", "Fields": ["Velocity X", "Velocity Y"], "Fields Scheme": "VV", "Dt": 0.1,
              "Domain Extent": [1.0, 1.0], "Boundary Conditions": ["periodic"] * 4, "Constants": ["Viscosity"]}


def write_corpus_file(path, arrays, norm_value=0.0):
    """A PBDL-style file with the `sims` attributes the readers require, like the upstream corpus."""
    with h5py.File(path, "w") as f:
        g = f.create_group("sims")
        for k, v in SIMS_ATTRS.items():
            g.attrs[k] = v
        for i, a in enumerate(arrays):
            d = g.create_dataset(f"sim{i}", data=a)
            d.attrs["Viscosity"] = 0.01 * (i + 1)
        c = arrays[0].shape[1]
        f.create_dataset("norm_fields_sca_mean", data=np.full((c, 1, 1), norm_value))
        f.create_dataset("norm_fields_sca_std", data=np.ones((c, 1, 1)))


def assert_readable(path, trajectories):
    """The check whose absence let the first release ship: open with pde_io, not just h5py."""
    for t in trajectories:
        frames, meta = pde_io.read_window("ape2d", path, t, 0, 1, 0)
        assert frames.ndim == 4 and meta["pde"] == SIMS_ATTRS["PDE"] and meta["fields"] == SIMS_ATTRS["Fields"]
        assert meta["constants"] == {"Viscosity": 0.01 * (t + 1)}


@pytest.fixture(scope="module")
def plan():
    lock = json.loads((DATA / "sources.lock.json").read_text())
    split = json.loads((DATA / "split.json").read_text())
    return build_release.public_payload(lock, split)


def test_every_held_out_only_file_goes_to_eval_not_train(plan):
    payload, _ = plan
    split = json.loads((DATA / "split.json").read_text())
    pools = {(e["source"], e["file"]) for p in ("train", "valid") for e in split["pools"][p]}
    for c in split["cases"]:
        if "file" not in c:
            continue
        key = (c["source"], c["file"])
        if key not in pools and key in payload:
            assert payload[key][0] == "eval", f"{key} holds only held-out cases; it must not be in train/"


def test_shared_files_are_truncated_below_every_held_out_index(plan):
    payload, _ = plan
    split = json.loads((DATA / "split.json").read_text())
    held = {}
    for c in split["cases"]:
        if "file" in c:
            held.setdefault((c["source"], c["file"]), set()).add(int(c["trajectory"]))
    truncated = [k for k, v in payload.items() if v[1] is not None]
    assert truncated, "the ten shared corpus files should be marked for truncation"
    for key in truncated:
        keep = payload[key][1]
        assert key in held
        assert keep <= min(held[key]), f"{key}: keeping {keep} would include held-out trajectory {min(held[key])}"


def test_nothing_pretrained_is_ever_packed(plan):
    payload, _ = plan
    for source, path in payload:
        assert source in build_release.TASK_SOURCES
        assert "pretrained" not in path


def test_unlicensed_baseline_families_are_not_sources():
    """UNetConvNext and FNO weights carry `license: None` upstream; they must never be republished."""
    lock = json.loads((DATA / "sources.lock.json").read_text())
    ids = {s["id"] for s in lock["sources"]}
    for forbidden in ("UNetConvNext", "FNO", "Poseidon"):
        assert not any(forbidden.lower() in i.lower() for i in ids)
    for source in build_release.TASK_SOURCES:
        assert source in ids


def test_truncation_keeps_a_prefix_and_recomputes_statistics(tmp_path):
    src, dst = tmp_path / "src.hdf5", tmp_path / "dst.hdf5"
    rng = np.random.default_rng(0)
    arrays = [rng.normal(size=(3, 2, 4, 4)).astype("float32") for _ in range(6)]
    write_corpus_file(src, arrays)
    assert build_release.write_truncated(src, dst, 4) == 6
    assert_readable(dst, range(4))
    assert build_release.validate_member(dst)["n_trajectories"] == 4

    with h5py.File(dst, "r") as f:
        names = sorted(f["sims"], key=lambda n: int(n[3:]))
        assert names == [f"sim{i}" for i in range(4)], "kept trajectories must be a contiguous prefix"
        for i in range(4):
            assert np.array_equal(f["sims"][f"sim{i}"][()], arrays[i])
        kept = np.stack(arrays[:4])                       # (N, T, C, X, Y)
        want = kept.transpose(2, 0, 1, 3, 4).reshape(2, -1)  # channel first: the axis norm_* indexes
        assert np.allclose(f["norm_fields_sca_mean"][()].ravel(), want.mean(axis=1), atol=1e-6)
        assert np.allclose(f["norm_fields_sca_std"][()].ravel(), want.std(axis=1), atol=1e-6)
        assert "public_release" in " ".join(f.attrs)


def test_truncation_refuses_to_keep_more_than_exists(tmp_path):
    src, dst = tmp_path / "s.hdf5", tmp_path / "d.hdf5"
    with h5py.File(src, "w") as f:
        g = f.create_group("sims")
        for i in range(2):
            g.create_dataset(f"sim{i}", data=np.zeros((2, 1, 2, 2), dtype="float32"))
    with pytest.raises(SystemExit):
        build_release.write_truncated(src, dst, 5)


def test_download_source_pin_matches_its_manifest():
    """If a release has been published, the checked-in pin must name it exactly."""
    pin = DATA / "download-source.json"
    if not pin.exists():
        pytest.skip("no release published yet")
    d = json.loads(pin.read_text())
    assert set(d) >= {"repo_id", "revision", "manifest_sha256"}
    assert len(d["revision"]) == 40 and len(d["manifest_sha256"]) == 64


def test_leakage_checker_is_executable():
    out = subprocess.run([sys.executable, str(ROOT / "docs" / "validation" / "check_release_leakage.py"), "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "--release" in out.stdout


def test_generator_seeds_for_held_out_cases_are_never_released(plan):
    """PDEArena ships `pde_test_seed_*.yaml`: the seeds that regenerate the held-out trajectories.

    instruction.md forbids reconstructing held-out cases by re-running the benchmark's own
    generator with its seeds, so those files must not appear in a public release. The first
    build of this release included them, because the rule at the time was "ship every metadata
    file"; it only failed because they were absent from the local materialization.
    """
    payload, _ = plan
    leaked = [f"{s}/{p}" for (s, p), v in payload.items()
              if v[0] == "train" and "test" in p.rsplit("/", 1)[-1].lower()]
    assert not leaked, f"held-out metadata would reach an agent: {leaked[:5]}"
    assert all(payload[k][0] == "eval" for k in payload if "pde_test_seed" in k[1])


def test_held_out_metadata_detector():
    assert build_release.is_held_out_metadata("pde_test_seed_126188.yaml")
    assert build_release.is_held_out_metadata("data/test/foo.hdf5")
    assert not build_release.is_held_out_metadata("burgers.json")
    assert not build_release.is_held_out_metadata("pde_train_seed_1.yaml")
    assert not build_release.is_held_out_metadata("README.md")


def test_the_two_subtrees_reconstruct_the_original_and_neither_leaks(tmp_path):
    """train/ + eval/ == the upstream file; train/ alone must not contain a held-out trajectory."""
    import h5py
    import numpy as np

    src = tmp_path / "src.hdf5"
    rng = np.random.default_rng(7)
    arrays = [rng.normal(size=(2, 2, 3, 3)).astype("float32") for _ in range(10)]
    write_corpus_file(src, arrays)

    head, tail = tmp_path / "train.hdf5", tmp_path / "eval.hdf5"
    build_release.write_truncated(src, head, 7)
    assert build_release.write_tail(src, tail, 7) == 3
    assert_readable(head, range(7))
    assert_readable(tail, range(7, 10))

    with h5py.File(head) as fh, h5py.File(tail) as ft:
        kept = sorted(int(n[3:]) for n in fh["sims"])
        held = sorted(int(n[3:]) for n in ft["sims"])
        assert kept == list(range(7)), "train/ must be a contiguous prefix"
        assert held == [7, 8, 9], "eval/ must keep the original trajectory numbering"
        assert set(kept) & set(held) == set(), "the subtrees must not overlap"
        assert sorted(kept + held) == list(range(10)), "together they must reconstruct the original"
        for i in kept:
            assert np.array_equal(fh["sims"][f"sim{i}"][()], arrays[i])
        for i in held:
            assert np.array_equal(ft["sims"][f"sim{i}"][()], arrays[i])
        # the decisive property: no held-out array appears anywhere in train/
        train_bytes = {fh["sims"][n][()].tobytes() for n in fh["sims"]}
        assert not any(arrays[i].tobytes() in train_bytes for i in held)


def test_the_consumer_layout_matches_a_direct_materialization(tmp_path, monkeypatch):
    """A release member is stored as <subtree>/<source>/<path> but must land at <source>/<path>.

    Every reader in this task assumes the materialization layout. Writing the subtree prefix
    through would produce a tree nothing can read, and would make train/ and eval/ collide on
    the ten files the split cuts in two.
    """
    import h5py
    import numpy as np
    sys.path.insert(0, str(DATA))
    import materialize  # noqa: E402

    rng = np.random.default_rng(3)
    arrays = [rng.normal(size=(2, 2, 2, 2)).astype("float32") for _ in range(6)]
    src = tmp_path / "src.hdf5"
    write_corpus_file(src, arrays, norm_value=99.0)
    head, tail = tmp_path / "head.hdf5", tmp_path / "tail.hdf5"
    build_release.write_truncated(src, head, 4)      # the head's norm_* are recomputed over its 4 kept trajectories
    build_release.write_tail(src, tail, 4)           # the tail carries upstream's (99.0) through unchanged

    parts = [{"path": "train/src/a.hdf5", "sha256": "x", "size": head.stat().st_size},
             {"path": "eval/src/a.hdf5", "sha256": "y", "size": tail.stat().st_size}]
    monkeypatch.setattr(materialize, "sha256", lambda p: {head: "x", tail: "y"}[Path(p)])
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].hf_hub_download = (
        lambda repo_id, filename, revision=None, repo_type=None: str(head if "train/" in filename else tail))

    target = tmp_path / "out" / "src" / "a.hdf5"
    assert materialize.merge_members(target, parts, {"repo_id": "r", "revision": "v"}, False, False)
    with h5py.File(target) as f:
        got = sorted(int(n[3:]) for n in f["sims"])
        assert got == list(range(6)), "the merge must reconstruct every trajectory"
        for i in range(6):
            assert np.array_equal(f["sims"][f"sim{i}"][()], arrays[i])
        assert f["norm_fields_sca_mean"][()].ravel()[0] == 99.0, \
            "a merged file must carry upstream's norm_*, which only the eval tail preserves"
    assert_readable(target, range(6))
    # an existing file is left alone rather than silently re-merged
    assert not materialize.merge_members(target, parts, {"repo_id": "r", "revision": "v"}, False, False)


def test_merge_refuses_overlapping_members(tmp_path, monkeypatch):
    import h5py
    import numpy as np
    sys.path.insert(0, str(DATA))
    import materialize  # noqa: E402

    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    for path in (a, b):
        with h5py.File(path, "w") as f:
            g = f.create_group("sims")
            g.create_dataset("sim0", data=np.zeros((1, 1, 2, 2), dtype="float32"))
    parts = [{"path": "train/s/x.hdf5", "sha256": "1", "size": 1},
             {"path": "eval/s/x.hdf5", "sha256": "2", "size": 1}]
    monkeypatch.setattr(materialize, "sha256", lambda p: {a: "1", b: "2"}[Path(p)])
    monkeypatch.setitem(sys.modules, "huggingface_hub", type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].hf_hub_download = (
        lambda repo_id, filename, revision=None, repo_type=None: str(a if "train/" in filename else b))
    with pytest.raises(SystemExit):
        materialize.merge_members(tmp_path / "out.hdf5", parts, {"repo_id": "r", "revision": "v"}, False, False)


def test_a_member_without_the_group_attributes_is_refused(tmp_path):
    """What the first release shipped: datasets copied, `sims` attributes dropped, readers broken."""
    sys.path.insert(0, str(DATA))
    import materialize  # noqa: E402

    bad = tmp_path / "bad.hdf5"
    with h5py.File(bad, "w") as f:
        g = f.create_group("sims")
        g.create_dataset("sim0", data=np.zeros((2, 1, 2, 2), dtype="float32"))
    with pytest.raises(SystemExit):
        build_release.validate_member(bad)
    with pytest.raises(SystemExit):
        materialize.check_corpus_readable(bad)


def test_from_release_lands_members_at_the_direct_layout(tmp_path, monkeypatch):
    """`--from-release` must write <dest>/<source>/<path>, which is what make_cases.py and the
    starter read; the first implementation wrote <dest>/hf/<source>/<path>, a tree nothing reads."""
    import hashlib
    sys.path.insert(0, str(DATA))
    import materialize  # noqa: E402

    rng = np.random.default_rng(5)
    arrays = [rng.normal(size=(2, 2, 2, 2)).astype("float32") for _ in range(5)]
    src = tmp_path / "upstream.hdf5"
    write_corpus_file(src, arrays)
    hub = tmp_path / "hub"
    (hub / "train" / "ape2d-full").mkdir(parents=True); (hub / "eval" / "ape2d-full").mkdir(parents=True)
    build_release.write_truncated(src, hub / "train" / "ape2d-full" / "x.hdf5", 3)
    build_release.write_tail(src, hub / "eval" / "ape2d-full" / "x.hdf5", 3)
    (hub / "train" / "ape2d-full" / "x.json").write_text("{}")

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    records = [{"path": rel, "subtree": rel.split("/")[0], "size": (hub / rel).stat().st_size, "sha256": digest(hub / rel)}
               for rel in ("train/ape2d-full/x.hdf5", "eval/ape2d-full/x.hdf5", "train/ape2d-full/x.json")]
    (hub / "payload-files.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    manifest = {"release_id": "test", "inventory": {"path": "payload-files.jsonl", "sha256": digest(hub / "payload-files.jsonl")}}
    (hub / "release-manifest.json").write_text(json.dumps(manifest))
    pin = tmp_path / "download-source.json"
    pin.write_text(json.dumps({"repo_id": "r", "revision": "v", "manifest_sha256": digest(hub / "release-manifest.json")}))
    monkeypatch.setattr(materialize, "RELEASE_SOURCE", pin)
    monkeypatch.setitem(sys.modules, "huggingface_hub", type(sys)("huggingface_hub"))
    sys.modules["huggingface_hub"].hf_hub_download = (
        lambda repo_id, filename, revision=None, repo_type=None: str(hub / filename))

    dest = tmp_path / "dest"
    materialize.from_release(dest, False, False, "train")
    assert (dest / "ape2d-full" / "x.hdf5").exists() and (dest / "ape2d-full" / "x.json").exists()
    assert not (dest / "hf").exists() and not (dest / "train").exists()
    assert_readable(dest / "ape2d-full" / "x.hdf5", range(3))
    assert json.loads((dest / "release-receipt.json").read_text())["subtree"] == "train"

    full = tmp_path / "dest_full"
    materialize.from_release(full, False, False, "")
    assert_readable(full / "ape2d-full" / "x.hdf5", range(5))      # head + tail merged back into the upstream file


def test_release_head_is_recognised_and_only_a_pools_host_accepts_it(tmp_path):
    """An agent host holds the truncated heads from the release; `materialize.py --pools` must accept them
    instead of reporting a size mismatch against the upstream file, and a verifier host must not."""
    sys.path.insert(0, str(DATA))
    import materialize  # noqa: E402

    rng = np.random.default_rng(11)
    arrays = [rng.normal(size=(2, 2, 2, 2)).astype("float32") for _ in range(5)]
    src, head = tmp_path / "src.hdf5", tmp_path / "head.hdf5"
    write_corpus_file(src, arrays)
    build_release.write_truncated(src, head, 3)
    assert materialize.release_head(src) is None
    assert materialize.release_head(head).startswith("training-only")
