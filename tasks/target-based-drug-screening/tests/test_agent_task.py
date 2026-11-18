from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys

import lmdb
import numpy as np
import pytest
from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC

RANDOM_FIXTURE_BEDROC = .1 * (1 - math.exp(-80.5)) / (1 - math.exp(-8.05))

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment"))
sys.path.insert(0, str(TASK / "environment/evaluation"))
from input_view import BENCHMARK_TARGETS, FIELDS, load_permutation, redact_lmdb, stage_inputs
from publish import publish
from score import evaluate, ranking_metrics, score_target

spec = importlib.util.spec_from_file_location("screening_grader", TASK / "tests/grade.py")
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
spec = importlib.util.spec_from_file_location("screening_stage", TASK / "environment/data/stage_task_data.py")
stager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stager)


def write_lmdb(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(path), subdir=False, map_size=8 << 20)
    with db.begin(write=True) as txn:
        for index, record in enumerate(records):
            txn.put(str(index).encode(), pickle.dumps(record))
    db.close()


def molecules(count=200):
    for index in range(count):
        yield {"atoms": ["C", "O"] if index % 2 else ["C", "N"],
               "coordinates": [np.array([[0., 0., 0.], [1., 0., 0.]], dtype=np.float32)],
               "smi": "CO" if index % 2 else "CN", "label": int(index < 20),
               "source_path": "/hidden/active-list", "compound_id": f"answer-{index}"}


def add_target(root, benchmark, target):
    directory = root / "test" / benchmark / target
    write_lmdb(directory / "mols.lmdb", molecules())
    pocket = "pocket.lmdb" if benchmark == "dude" else "pockets.lmdb"
    write_lmdb(directory / pocket, [
        {"pocket_atoms": ["C", "N"], "pocket_coordinates": np.zeros((2, 3)),
         "label": 1, "pocket": "secret-pocket"}])


def predictions(root, targets, *, score="0"):
    path = root / "scores.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("benchmark", "target_id", "ligand_index", "score"))
        for benchmark, target in targets:
            for index in range(200):
                writer.writerow((benchmark, target, index, score))
    return path


def fake_package(path, script):
    path.mkdir()
    (path / "weights.bin").write_bytes(b"synthetic-test-weights")
    (path / "config.json").write_text("{}")
    (path / "model_manifest.json").write_text(json.dumps({
        "method": "test-fixture", "parameter_count": 1, "model_config": "config.json",
        "checkpoint": {"path": "weights.bin",
                       "sha256": hashlib.sha256(b"synthetic-test-weights").hexdigest()}}))
    (path / "infer.sh").write_text(script)
    (path / "README.md").write_text("Synthetic verifier fixture, no scientific claim.")


def test_redaction_preserves_structure_and_removes_all_metadata(tmp_path):
    source = tmp_path / "source.lmdb"
    write_lmdb(source, molecules())
    target = tmp_path / "clean.lmdb"
    assert redact_lmdb(source, target, "molecule") == 200
    db = lmdb.open(str(target), subdir=False, readonly=True, lock=False)
    with db.begin() as txn:
        record = pickle.loads(txn.get(b"0"))
        assert set(record) == set(FIELDS["molecule"])
        assert record["atoms"] == ["C", "N"]
        np.testing.assert_array_equal(record["coordinates"][0], next(molecules())["coordinates"][0])
        assert txn.get(b"199") is not None
    db.close()


def test_exact_ties_average_all_permutations_and_ignore_original_order():
    values = [[1., 1], [1., 0], [0., 1], [0., 0]]
    result = ranking_metrics(values)
    permutations = [
        list(a) + list(b)
        for a in itertools.permutations(values[:2])
        for b in itertools.permutations(values[2:])]
    assert result["roc_auc"] == pytest.approx(np.mean([CalcAUC(v, 1) for v in permutations]))
    assert result["bedroc_80.5"] == pytest.approx(np.mean([CalcBEDROC(v, 1, 80.5) for v in permutations]))
    constant = [[0., int(index < 20)] for index in range(200)]
    random_metrics = ranking_metrics(constant)
    assert random_metrics["roc_auc"] == pytest.approx(0.5)
    assert random_metrics["ef_0.01"] == pytest.approx(1.)
    assert ranking_metrics(list(reversed(constant))) == pytest.approx(random_metrics)


@pytest.fixture
def small_suite(tmp_path):
    root = tmp_path / "benchmark"
    targets = [("dude", "target-a"), ("lit_pcba", "target-b")]
    for benchmark, target in targets:
        add_target(root, benchmark, target)
    return root, targets


def test_complete_predictions_and_coverage_rejections(tmp_path, small_suite):
    root, targets = small_suite
    path = predictions(tmp_path, targets)
    result = evaluate(root, path)
    assert result["overall_mean_ef_0.01"] == pytest.approx(1.)
    assert len(result["per_target"]) == 2
    for selected, message in [(targets[:1], "missing targets"), (targets + targets[:1], "repeated target")]:
        with pytest.raises(ValueError, match=message):
            evaluate(root, predictions(tmp_path, selected))
    with pytest.raises(ValueError, match="invalid label or score"):
        evaluate(root, predictions(tmp_path, targets, score="nan"))
    with pytest.raises(ValueError, match="unexpected"):
        evaluate(root, predictions(tmp_path, [("dude", "../escape")]))
    path = predictions(tmp_path, targets)
    text = path.read_text().replace("target-a,1,0", "target-a,0,0")
    path.write_text(text)
    with pytest.raises(ValueError, match="indices"):
        evaluate(root, path)


def test_atomic_publication_and_invalid_package_retains_current(tmp_path):
    source, live = tmp_path / "candidate", tmp_path / "submission"
    fake_package(source, "exit 0\n")
    live.mkdir()
    publish(source, live)
    (source / "README.md").write_text("replacement")
    publish(source, live)
    assert (live / "README.md").read_text() == "replacement"
    (source / "weights.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        publish(source, live)
    assert (live / "README.md").read_text() == "replacement"
    assert not list(tmp_path.glob(".publish-*"))


def test_overlay_first_publish_preserves_populated_destination(tmp_path, monkeypatch):
    class OverlayLibc:
        def renameat2(*args):
            ctypes.set_errno(errno.EXDEV)
            return -1

    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: OverlayLibc)
    source, live = tmp_path / "candidate", tmp_path / "submission"
    fake_package(source, "exit 0\n")
    live.mkdir()
    publish(source, live)
    previous = (live / "README.md").read_bytes()
    (source / "README.md").write_text("second package")
    with pytest.raises(OSError):
        publish(source, live)
    assert (live / "README.md").read_bytes() == previous
    assert not list(tmp_path.glob(".publish-*"))


@pytest.mark.skipif(os.geteuid() != 0, reason="tests actual verifier privilege drop")
@pytest.mark.parametrize("behavior", ["complete", "exit-error", "timeout"])
def test_full_suite_unprivileged_replay(tmp_path, behavior, monkeypatch):
    monkeypatch.setenv("SSI_TEST_SECRET", "must-not-reach-inference")
    probe = tmp_path / "uid-probe"
    probe.touch()
    try:
        os.chown(probe, grader.INFERENCE_UID, grader.INFERENCE_UID)
    except OSError:
        pytest.skip("host user namespace does not map inference UID/GID 65534")
    probe.unlink()
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    for benchmark, count in BENCHMARK_TARGETS.items():
        for index in range(count):
            add_target(root, benchmark, f"target-{index:03d}")
    script = "set -eu\n"
    script += 'test "$(id -u)" = 65534\n'
    script += f'test ! -r "{root}/test/dude/target-000/mols.lmdb"\n'
    script += """python - "$1" "$2" <<'PYPROBE'
import os
from pathlib import Path
import sys
assert os.getuid() == os.getgid() == 65534
assert os.getgroups() == []
assert 'SSI_TEST_SECRET' not in os.environ
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert int(status['CapEff'].strip(), 16) == 0
inputs = Path(sys.argv[1])
for path in (inputs / 'test/dude/target-000/mols.lmdb', Path.cwd() / 'weights.bin'):
    try:
        with path.open('ab'):
            pass
    except PermissionError:
        pass
    else:
        raise AssertionError(f'writable trusted file: {path}')
    try:
        path.chmod(0o666)
    except PermissionError:
        pass
    else:
        raise AssertionError(f'can change trusted permissions: {path}')
if Path('/tests').exists():
    assert not os.access('/tests/grade.py', os.R_OK)
for parent in (inputs, Path.cwd()):
    try:
        (parent / 'injected').touch()
    except PermissionError:
        pass
    else:
        raise AssertionError(f'writable trusted directory: {parent}')
(Path(sys.argv[2]) / 'writable-probe').touch()
Path(os.environ['HOME'], 'writable-probe').touch()
PYPROBE
"""
    script += f'test ! -r "{tmp_path}/logs/private-order/manifest.json"\n'
    script += f'test ! -w "{tmp_path}/logs"\n'
    if behavior == "exit-error":
        script += "exit 7\n"
    elif behavior == "timeout":
        script += "sleep 10\n"
    else:
        script += """printf 'benchmark,target_id,ligand_index,score\\n' > "$2/scores.csv"
for benchmark in dude lit_pcba; do
  for target in "$1/test/$benchmark/"*; do
    name=${target##*/}
    for index in $(seq 0 199); do
      printf '%s,%s,%s,0\\n' "$benchmark" "$name" "$index"
    done
  done
done >> "$2/scores.csv"
"""
    package = tmp_path / "candidate"
    fake_package(package, script)
    result = grader.grade(package, root, tmp_path / "logs", timeout=0.1 if behavior == "timeout" else 30)
    if behavior == "complete":
        assert result["valid"]
        assert result["score"] == pytest.approx(RANDOM_FIXTURE_BEDROC)
        assert result["reward"] == pytest.approx(RANDOM_FIXTURE_BEDROC)
        assert len(result["per_target"]) == 117
        assert result["swapped_pocket_counterfactual"]["status"] == "ok"
    else:
        assert not result["valid"]
        assert result["score"] == 0
        assert result["reward"] == 0


@pytest.mark.skipif(os.geteuid() != 0, reason="requires verifier root preflight")
def test_infrastructure_failure_removes_stale_reward(tmp_path):
    root, output = tmp_path / "private", tmp_path / "logs"
    root.mkdir(mode=0o700)
    output.mkdir()
    (output / "reward.json").write_text('{"reward":1}')
    with pytest.raises(ValueError, match="expected 102 targets"):
        grader.grade(tmp_path / "missing-package", root, output)
    assert not (output / "reward.json").exists()


@pytest.mark.parametrize("behavior", ["complete", "exit-error", "timeout"])
def test_grade_control_flow_with_privilege_hooks_mocked(tmp_path, monkeypatch, behavior):
    """Exercise the complete orchestration even in a single-UID host namespace."""
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    for benchmark, count in BENCHMARK_TARGETS.items():
        for index in range(count):
            add_target(root, benchmark, f"target-{index:03d}")
    monkeypatch.setattr(grader, "demote", lambda: None)
    monkeypatch.setattr(grader.os, "chown", lambda *args: None)
    if behavior == "exit-error":
        script = "exit 7\n"
    elif behavior == "timeout":
        script = "sleep 10\n"
    else:
        script = """mkdir -p "$HOME/cache"
test -z "$(ls -A "$2")"
printf 'benchmark,target_id,ligand_index,score\\n' > "$2/scores.csv"
for benchmark in dude lit_pcba; do
  for target in "$1/test/$benchmark/"*; do
    name=${target##*/}
    for index in $(seq 0 199); do
      printf '%s,%s,%s,0\\n' "$benchmark" "$name" "$index"
    done
  done
done >> "$2/scores.csv"
"""
    package = tmp_path / "candidate"
    fake_package(package, script)
    output = tmp_path / "logs"
    result = grader.grade(package, root, output, timeout=0.1 if behavior == "timeout" else 30)
    assert result["valid"] == (behavior == "complete")
    assert result["score"] == pytest.approx(RANDOM_FIXTURE_BEDROC if behavior == "complete" else 0)
    assert result["reward"] == pytest.approx(RANDOM_FIXTURE_BEDROC if behavior == "complete" else 0)
    scores = json.loads((output / "reward.json").read_text())
    if result["valid"]:
        assert scores["overall_mean_ef_0.01"] == pytest.approx(1)
        assert scores["overall_mean_roc_auc"] == pytest.approx(0.5)
        assert scores["dataset_balanced_bedroc"] == result["reward"]
        assert scores["score"] == result["score"]
        assert result["scoring_protocol"] == "drugclip-dataset-bedroc/v2"
        assert result["swapped_pocket_counterfactual"]["status"] == "ok"
        assert result["swapped_pocket_counterfactual"]["pocket_specificity_delta"] == pytest.approx(0)
        assert len(scores) == 9
    else:
        assert scores == {"score": 0, "reward": 0}
    assert scores["score"] == result["score"]
    assert scores["reward"] == result["reward"]
    assert json.loads((output / "report.json").read_text()) == result
    assert json.loads((output / "details.json").read_text()) == result


def test_native_starter_training_export_and_label_free_replay(tmp_path, small_suite):
    pytest.importorskip("torch")
    training = tmp_path / "training"
    paired = []
    for index, molecule in enumerate(molecules(8)):
        paired.append({**molecule, "pocket_atoms": molecule["atoms"],
                       "pocket_coordinates": np.zeros((2, 3))})
    for split in ("profsa_pretrain_500k", "biolip_train", "biolip_dev"):
        write_lmdb(training / "processed" / split / "data.lmdb", paired)
    source = TASK / "environment/starter/train.py"
    package = tmp_path / "model"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run([sys.executable, str(source), "--input-root", str(training),
                    "--output-dir", str(package), "--epochs", "1", "--batch-size", "2",
                    "--workers", "0", "--device", "cpu"], check=True, env=env, capture_output=True)
    root, _ = small_suite
    inputs, output = tmp_path / "inputs", tmp_path / "predictions"
    mapping = tmp_path / "private-order"
    stage_inputs(root, inputs, mapping, full_suite=False)
    subprocess.run(["bash", str(package / "infer.sh"), str(inputs), str(output),
                    "--workers", "0", "--device", "cpu"],
                   check=True, env=env, capture_output=True)
    result = evaluate(root, output / "scores.csv", mapping)
    assert np.isfinite(result["overall_mean_ef_0.01"])
    assert len(result["per_target"]) == 2
    live = tmp_path / "submission"
    publish(package, live)
    assert (live / "model_manifest.json").is_file()
    assert not (live / "pretrained").exists()


def test_private_shuffle_maps_every_record_and_preserves_correct_scores(tmp_path, small_suite):
    root, targets = small_suite
    # Distinguishable allowed features prove the mapping matches copied bytes.
    for benchmark, target in targets:
        records = [{**record, "smi": f"fixture-{index}"}
                   for index, record in enumerate(molecules())]
        write_lmdb(root / "test" / benchmark / target / "mols.lmdb", records)
    inputs, mapping = tmp_path / "inputs", tmp_path / "private-order"
    stage_inputs(root, inputs, mapping, full_suite=False, seed=b"x" * 32)
    assert mapping.stat().st_mode & 0o077 == 0
    assert not any(path.suffix in (".npy", ".json") for path in inputs.rglob("*"))
    orders = []
    with (tmp_path / "scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("benchmark", "target_id", "ligand_index", "score"))
        for benchmark, target in targets:
            order = load_permutation(mapping, benchmark, target, 200)
            orders.append(order)
            assert not np.array_equal(order, np.arange(200))
            db = lmdb.open(str(inputs / "test" / benchmark / target / "mols.lmdb"),
                           subdir=False, readonly=True, lock=False)
            with db.begin() as transaction:
                for index, original in enumerate(order):
                    record = pickle.loads(transaction.get(str(index).encode()))
                    assert set(record) == set(FIELDS["molecule"])
                    assert record["smi"] == f"fixture-{original}"
                    writer.writerow((benchmark, target, index, int(original < 20)))
            db.close()
    assert not np.array_equal(*orders)
    result = evaluate(root, tmp_path / "scores.csv", mapping)
    assert result["overall_mean_ef_0.01"] == pytest.approx(10)
    assert all(row["roc_auc"] == pytest.approx(1) for row in result["per_target"])
    # Reusing a controller-private seed reproduces byte-identical input/order.
    stage_inputs(root, tmp_path / "inputs2", tmp_path / "maps2",
                 full_suite=False, seed=b"x" * 32)
    assert (mapping / "manifest.json").read_bytes() == (tmp_path / "maps2/manifest.json").read_bytes()
    for benchmark, target in targets:
        assert (inputs / "test" / benchmark / target / "mols.lmdb").read_bytes() == \
            (tmp_path / "inputs2/test" / benchmark / target / "mols.lmdb").read_bytes()
    # A changed seed gives a fresh, independently shuffled view.
    stage_inputs(root, tmp_path / "inputs3", tmp_path / "maps3",
                 full_suite=False, seed=b"y" * 32)
    assert not np.array_equal(orders[0], load_permutation(tmp_path / "maps3", *targets[0], 200))


def test_index_only_predictor_loses_actives_first_advantage(tmp_path, small_suite):
    root, targets = small_suite
    benchmark, target = targets[0]
    rows = [{"benchmark": benchmark, "target_id": target,
             "ligand_index": str(index), "score": str(-index)} for index in range(200)]
    assert score_target(rows, root)["ef_0.01"] == pytest.approx(10)
    auc, ef = [], []
    # Fixed independent private seeds make this statistical regression stable.
    for trial in range(32):
        mapping = tmp_path / f"map-{trial}"
        stage_inputs(root, tmp_path / f"input-{trial}", mapping, full_suite=False,
                     seed=hashlib.sha256(f"private-fixture-{trial}".encode()).digest())
        result = score_target(rows, root, mapping)
        auc.append(result["roc_auc"])
        ef.append(result["ef_0.01"])
    assert np.mean(auc) == pytest.approx(0.5, abs=0.03)
    assert np.mean(ef) == pytest.approx(1, abs=0.6)
    constant = [{**row, "score": "0"} for row in rows]
    assert score_target(constant, root, mapping)["ef_0.01"] == pytest.approx(1)


def test_mapping_corruption_is_infrastructure_failure(tmp_path, small_suite):
    root, targets = small_suite
    mapping = tmp_path / "maps"
    with pytest.raises(ValueError, match="separate"):
        stage_inputs(root, tmp_path / "inputs", tmp_path / "inputs/maps", full_suite=False)
    stage_inputs(root, tmp_path / "inputs", mapping, full_suite=False)
    benchmark, target = targets[0]
    path = mapping / benchmark / f"{target}.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        load_permutation(mapping, benchmark, target, 200)
    np.save(path, np.zeros(200, dtype=np.int64))
    manifest_path = mapping / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["targets"][f"{benchmark}/{target}"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="complete permutation"):
        load_permutation(mapping, benchmark, target, 200)


def test_controller_views_allowlist_files_and_preserve_source(tmp_path, monkeypatch):
    data, pretrained = tmp_path / "data", tmp_path / "pretrained"
    data.mkdir()
    pretrained.mkdir()
    (data / ".complete.json").write_text("{}")
    (data / "VALIDATION.json").write_text('{"status":"PASS","smoke":false}')
    for name in stager.DEVELOPMENT_FILES:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"training-view-fixture")
    for benchmark, count in BENCHMARK_TARGETS.items():
        for index in range(count):
            add_target(data, benchmark, f"target-{index:03d}")
    (data / "test/answer-table.csv").write_text("never copy into development")
    (pretrained / "dude_identity_90.pt").write_text("forbidden benchmark weights")
    assets = {}
    for key, name in (("molecule", "mol_pre_no_h_220816.pt"),
                      ("molecule_dictionary", "dict_mol.txt"),
                      ("pocket_dictionary", "dict_pkt.txt")):
        value = name.encode()
        (pretrained / name).write_bytes(value)
        assets[key] = {"path": name, "size": len(value),
                       "sha256": hashlib.sha256(value).hexdigest()}
    # Use a test-owned pinned template with tiny fixture assets; production
    # always reads the committed official template.
    recipe = tmp_path / "recipe"
    template = recipe / "baseline/drugclip/pretrained_manifest.example.json"
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps(assets))
    monkeypatch.setattr(stager, "HERE", recipe / "data")
    (pretrained / "manifest.json").write_text(json.dumps(assets))
    dev, private = tmp_path / "development", tmp_path / "private"
    stager.stage(data, pretrained, dev, private)
    assert not (dev / "test").exists()
    assert not (dev / "pretrained/dude_identity_90.pt").exists()
    assert (private / "test/dude/target-000/mols.lmdb").is_file()
    assert not (private / "test/answer-table.csv").exists()
    dev_receipt = json.loads((dev / ".complete.json").read_text())
    private_receipt = json.loads((private / ".complete.json").read_text())
    assert dev_receipt["view"] == "development"
    assert private_receipt["view"] == "private"
    dev_files = {item["path"]: item for item in dev_receipt["files"]}
    assert not any(path.startswith("test/") for path in dev_files)
    assert len(dev_files) == len(stager.DEVELOPMENT_FILES) + 4
    for root, receipt in ((dev, dev_receipt), (private, private_receipt)):
        encoded = json.dumps(receipt["files"], sort_keys=True, separators=(",", ":")).encode()
        assert receipt["file_manifest_sha256"] == hashlib.sha256(encoded).hexdigest()
        for item in receipt["files"]:
            path = root / item["path"]
            assert item["size"] == path.stat().st_size
            assert item["sha256"] == stager.sha256(path)
    copied = dev / stager.DEVELOPMENT_FILES[0]
    original = data / stager.DEVELOPMENT_FILES[0]
    assert copied.stat().st_ino != original.stat().st_ino
    copied.write_bytes(b"modified copy")
    assert original.read_bytes() == b"training-view-fixture"
    with pytest.raises(ValueError, match="new, separate"):
        stager.stage(data, pretrained, dev, private)
    (data / "VALIDATION.json").write_text('{"status":"PASS","smoke":true}')
    with pytest.raises(ValueError, match="validation"):
        stager.stage(data, pretrained, tmp_path / "dev2", tmp_path / "private2")


def test_runtime_report_records_environment_without_dumping_credentials(tmp_path):
    report_path = tmp_path / "runtime.json"
    subprocess.run([sys.executable, str(TASK / "environment/record_environment.py"),
                    "--role", "controller", "--code-commit", "a" * 40,
                    "--image-id", "sha256:" + "b" * 64,
                    "--output", str(report_path)], check=True, capture_output=True)
    report = json.loads(report_path.read_text())
    assert report["role"] == "controller"
    assert report["python"]["version"]
    assert set(report["packages"]) == {"torch", "numpy", "rdkit", "lmdb", "harbor", "pytest"}
    assert report["controller_identifiers"]["code_commit"] == "a" * 40
    assert "environment_variables" not in report
    assert not report_path.with_name(report_path.name + ".tmp").exists()
