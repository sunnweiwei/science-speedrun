import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/data"))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "environment/baseline/uma-s-pretrain")
)
from prepare_training import prepare, _AppendOnlyIDs, fit_element_references


def test_training_schedule_rejects_ambiguous_duration(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        prepare(tmp_path, tmp_path / "output", steps=500, epochs=1)
    with pytest.raises(ValueError, match="exactly one"):
        prepare(tmp_path, tmp_path / "output", steps=None, epochs=None)


def test_bulk_lmdb_id_index_preserves_order_labels_and_metadata(tmp_path):
    pytest.importorskip("ase_db_backends")
    from ase.db import connect
    from ase.calculators.singlepoint import SinglePointCalculator

    path = tmp_path / "frames.aselmdb"
    with connect(path) as db:
        db.ids = _AppendOnlyIDs(db.ids)
        for i in range(40):
            atoms = bulk("Cu").repeat((1 + i % 3, 1, 1))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(i),
                forces=np.full((len(atoms), 3), i / 10),
                stress=np.eye(3) * i / 100,
            )
            assert db.write(atoms, data={"sid": f"frame-{i}"}) == i + 1
        assert list(db.ids) == list(range(1, 41))
    with connect(path, readonly=True) as db:
        assert len(db) == 40
        for i, row in enumerate(db.select()):
            assert row.id == i + 1
            assert row.data.sid == f"frame-{i}"
            assert row.energy == float(i)
            assert np.allclose(row.forces, i / 10)
            assert np.allclose(
                row.toatoms().get_cell(), bulk("Cu").repeat((1 + i % 3, 1, 1)).cell
            )
            assert np.allclose(
                row.toatoms().get_stress(voigt=False),
                np.eye(3) * i / 100,
            )


def test_scratch_recipe_uses_only_training_statistics_and_no_checkpoint(tmp_path):
    import json
    import yaml
    from common import digest, encode_atoms, write_rows

    data = tmp_path / "data"
    data.mkdir()
    row = dict(
        id="mptrj/frame",
        parent="mp-1",
        source="mptrj",
        atoms=encode_atoms(bulk("Cu")),
        energy=-1.0,
        forces=[[0.1, 0.2, 0.3]],
        stress=np.eye(3).tolist(),
    )
    for split in ("train", "val"):
        write_rows(data / f"{split}.jsonl", [row])
    (data / "element-references.yaml").write_text("mpa_elem_refs: {29: -1.0}\n")
    (data / "manifest.json").write_text(
        json.dumps(
            dict(
                energy_convention="mp-pbe-raw-total",
                corrections_applied=False,
                files={
                    f"{s}.jsonl": digest(data / f"{s}.jsonl") for s in ("train", "val")
                },
                element_reference_resource={
                    "sha256": digest(data / "element-references.yaml")
                },
            )
        )
    )
    config_path = prepare(
        data,
        tmp_path / "output",
        steps=None,
        epochs=3,
        world_size=8,
        batch_size=4,
        evaluate_every=1000,
    )
    config = yaml.safe_load(config_path.read_text())
    assert config["epochs"] == 3 and config["steps"] is None
    assert config["batch_size"] == 4
    assert config["job"]["scheduler"]["ranks_per_node"] == 8
    assert config["checkpoint_every_n_steps"] == 5000
    assert config["runner"]["train_eval_unit"]["ema_decay"] is None
    assert (
        config["runner"]["callbacks"][-1]["_target_"]
        == "learning_curves.LearningCurveAudit"
    )
    assert any(
        c.get("max_saved_checkpoints") == 12 for c in config["runner"]["callbacks"]
    )
    model = config["runner"]["train_eval_unit"]["model"]
    assert model["_target_"] == "fairchem.core.models.base.HydraModel"
    assert "checkpoint_location" not in model and "finetune_config" not in model
    assert model["backbone"]["num_experts"] == 32
    assert model["backbone"]["direct_forces"] is False
    before = json.loads((config_path.parent / "preparation.json").read_text())
    assert before["initial_checkpoint_sha256"] is None
    assert before["element_reference_values_eV"][29] == pytest.approx(-1.0)
    assert before["force_rms"] == pytest.approx(np.sqrt(0.14 / 3))

    # Neither development labels nor an external fitted-reference file may
    # influence the learned offsets / normalizers in a from-scratch track.
    row.update(energy=-999.0, forces=[[10.0, 20.0, 30.0]])
    write_rows(data / "val.jsonl", [row])
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["files"]["val.jsonl"] = digest(data / "val.jsonl")
    (data / "manifest.json").write_text(json.dumps(manifest))
    (data / "element-references.yaml").unlink()
    path = prepare(data, tmp_path / "changed-dev", steps=2)
    after = json.loads((path.parent / "preparation.json").read_text())
    for key in ["force_rms", "element_reference_values_eV", "statistics_train_sha256"]:
        assert after[key] == before[key]

    from export_model import export

    # Reject a mismatched or warm-start receipt before opening checkpoint weights.
    for key, value in [
        ("initial_checkpoint_sha256", "pretrained-sha"),
        ("pretrained_weights_allowed", True),
        ("training_manifest_sha256", "another-dataset"),
        ("statistics_train_sha256", "another-training-split"),
    ]:
        bad = {**after, key: value}
        receipt = tmp_path / "bad-preparation.json"
        receipt.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="pretraining provenance"):
            export(
                tmp_path / "absent-model",
                data,
                tmp_path / "artifact",
                preparation=receipt,
            )


def test_element_offsets_recover_known_mixture_and_ignore_absent_elements():
    compositions = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.5, 0.5]])
    refs, present, rank = fit_element_references(
        compositions, np.array([-2.0, -4.0, -3.0])
    )
    assert refs == pytest.approx([0.0, -2.0, -4.0])
    assert present == [1, 2] and rank == 2
    refs, _, rank = fit_element_references(compositions[2:], np.array([-3.0]))
    assert refs == pytest.approx([0.0, -3.0, -3.0])
    assert rank == 1


def test_export_package_has_verifier_entrypoint(tmp_path):
    import json
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from common import digest, encode_atoms, write_rows
    from export_model import export

    data = tmp_path / "data"
    data.mkdir()
    row = dict(
        id="mptrj/one",
        parent="mp-1",
        source="mptrj",
        atoms=encode_atoms(bulk("Cu")),
        energy=-1.0,
        forces=[[0.1, 0.2, 0.3]],
        stress=np.eye(3).tolist(),
    )
    for split in ("train", "val"):
        write_rows(data / f"{split}.jsonl", [row])
    (data / "manifest.json").write_text(
        json.dumps(
            dict(
                energy_convention="mp-pbe-raw-total",
                corrections_applied=False,
                files={
                    f"{s}.jsonl": digest(data / f"{s}.jsonl") for s in ("train", "val")
                },
            )
        )
    )
    config = prepare(data, tmp_path / "prepared", steps=2)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        SimpleNamespace(model_state_dict={"weight": torch.tensor([1.0])}), checkpoint
    )
    package = export(
        checkpoint,
        data,
        tmp_path / "package",
        preparation=config.parent / "preparation.json",
    )
    for filename in [
        "calculator.py",
        "README.md",
        "model.pt",
        "preparation.json",
        "training-manifest.json",
    ]:
        assert (package / filename).is_file()
    exported = torch.load(package / "model.pt", weights_only=False)
    assert exported.ema_state_dict["module.weight"].item() == 1.0
    assert exported.ema_state_dict["n_averaged"].item() == 1
