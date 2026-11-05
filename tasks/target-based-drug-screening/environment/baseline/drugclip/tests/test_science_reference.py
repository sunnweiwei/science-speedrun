from pathlib import Path
import hashlib
import json
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from rdkit.ML.Scoring.Scoring import CalcBEDROC

STARTER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STARTER))

from data import (UNIMOL_VOCABULARY, UNIMOL_POCKET_VOCABULARY,  # noqa: E402
                  structure_features, collate_pairs, pocket_crop_indices,
                  SingleStructureDataset)
from model import PaperConfig, PaperDrugCLIP, parameter_count  # noqa: E402
from predict import author_pool_scores  # noqa: E402
from train import (pretrained_identity, make_validation_loader, stop_decision,
                   broadcast_stop, require_compatible_profsa_crop)  # noqa: E402
from training import PolynomialSchedule  # noqa: E402
from validation import (aggregate_metrics, masked_logits,  # noqa: E402
                        retrieval_batch_metrics, selection_improves)


def test_paper_architecture_and_dictionaries():
    model = PaperDrugCLIP(PaperConfig())
    assert len(model.molecule_encoder.encoder.layers) == 15
    assert len(model.pocket_encoder.encoder.layers) == 15
    assert model.molecule_encoder.vocab_size == len(UNIMOL_VOCABULARY) == 31
    assert model.pocket_encoder.vocab_size == len(UNIMOL_POCKET_VOCABULARY) == 10
    assert model.pocket_encoder.encoder.layers[0].fc1.out_features == 2048
    assert model.pocket_encoder.encoder.layers[0].self_attn.in_proj.in_features == 512
    assert parameter_count(model) == 95_305_418
    assert not any(name == "logit_scale" for name, _ in model.named_parameters())
    assert model.molecule_projection.layers[0].out_features == 512
    assert model.pocket_projection.layers[-1].out_features == 128


def test_official_token_order_and_pair_edges():
    atoms = ["C", "N", "H"]
    xyz = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    mol = structure_features(atoms, xyz, max_atoms=8, seed=1, unimol=True)
    pkt = structure_features(atoms, xyz, max_atoms=8, seed=1, unimol_pocket=True)
    assert mol["tokens"].tolist() == [1, 4, 5, 2]
    assert pkt["tokens"].tolist() == [1, 4, 5, 2]
    assert mol["edge_types"][1, 2].item() == 4 * 31 + 5
    assert pkt["edge_types"][1, 2].item() == 4 * 10 + 5
    assert UNIMOL_POCKET_VOCABULARY[-1] == "[MASK]"


def test_author_benchmark_pocket_atom_names():
    atoms = ["CA", "CB", "N.am", "O.2", "H"]
    xyz = np.arange(15, dtype=np.float32).reshape(5, 3)
    pocket = structure_features(atoms, xyz, max_atoms=8, seed=1,
                                unimol_pocket=True)
    assert pocket["tokens"].tolist() == [1, 4, 4, 5, 6, 2]


def test_author_benchmark_pocket_score_pooling():
    scores = np.array([[0.9, 0.8, 0.7, 0.6],
                       [0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    assert np.allclose(author_pool_scores(scores, "lit_pcba"), scores.max(axis=0))
    medians = np.median(scores, axis=1, keepdims=True)
    mads = np.median(np.abs(scores - medians), axis=1, keepdims=True)
    expected = (0.6745 * (scores - medians) / (mads + 1e-6)).max(axis=0)
    assert np.allclose(author_pool_scores(scores, "dude"), expected)


def _paired_batch():
    items = []
    for index, atoms in enumerate((["C", "N", "O"], ["N", "C", "S"])):
        xyz = np.array([[0, 0, 0], [1 + index, 0, 0], [0, 1, 0]], dtype=np.float32)
        items.append({"molecule": structure_features(atoms, xyz, max_atoms=8,
                                                      seed=index, unimol=True),
                      "pocket": structure_features(atoms, xyz, max_atoms=8,
                                                    seed=index, unimol_pocket=True),
                      "molecule_id": f"m{index}", "pocket_id": f"p{index}"})
    return collate_pairs(items)


def test_stage_updates_freeze_then_unfreeze():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = PaperDrugCLIP(PaperConfig(gradient_checkpointing=False, dropout=0.0))
    batch = _paired_batch()
    mol_before = model.molecule_encoder.encoder.layers[0].fc1.weight.detach().clone()
    pkt_before = model.pocket_encoder.encoder.layers[0].fc1.weight.detach().clone()
    proj_before = model.molecule_projection.layers[0].weight.detach().clone()
    model.set_stage("profsa")
    model.train()
    assert not model.molecule_encoder.training
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    loss = model.contrastive_loss(model(batch["molecule"], batch["pocket"]),
                                  batch["molecule_ids"], batch["pocket_ids"])
    loss.backward()
    opt.step()
    assert torch.equal(mol_before, model.molecule_encoder.encoder.layers[0].fc1.weight)
    assert not torch.equal(pkt_before, model.pocket_encoder.encoder.layers[0].fc1.weight)
    assert not torch.equal(proj_before, model.molecule_projection.layers[0].weight)
    pkt_after_profsa = model.pocket_encoder.encoder.layers[0].fc1.weight.detach().clone()
    model.set_stage("finetune")
    model.train()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    model.zero_grad(set_to_none=True)
    loss = model.contrastive_loss(model(batch["molecule"], batch["pocket"]),
                                  batch["molecule_ids"], batch["pocket_ids"])
    loss.backward()
    opt.step()
    assert not torch.equal(mol_before, model.molecule_encoder.encoder.layers[0].fc1.weight)
    assert not torch.equal(pkt_after_profsa,
                           model.pocket_encoder.encoder.layers[0].fc1.weight)


def test_pretrained_manifest_rejects_corruption():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        files = ("mol.pt", "dict_mol.txt", "dict_pkt.txt")
        for name in files:
            (root / name).write_bytes(name.encode())
        hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in files}
        manifest = {"schema": "drugclip-pretrained-assets/v2",
                    "molecule": {"path": "mol.pt", "size": 6,
                                 "sha256": hashes["mol.pt"]},
                    "molecule_dictionary": {"path": "dict_mol.txt", "size": 12,
                                             "sha256": hashes["dict_mol.txt"]},
                    "pocket_dictionary": {"path": "dict_pkt.txt", "size": 12,
                                          "sha256": hashes["dict_pkt.txt"]}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        with patch("train.MOLECULE_SHA256", hashes["mol.pt"]), \
             patch("train.MOLECULE_DICTIONARY_SHA256", hashes["dict_mol.txt"]), \
             patch("train.POCKET_DICTIONARY_SHA256", hashes["dict_pkt.txt"]):
            assert pretrained_identity(root) == manifest
            (root / "dict_pkt.txt").write_bytes(b"corrupted")
            with unittest.TestCase().assertRaisesRegex(ValueError, "SHA256 mismatch"):
                pretrained_identity(root)


def test_polynomial_curve_and_resume():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([param], lr=1e-3)
    schedule = PolynomialSchedule(optimizer, base_lr=1e-3, total_updates=100,
                                  warmup_ratio=0.06, power=1.0, end_lr=0.0)
    assert schedule.warmup_updates == 6
    assert abs(optimizer.param_groups[0]["lr"] - 1e-3 / 6) < 1e-12
    curve = []
    for _ in range(50):
        curve.append(schedule.step())
    second = PolynomialSchedule(optimizer, base_lr=1e-3, total_updates=100,
                                warmup_ratio=0.06, power=1.0, end_lr=0.0)
    second.load_state_dict(schedule.state_dict())
    assert second.step() == schedule.step()
    assert curve[5] == 1e-3
    for _ in range(49):
        schedule.step()
    assert schedule.updates == 100
    assert schedule.optimizer.param_groups[0]["lr"] == 0.0


def test_stage_switch_uses_fresh_adam_and_learning_rate():
    param = torch.nn.Parameter(torch.tensor(1.0))
    profsa_optimizer = torch.optim.Adam([param], lr=1e-4)
    profsa_schedule = PolynomialSchedule(profsa_optimizer, base_lr=1e-4,
                                         total_updates=4, warmup_ratio=0.0)
    for _ in range(4):
        profsa_schedule.step()
    assert profsa_optimizer.param_groups[0]["lr"] == 0.0
    finetune_optimizer = torch.optim.Adam([param], lr=1e-3)
    finetune_schedule = PolynomialSchedule(finetune_optimizer, base_lr=1e-3,
                                           total_updates=4, warmup_ratio=0.0)
    assert finetune_schedule.updates == 0
    assert finetune_optimizer.param_groups[0]["lr"] == 1e-3


def test_bedroc_author_function_duplicate_mask_and_selection():
    logits = torch.tensor([[1.0, 9.0, 0.2], [0.1, 2.0, 0.9], [0.4, 0.2, 3.0]])
    ids = ["same", "same", "other"]
    pockets = ["p0", "p1", "p2"]
    masked = masked_logits(logits, ids, pockets)
    assert masked[0, 1] == logits[0, 1] - 1e6 and masked[0, 0] == logits[0, 0]
    assert (masked_logits(logits, ids, ["same", "same", "other"])[0, 1]
            == logits[0, 1] - 2e6)
    got = retrieval_batch_metrics(logits, ids, pockets)
    expected = 0.0
    for row in range(3):
        score = masked[row].numpy()
        label = np.zeros(3)
        label[row] = 1
        order = np.argsort(score)[::-1]
        expected += CalcBEDROC(np.column_stack((score[order], label[order])), 1, 80.5)
    assert abs(got["bedroc_sum"] - expected) < 1e-12
    assert got["queries"] == 3
    old = {"valid_loss": 0.8, "valid_bedroc": 0.7}
    candidate = {"valid_loss": 0.1, "valid_bedroc": 0.6}
    assert candidate["valid_loss"] < old["valid_loss"]
    assert not selection_improves(candidate["valid_bedroc"], old["valid_bedroc"])
    assert not selection_improves(0.7, 0.7)  # tie keeps earlier epoch
    assert selection_improves(0.8, 0.7)


def test_validation_tail_and_distributed_sum():
    class Args:
        validation_batch_size = 3
        workers = 0
        device = "cpu"
    class Dataset:
        def __len__(self): return 11
    left = make_validation_loader(Dataset(), Args(), rank=0, world=2)
    right = make_validation_loader(Dataset(), Args(), rank=1, world=2)
    groups = list(left.batch_sampler) + list(right.batch_sampler)
    assert sorted(index for group in groups for index in group) == list(range(11))
    assert all(len(group) >= 2 for group in groups)
    a = {"loss_sum": 3.0, "bedroc_sum": 2.0, "auc_sum": 1.5,
         "top1_sum": 1.0, "top3_sum": 2.0, "queries": 3.0}
    b = {"loss_sum": 5.0, "bedroc_sum": 3.0, "auc_sum": 2.0,
         "top1_sum": 2.0, "top3_sum": 3.0, "queries": 5.0}
    total = {key: a[key] + b[key] for key in a}
    assert aggregate_metrics(total)["valid_bedroc"] == 5 / 8
    assert aggregate_metrics(total)["valid_queries"] == 8


def test_pocket_crop_matches_author_reference_and_preserves_rng():
    rng = np.random.RandomState(27)
    coordinates = rng.normal(size=(300, 3)).astype(np.float32)
    atoms = ["C", "N", "O", "S", "H"] * 60
    heavy = np.asarray([i for i, atom in enumerate(atoms) if atom != "H"])
    # Add another 60 heavy atoms so cropping happens after dehydrogenation.
    atoms = atoms + ["C", "N", "O", "S"] * 15
    coordinates = np.vstack((coordinates, rng.normal(size=(60, 3)).astype(np.float32)))
    heavy = np.asarray([i for i, atom in enumerate(atoms) if atom != "H"])
    xyz = coordinates[heavy]
    mixed = int(hash((11, 3, 17)) % 1e8)
    author_rng = np.random.RandomState(mixed)
    distance = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)
    logits = np.reciprocal(distance + 1.0)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    expected = author_rng.choice(len(xyz), 256, replace=False, p=weights)
    before = np.random.get_state()
    got = pocket_crop_indices(xyz, 256, seed=11, epoch=3, index=17)
    assert np.array_equal(got, expected)
    after = np.random.get_state()
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(before, after))
    features = structure_features(
        atoms, coordinates, max_atoms=256, seed=1, unimol_pocket=True,
        crop_policy="pocket_distance_weighted", crop_seed=11, epoch=3, index=17,
    )
    selected_atoms = np.asarray(atoms)[heavy][expected]
    token_map = {name: i for i, name in enumerate(UNIMOL_POCKET_VOCABULARY)}
    assert features["tokens"].tolist() == [1, *[token_map[a] for a in selected_atoms], 2]
    selected = xyz[expected]
    selected -= selected.mean(axis=0, keepdims=True)
    expected_dist = np.linalg.norm(selected[:, None] - selected[None, :], axis=-1)
    assert np.allclose(features["distances"][1:-1, 1:-1], expected_dist)
    again = structure_features(
        atoms, coordinates + np.array([5, -7, 3], dtype=np.float32),
        max_atoms=256, seed=1, unimol_pocket=True,
        crop_policy="pocket_distance_weighted", crop_seed=11, epoch=3, index=17,
    )
    assert torch.equal(features["tokens"], again["tokens"])
    assert torch.allclose(features["distances"], again["distances"], atol=1e-4)
    short = structure_features(
        atoms[:50], coordinates[:50], max_atoms=256, seed=1,
        unimol_pocket=True, crop_policy="pocket_distance_weighted",
        crop_seed=11, epoch=3, index=17,
    )
    assert short["tokens"].shape[0] == 42


def test_pocket_inference_crop_is_independent_of_absolute_path():
    import lmdb

    atoms = ["C", "N", "O", "S"] * 75
    coordinates = np.random.RandomState(4).normal(size=(300, 3)).astype(np.float32)
    payload = pickle.dumps({"pocket_atoms": atoms,
                            "pocket_coordinates": coordinates})
    with tempfile.TemporaryDirectory() as directory:
        outputs = []
        for name in ("first", "second"):
            path = Path(directory) / f"{name}.lmdb"
            environment = lmdb.open(str(path), subdir=False, map_size=1 << 20)
            with environment.begin(write=True) as transaction:
                transaction.put(b"0", payload)
            environment.close()
            outputs.append(SingleStructureDataset(
                path, "pocket", max_atoms=256, unimol_pocket=True,
                seed=1)[0]["features"])
        assert torch.equal(outputs[0]["tokens"], outputs[1]["tokens"])
        assert torch.equal(outputs[0]["distances"], outputs[1]["distances"])


def test_rank_zero_stop_precedence():
    common = dict(stage="finetune", epoch=1, max_epochs=2,
                  another_stage=False, elapsed=0.5, max_seconds=1.0,
                  patience_counter=0, patience=20)
    assert stop_decision(**common) == "continue"
    assert stop_decision(**dict(common, elapsed=1.5)) == "time_budget"
    assert stop_decision(**dict(common, patience_counter=20)) == "patience"
    assert stop_decision(**dict(common, epoch=2, elapsed=1.5)) == "max_epochs"
    assert stop_decision(**dict(common, stage="profsa", epoch=2,
                                another_stage=True)) == "next_stage"
    assert stop_decision(**dict(common, stage="profsa", epoch=2,
                                another_stage=True, elapsed=1.5)) == "time_budget"
    assert broadcast_stop("continue", 1.25, device=torch.device("cpu"), world=1) == (
        "continue", 1.25)


def test_molecule_inference_features_are_independent_of_path_and_private_index():
    import lmdb

    atoms = ["C", "N", "O", "S"] * 75
    conformers = [np.random.RandomState(seed).normal(size=(300, 3)).astype(np.float32)
                  for seed in (4, 5)]
    record = {"atoms": atoms, "coordinates": conformers, "smi": "fixture-molecule"}
    with tempfile.TemporaryDirectory() as directory:
        outputs = []
        for name, order, index in (("first", [record, {**record, "smi": "other"}], 0),
                                   ("second", [{**record, "smi": "other"}, record], 1)):
            path = Path(directory) / f"{name}.lmdb"
            environment = lmdb.open(str(path), subdir=False, map_size=1 << 20)
            with environment.begin(write=True) as transaction:
                for key, item in enumerate(order):
                    transaction.put(str(key).encode(), pickle.dumps(item))
            environment.close()
            outputs.append(SingleStructureDataset(
                path, "molecule", max_atoms=64, unimol_molecule=True, seed=1)[index]["features"])
        for key in outputs[0]:
            assert torch.equal(outputs[0][key], outputs[1][key])


def test_old_or_mismatched_profsa_crop_is_rejected():
    with unittest.TestCase().assertRaisesRegex(ValueError, "incompatible pocket crop"):
        require_compatible_profsa_crop({}, 1)
    with unittest.TestCase().assertRaisesRegex(ValueError, "incompatible pocket crop"):
        require_compatible_profsa_crop({"pocket_crop": {
            "policy": "author_distance_weighted_v1", "seed": 9}}, 1)
    require_compatible_profsa_crop({"pocket_crop": {
        "policy": "author_distance_weighted_v1", "seed": 1}}, 1)


def _gloo_stop_worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            rank=rank, world_size=2)
    try:
        decision, elapsed = broadcast_stop(
            "time_budget" if rank == 0 else None,
            4.25 if rank == 0 else -1.0,
            device=torch.device("cpu"), world=2,
        )
        assert decision == "time_budget"
        assert elapsed == 4.25
        # Both workers must reach the collective following the stop decision.
        gathered = [None, None]
        dist.all_gather_object(gathered, rank)
        assert gathered == [0, 1]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_process_gloo_stop_broadcast():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_gloo_stop_worker,
                 args=(str(Path(directory) / "process-group"),), nprocs=2,
                 join=True)

