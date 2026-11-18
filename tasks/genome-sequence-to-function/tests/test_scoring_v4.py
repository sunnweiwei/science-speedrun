"""Prediction contract v4: sparse splice usage and junctions, shared by grader and agent."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment" / "evaluation"))
from scoring import (FIXED_SHAPES, GENOME_CONSTANT, MAX_JUNCTION_PAIRS, PerTrackMoments,  # noqa: E402
                     TRACK_GROUPS, _junction_prediction, _usage_prediction,
                     ssi_v4_genome_score, validate_example)


def test_oracle_moments_score_one():
    target = np.array([[0, 2], [1, 1], [2, 0]], np.float32)
    moments = PerTrackMoments(2)
    moments.update(target, target)
    assert np.isclose(moments.report()["mean_pearson"], 1.0)


def _example(tmp_path: Path, pairs=((0, 104, 209),), usage_sites=(5, 17)):
    group = zarr.open_group(str(tmp_path / "p.zarr"), mode="w")
    for name, shape in FIXED_SHAPES.items():
        group.create_dataset(name, shape=shape, dtype="f2", fill_value=0, chunks=(4096, 64))
    group.create_dataset("splice_usage_positions", data=np.asarray(usage_sites, np.int32))
    usage = np.zeros((len(usage_sites), 734), np.float16); usage[1, 3] = 2
    group.create_dataset("splice_usage", data=usage)
    half, donor, acceptor = (np.asarray(v) for v in zip(*pairs)) if pairs else ([], [], [])
    group.create_dataset("junction_half", data=np.asarray(half, np.int8))
    group.create_dataset("junction_donor", data=np.asarray(donor, np.int32))
    group.create_dataset("junction_acceptor", data=np.asarray(acceptor, np.int32))
    values = np.zeros((len(pairs), 734), np.float16)
    if len(pairs):
        values[0, 3] = 7
    group.create_dataset("junctions", data=values)
    return group


def test_usage_at_unlisted_hidden_site_is_a_zero_prediction(tmp_path: Path):
    group = _example(tmp_path)
    values, missing = _usage_prediction(group, np.array([17, 9]))
    assert values[0, 3] == 2 and not values[1].any()
    assert missing.tolist() == [False, True]


def test_junction_query_matches_strand_half_and_both_endpoints(tmp_path: Path):
    group = _example(tmp_path)
    halves = np.repeat([0, 1], 367)  # track 3 is a plus-strand (half 0) track
    queries = pd.DataFrame({"donor_position": [104, 104, 105], "acceptor_position": [209, 209, 209],
                            "track": [3, 400, 3]})
    values, missing = _junction_prediction(group, queries, halves)
    assert values.tolist() == [7.0, 0.0, 0.0]
    assert missing.tolist() == [False, True, True]


def test_validation_accepts_the_contract_and_rejects_bad_keys(tmp_path: Path):
    validate_example(_example(tmp_path), "ok")
    with pytest.raises(ValueError, match="unique"):
        validate_example(_example(tmp_path / "a", pairs=((0, 1, 2), (0, 1, 2))), "dup")
    with pytest.raises(ValueError, match="sorted"):
        validate_example(_example(tmp_path / "b", usage_sites=(17, 5)), "order")
    with pytest.raises(ValueError, match="out of range"):
        validate_example(_example(tmp_path / "c", pairs=((2, 1, 2),)), "half")
    group = _example(tmp_path / "d", pairs=())
    validate_example(group, "empty")  # a model may list no junction pairs at all
    assert MAX_JUNCTION_PAIRS == 16_384


def _load_harness_v3():
    import importlib.util
    harness_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    spec = importlib.util.spec_from_file_location("ssi_index_v3", harness_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_ssi_v4_genome_matches_harness_v3_and_excludes_eqtl_sign():
    import json
    v3 = _load_harness_v3()
    task_root = Path(__file__).resolve().parents[1]
    rep_dir = task_root / "environment/baseline/alphagenome-reference/reproduction"
    vt = json.loads((rep_dir / "validation_tracks.json").read_text())
    vv = json.loads((rep_dir / "validation_variants.json").read_text())

    emp_fams = {k: v["mean_pearson"] for k, v in vt["families"].items()}
    emp_vars = {k: v["model"] for k, v in vv.items()}
    all_fams = [k for names in TRACK_GROUPS.values() for k in names]

    fixtures = [
        ("empirical_215m", emp_fams, emp_vars),
        ("trivial_constant", {k: 0.0 for k in all_fams}, dict(GENOME_CONSTANT)),
        (
            "negative_below_chance",
            {k: -0.15 for k in all_fams},
            {k: (b - 0.1) for k, b in GENOME_CONSTANT.items()},
        ),
        (
            "oracle_upper_bound",
            {k: 1.0 for k in all_fams},
            {k: 1.0 for k in GENOME_CONSTANT},
        ),
        (
            "partial_tracks_only_signal",
            {k: (0.6 if k in ("atac", "dnase") else 0.0) for k in all_fams},
            dict(GENOME_CONSTANT),
        ),
    ]
    for name, fams, var_map in fixtures:
        got = ssi_v4_genome_score(fams, var_map)
        expected = v3.genome(fams, var_map)
        assert abs(got["score"] - expected["score"]) < 1e-12, name
        assert abs(got["reward"] - expected["score"]) < 1e-12, name
        assert abs(got["signed_score"] - expected["signed_score"]) < 1e-12, name
        if name == "trivial_constant":
            assert abs(got["score"]) < 1e-15
            assert abs(got["tracks_axis"]) < 1e-15
            assert abs(got["variants_axis"]) < 1e-15
        if name == "negative_below_chance":
            assert got["signed_score"] < 0.0
            assert got["score"] == 0.0
        if name == "oracle_upper_bound":
            assert abs(got["score"] - 1.0) < 1e-15

    # Verify eqtl_sign is strictly diagnostic and excluded from reward
    low_sign = ssi_v4_genome_score(emp_fams, {**emp_vars, "eqtl_sign": 0.0})
    high_sign = ssi_v4_genome_score(emp_fams, {**emp_vars, "eqtl_sign": 1.0})
    assert abs(low_sign["score"] - high_sign["score"]) == 0.0


def test_grader_end_to_end_v3_parity_and_pure_sequence_anchors(tmp_path: Path):
    import json
    import subprocess
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import sha256, tree_sha256, write_json
    from variant_scoring import METRICS, PERTURBATION_SCHEMA

    v3 = _load_harness_v3()
    labels = tmp_path / "labels_root"
    (labels / "public").mkdir(parents=True)
    (labels / "labels").mkdir(parents=True)
    (labels / "scoring/test").mkdir(parents=True)
    (labels / "targets/test").mkdir(parents=True)
    ve = labels / "variant_evals"
    (ve / "public").mkdir(parents=True)
    (ve / "labels").mkdir(parents=True)

    pd.DataFrame({
        "example_id": ["ex0"], "split": ["test"], "chromosome": ["chr1"],
        "input_start": [0], "input_end": [1_048_576], "input_pad_left": [0],
        "input_pad_right": [0], "target_start": [425_984], "target_end": [622_592],
    }).to_parquet(labels / "public/test.parquet", index=False)
    pd.DataFrame({
        "tensor_index": np.arange(734),
        "junction_channel_half": np.repeat([0, 1], 367),
    }).to_parquet(labels / "public/junction_tracks.parquet", index=False)

    t_store = zarr.open_group(str(labels / "targets/test/s0.zarr"), mode="w")
    tg = t_store.create_group("examples/ex0")
    rng = np.random.default_rng(7)
    sealed = tmp_path / "phase/sealed"
    p_store = zarr.open_group(str(sealed / "predictions.zarr"), mode="w")
    p_store.attrs["schema"] = "genome-sequence-to-function-predictions/v4"
    pg = p_store.create_group("examples/ex0")

    for name, shape in FIXED_SHAPES.items():
        y = rng.uniform(0.1, 2.0, size=shape).astype(np.float32)
        if name == "splice_sites":
            tg.create_dataset("splice_sites/values", data=y)
            pg.create_dataset("splice_sites", data=(0.8 * y + 0.1).astype(np.float32))
        elif name == "contact_maps":
            tg.create_dataset("contact_maps/upper_triangle", data=np.sinh(y))
            tg.create_dataset("contact_maps/mask", data=np.ones(shape, dtype=bool))
            pg.create_dataset("contact_maps", data=(0.7 * y).astype(np.float32))
        else:
            tg.create_dataset(f"{name}/values", data=np.sinh(y))
            tg.create_dataset(f"{name}/mask", data=np.ones(shape, dtype=bool))
            pg.create_dataset(name, data=(0.75 * y).astype(np.float32))

    tg.create_dataset("splice_site_usage/positions", data=np.array([2, 5, 9], np.int32))
    u_val = rng.uniform(0.1, 1.5, size=(3, 734)).astype(np.float32)
    tg.create_dataset("splice_site_usage/values", data=np.sinh(u_val))
    tg.create_dataset("splice_site_usage/mask", data=np.ones((3, 734), dtype=bool))
    pg.create_dataset("splice_usage_positions", data=np.array([2, 5, 9], np.int32))
    pg.create_dataset("splice_usage", data=(0.5 * u_val).astype(np.float32))

    pg.create_dataset("junction_half", data=np.array([0, 0, 0], np.int8))
    pg.create_dataset("junction_donor", data=np.array([10, 20, 30], np.int32))
    pg.create_dataset("junction_acceptor", data=np.array([100, 200, 300], np.int32))
    j_mat = np.zeros((3, 734), np.float32)
    j_mat[:, 0] = [0.2, 0.9, 1.6]
    pg.create_dataset("junctions", data=j_mat)

    pd.DataFrame({
        "donor_position": [10, 20, 30],
        "acceptor_position": [100, 200, 300],
        "track": [0, 0, 0],
        "target_value": np.sinh(np.array([0.3, 0.8, 1.5], np.float32)),
        "valid": [True, True, True],
    }).to_parquet(labels / "scoring/test/part-000000.parquet", index=False)

    pd.DataFrame({
        "example_id": ["ex0"],
        "target_shard": ["targets/test/s0.zarr"],
        "target_group": ["examples/ex0"],
    }).to_parquet(labels / "labels/test.parquet", index=False)

    # Synthetic perturbations & 13 variant label tables
    all_dense = ["atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone"]
    pd.DataFrame({
        "perturbation_id": ["v0", "v1"],
        "families": [all_dense + ["splice_sites"], all_dense + ["splice_sites"]],
        "dense_bin_start": [0, 0], "dense_bin_end": [4, 4],
        "splice_start": [0, 0], "splice_end": [8, 8],
    }).to_parquet(ve / "public/perturbations.parquet", index=False)

    tracks_rows = []
    for fam, cnt in [("ATAC", 167), ("DNASE", 305), ("PROCAP", 12), ("CAGE", 546),
                     ("RNA_SEQ", 667), ("CHIP_TF", 1617), ("CHIP_HISTONE", 1116)]:
        for idx in range(cnt):
            tracks_rows.append({
                "family": fam,
                "data_source": "encode",
                "strand": "POSITIVE",
                "ontology_curie": "EFO:0002784" if fam in ("DNASE", "CHIP_TF") else "EFO:0002067",
                "name": "SPI1 track" if fam == "CHIP_TF" and idx == 0 else f"{fam}_{idx}",
            })
    pd.DataFrame(tracks_rows).to_parquet(ve / "labels/tracks.parquet", index=False)

    pert_zarr = zarr.open_group(str(sealed / "perturbations.zarr"), mode="w")
    pert_zarr.attrs.update({"schema": PERTURBATION_SCHEMA, "perturbation_ids": ["v0", "v1"]})
    for idx, amp in enumerate([0.0, 2.0]):
        for fam, cnt in [("atac", 167), ("dnase", 305), ("procap", 12), ("cage", 546),
                         ("rna_seq", 667), ("chip_tf", 1617), ("chip_histone", 1116)]:
            base_a = np.zeros((4, cnt), np.float32)
            base_b = np.full((4, cnt), amp, np.float32)
            pert_zarr.create_dataset(f"perturbations/p{idx:06d}/a/{fam}", data=base_a)
            pert_zarr.create_dataset(f"perturbations/p{idx:06d}/b/{fam}", data=base_b)
        sa = np.full((8, 4), -3.0, np.float32)
        sb = sa.copy()
        if amp > 0:
            sb[2, 0] = 3.0
        pert_zarr.create_dataset(f"perturbations/p{idx:06d}/a/splice_sites", data=sa)
        pert_zarr.create_dataset(f"perturbations/p{idx:06d}/b/splice_sites", data=sb)

    for eval_name, (_, agg, _) in METRICS.items():
        if agg == "tissue":
            df = pd.DataFrame({
                "perturbation_id": ["v0", "v1"],
                "tissue": ["Whole_Blood", "Whole_Blood"],
                "gene_strand": ["+", "+"],
                "exon_bins": [[0, 1], [0, 1]],
                "target": [0, 1] if eval_name != "eqtl_coefficient" else [0.1, 0.9],
                "alphagenome_prediction": [0.2, 0.8],
            })
        elif agg == "chrom":
            df = pd.DataFrame({
                "perturbation_id": ["v0", "v1"],
                "chrom": ["chr1", "chr1"],
                "label": [0, 1],
                "anchor_Borzoi_L2_L2.plus.all": [0.1, 0.8],
                "anchor_Enformer_L2_L2.plus.all": [0.2, 0.7],
                "anchor_CADD.plus.RawScore": [0.05, 0.95],
                "anchor_GPN-MSA_absLLR.plus.score": [0.05, 0.92],
            })
        else:
            df = pd.DataFrame({
                "perturbation_id": ["v0", "v1"],
                "gene_strand": ["+", "+"],
                "exon_bins": [[0, 1], [0, 1]],
                "target": [0, 1],
                "alphagenome_prediction": [0.1, 0.9],
            })
        df.to_parquet(ve / f"labels/{eval_name}.parquet", index=False)

    snap = {
        "schema": "genome-sequence-to-function-prediction-snapshot/v4",
        "input_manifest_sha256": sha256(labels / "public/test.parquet"),
        "prediction_tree_sha256": tree_sha256(sealed / "predictions.zarr"),
        "perturbation_manifest_sha256": sha256(ve / "public/perturbations.parquet"),
        "perturbation_tree_sha256": tree_sha256(sealed / "perturbations.zarr"),
    }
    write_json(sealed / "snapshot.json", snap)
    (sealed.parent / "READY").write_text(sha256(sealed / "snapshot.json") + "\n")

    out_dir = tmp_path / "graded"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "grader.py"),
            "--sealed", str(sealed),
            "--labels", str(labels),
            "--output", str(out_dir),
            "--wait-seconds", "5",
        ],
        check=True,
    )
    rew = json.loads((out_dir / "reward.json").read_text())
    rep = json.loads((out_dir / "report.json").read_text())
    expected = v3.genome(
        {k: v["mean_pearson"] for k, v in rep["families"].items()},
        {k: v["model"] for k, v in rep["variant_effects"].items()},
    )
    assert abs(rew["score"] - expected["score"]) < 1e-12
    assert abs(rew["reward"] - expected["score"]) < 1e-12
    assert abs(rew["signed_score"] - expected["signed_score"]) < 1e-12

    # Ensure CADD and GPN-MSA are excluded from pure-sequence anchors
    tg_anchors = rep["variant_effects"]["traitgym_mendelian"]["anchors"]
    assert set(tg_anchors.keys()) == {"Borzoi_L2_L2.plus.all", "Enformer_L2_L2.plus.all"}
    assert "CADD.plus.RawScore" not in tg_anchors


def test_genome_docs_pure_sequence_decontamination_and_visibility_boundary():
    task_root = Path(__file__).resolve().parents[1]
    instruction = (task_root / "instruction.md").read_text()
    contract = (task_root / "environment/contract.md").read_text()
    refs_readme = (task_root / "environment/references/README.md").read_text()
    tests_readme = (task_root / "tests/README.md").read_text()
    dev_dockerfile = (task_root / "environment/Dockerfile").read_text()
    dev_compose = (task_root / "environment/docker-compose.yaml").read_text()

    for doc in (instruction, contract):
        assert "0%" not in doc
        assert "not part of `reward`" not in doc
        assert "0.5 * tracks_axis + 0.5 * variants_axis" in doc
        assert "eqtl_sign" in doc

    assert "CADD" not in refs_readme
    assert "CADD" not in tests_readme
    assert "CADD" not in instruction

    for heading in (
        "## Scientific & Literature Justification of Verifier Design",
        "### 1. Primary Metric & Upstream Paper Provenance",
        "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
        "### 3. Diagnostic Signals Retained in report.json",
    ):
        assert heading in tests_readme

    assert "COPY tests" not in dev_dockerfile
    assert "../tests" not in dev_dockerfile
    assert "/tests" not in dev_compose
