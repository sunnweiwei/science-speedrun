from __future__ import annotations

import gzip
from pathlib import Path

from agbench.protocol import INPUT_BP
from agbench.manifests import (
    apply_train_exclusion,
    build_full_manifest,
    select_splits,
)
from agbench.source_index import _model_input_span, _target_span


def _write_folds(path: Path) -> None:
    rows = [
        ("chr1", 2_000_000, 2_196_608, "fold0"),
        ("chr1", 5_000_000, 5_196_608, "fold1"),
    ]
    for fold in range(2, 8):
        for index in range(4):
            start = 8_000_000 + fold * 2_000_000 + index * 1_500_000
            rows.append(("chr1", start, start + 196_608, f"fold{fold}"))
    with gzip.open(path, "wt") as handle:
        for row in rows:
            handle.write("\t".join(map(str, row)) + "\n")


def test_manifest_is_source_anchored_and_deterministic(tmp_path: Path):
    bed = tmp_path / "folds.bed.gz"
    fai = tmp_path / "reference.fa.fai"
    _write_folds(bed)
    fai.write_text("chr1\t50000000\t0\t60\t61\n")
    full = build_full_manifest(bed, fai)
    assert set(full.source_fold) == {f"fold{i}" for i in range(8)}
    assert (full.input_end - full.input_start == INPUT_BP).all()
    assert (full.target_end - full.target_start == 196_608).all()
    assert not full.example_id.duplicated().any()
    assert full.equals(build_full_manifest(bed, fai))


def test_overlap_filter_removes_training_only(tmp_path: Path):
    bed = tmp_path / "folds.bed.gz"
    fai = tmp_path / "reference.fa.fai"
    _write_folds(bed)
    fai.write_text("chr1\t50000000\t0\t60\t61\n")
    full = build_full_manifest(bed, fai)
    audit = apply_train_exclusion(full)
    assert set(audit.source_fold).isdisjoint({"fold0", "fold1"})
    assert len(full.loc[full.split == "test"]) == 1


def test_selection_never_exceeds_available_rows(tmp_path: Path, monkeypatch):
    bed = tmp_path / "folds.bed.gz"
    fai = tmp_path / "reference.fa.fai"
    _write_folds(bed)
    fai.write_text("chr1\t50000000\t0\t60\t61\n")
    full = build_full_manifest(bed, fai)
    selected, counts = select_splits(full)
    assert len(selected["valid"]) == 1
    assert len(selected["test"]) == 1
    assert len(selected["train"]) <= counts["train_eligible"]


def test_target_span_is_centered_in_one_mb_source_interval():
    assert _target_span(10, 10 + 1_048_576) == (425_994, 622_602)


def test_official_padded_context_is_cropped_symmetrically_to_one_mb():
    source = (150_579_770, 151_632_442)
    assert source[1] - source[0] == 1_052_672
    assert _target_span(*source) == (151_007_802, 151_204_410)
    assert _model_input_span(*source) == (150_581_818, 151_630_394)
