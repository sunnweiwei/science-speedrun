import sys
from pathlib import Path
import copy
from collections import Counter

import numpy as np
from ase.build import bulk

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/data"))
from prepare_standard import PrioritySelection, priority
from common import encode_atoms
from splits import atoms_key


def test_full_reservoir_is_order_independent_and_honors_parent_caps():
    atoms = encode_atoms(bulk("Cu"))
    group = atoms_key(atoms)
    rows = [
        dict(
            id=f"source/{p}/{f}",
            parent=str(p),
            source="source",
            atoms=atoms,
            energy=0.0,
            forces=np.zeros((1, 3)).tolist(),
            stress=np.zeros((3, 3)).tolist(),
        )
        for p in range(8)
        for f in range(9)
    ]
    results = []
    for ordered in [
        rows,
        rows[::-1],
        sorted(rows, key=lambda r: priority("other:" + r["id"])),
    ]:
        select = PrioritySelection(
            {"train": 13, "val": 13}, (set(), set()), parent_cap=2
        )
        for row in ordered:
            select.offer(copy.deepcopy(row), group)
        chosen = list(select.rows[select.split(group)].values())
        assert len(chosen) == 13
        assert max(Counter(r["parent"] for r in chosen).values()) <= 2
        results.append({r["id"] for r in chosen})
    assert results[0] == results[1] == results[2]
    expected = []
    counts = Counter()
    for row in sorted(rows, key=lambda r: priority(r["id"])):
        if counts[row["parent"]] < 2:
            expected.append(row["id"])
            counts[row["parent"]] += 1
        if len(expected) == 13:
            break
    assert results[0] == set(expected)


def test_standard_selection_excludes_both_physical_partitions():
    select = PrioritySelection({"train": 1, "val": 1}, ({"Cu:1"}, {"blocked"}))
    assert not select.eligible("x", "ok", "Cu:1")
    assert not select.eligible("x", "blocked", "Al:1")


def test_compact_panels_retain_frozen_splits_and_exact_sizes():
    import json
    from panels import DEFAULT_PANELS, COUNTS, select
    import pytest
    from splits import load_split

    if not DEFAULT_PANELS.exists():
        pytest.skip("Materialize --components protocol for frozen-release checks")
    rows, _ = load_split()
    manifest = json.loads(DEFAULT_PANELS.read_text())
    sides = {}
    for split, suites in COUNTS.items():
        selected = []
        for suite, count in suites.items():
            population = [
                r for r in rows if r["split"] == split and r["suite"] == suite
            ]
            panel = select(population, count)
            assert panel == manifest["panels"][split][suite]
            assert select(population[::-1], count) == panel
            assert all(x["selected"] >= 1 for x in panel["strata"].values())
            selected.extend(r for r in population if r["id"] in set(panel["ids"]))
        sides[split] = {r["group"] for r in selected}
    assert not sides["val"] & sides["test"]


def test_learning_curve_aggregation_averages_losses_not_reduced_metrics(tmp_path):
    import json
    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "environment/baseline/uma-mp-finetune"))
    from plot_learning_curves import aggregate

    for rank, loss in [(0, 1.0), (1, 3.0)]:
        (tmp_path / f"steps-rank-{rank}.jsonl").write_text(
            json.dumps({"step": 1, "loss": loss}) + "\n"
        )
        (tmp_path / f"validation-rank-{rank}.jsonl").write_text(
            json.dumps(
                {
                    "step": 1,
                    "weights": "model",
                    "metrics": {"val/loss": loss, "val/omat.val,forces,mae": 0.05},
                }
            )
            + "\n"
        )
    train, val = aggregate(tmp_path, world_size=2)
    assert train[0]["loss"] == val[0]["metrics"]["val/loss"] == 2.0
    assert val[0]["metrics"]["val/omat.val,forces,mae"] == 0.05
    (tmp_path / "steps-rank-1.jsonl").unlink()
    with pytest.raises(ValueError, match="Missing train ranks"):
        aggregate(tmp_path, world_size=2)
