import pytest
from ase.build import bulk

from common import encode_atoms, write_json, write_rows
from distributed import chunk_plan, merge
from predict import load_calculator, run


def cases():
    return [
        {
            "id": f"discovery/copper-{i}",
            "suite": "discovery",
            "atoms": encode_atoms(bulk("Cu")),
        }
        for i in range(3)
    ]


def test_chunk_plan_is_deterministic_and_exact():
    rows = cases()
    plan = chunk_plan(rows)
    assert plan == chunk_plan(rows[::-1])
    assert sorted(key for c in plan for key in c["case_ids"]) == sorted(
        r["id"] for r in rows
    )
    with pytest.raises(ValueError, match="Duplicate"):
        chunk_plan(rows + rows[:1])


def test_reused_calculator_and_strict_merge(tmp_path):
    rows = cases()
    plan = chunk_plan(rows)
    canonical = tmp_path / "inputs.jsonl"
    write_rows(canonical, rows)
    chunk = plan[0]
    idx = {r["id"]: r for r in rows}
    chunk_inputs = tmp_path / "chunk-inputs.jsonl"
    write_rows(chunk_inputs, [idx[k] for k in chunk["case_ids"]])
    destination = tmp_path / "chunks" / chunk["chunk_id"]
    calculator = load_calculator("emt", None, "cpu")
    run(chunk_inputs, destination, "emt", calculator=calculator, verbose=False)
    (destination / "inputs.jsonl").write_bytes(chunk_inputs.read_bytes())
    write_json(
        destination / "worker.json",
        {"worker_id": "cpu-test", "chunk_input_sha256": chunk["input_sha256"]},
    )
    result = merge(canonical, tmp_path / "chunks", plan, tmp_path / "merged", None)
    assert result["coverage"] == {"ok": 3}
    assert result["suites"]["discovery"]["cases"] == 3
    # Corruption, even if all IDs remain present, cannot pass result assembly.
    with (destination / "predictions.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Corrupt"):
        merge(canonical, tmp_path / "chunks", plan, tmp_path / "bad", None)


def test_merge_rejects_missing_or_duplicated_work(tmp_path):
    path = tmp_path / "inputs.jsonl"
    write_rows(path, cases())
    plan = chunk_plan(cases())
    with pytest.raises(ValueError, match="exactly cover"):
        merge(path, tmp_path / "chunks", plan + plan, tmp_path / "bad", None)
    with pytest.raises(FileNotFoundError):
        merge(path, tmp_path / "chunks", plan, tmp_path / "missing", None)


def test_time_limit_merge_keeps_finished_chunks_and_fails_the_rest(tmp_path):
    rows = [dict(r, id=f"discovery/copper-{i}") for i, r in enumerate(cases() * 70)]  # 210 cases: 2 chunks
    plan = chunk_plan(rows)
    assert len(plan) == 2
    canonical = tmp_path / "inputs.jsonl"
    write_rows(canonical, rows)
    idx = {r["id"]: r for r in rows}
    finished = plan[0]
    chunk_inputs = tmp_path / "chunk-inputs.jsonl"
    write_rows(chunk_inputs, [idx[k] for k in finished["case_ids"]])
    destination = tmp_path / "chunks" / finished["chunk_id"]
    run(chunk_inputs, destination, "emt", calculator=load_calculator("emt", None, "cpu"), verbose=False)
    (destination / "inputs.jsonl").write_bytes(chunk_inputs.read_bytes())
    write_json(destination / "worker.json",
               {"worker_id": "cpu-test", "chunk_input_sha256": finished["input_sha256"]})
    with pytest.raises(FileNotFoundError):  # without the time limit an unfinished chunk is an error
        merge(canonical, tmp_path / "chunks", plan, tmp_path / "strict", None)
    result = merge(canonical, tmp_path / "chunks", plan, tmp_path / "merged", None,
                   unfinished_reason="verifier time limit reached")
    assert result["coverage"] == {"ok": len(finished["case_ids"]), "failed": len(plan[1]["case_ids"])}
    import json
    meta = json.loads((tmp_path / "merged" / "run.json").read_text())
    assert meta["complete"] and meta["unfinished_cases"] == len(plan[1]["case_ids"])
