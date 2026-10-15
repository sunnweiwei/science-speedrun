"""Offline checks for the fixed scientific recipe, accounting and dev/final parity."""

import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import relax_worker as worker, relaxation as protocol
from scoring.evaluate import evaluate


class RelaxationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "raw"
        self.output = self.root / "relaxation"
        self.source.mkdir()
        self.structure = SimpleNamespace(composition="Si", atomic_numbers=[14])
        def parse(path):
            if path.read_text() == "unparseable":
                raise ValueError("bad CIF")
            return self.structure
        class Writer:
            def __init__(self, structure):
                pass
            def write_file(self, path):
                path.write_text("relaxed")
        self.modules = {
            "torch": SimpleNamespace(set_num_threads=lambda _: None,
                                     cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)),
            "pymatgen.core": SimpleNamespace(Structure=SimpleNamespace(from_file=parse)),
            "pymatgen.io.cif": SimpleNamespace(CifWriter=Writer),
            "lemat_genbench.models.uma.calculator": SimpleNamespace(create_uma_calculator=lambda **kw: object()),
        }

    def inventory(self, contents):
        records = []
        for index, text in enumerate(contents):
            path = self.source / f"{index}.cif"
            path.write_text(text)
            records.append({"request_id": "unconditional", "sample_index": index,
                            "source_path": str(path), "structure_path": f"structures/{index}.cif",
                            "source_sha256": protocol.sha256(path)})
        return records

    def run_worker(self, records, effect=None, index=0, workers=1, claims=None):
        with patch.dict(sys.modules, self.modules), patch.object(worker.signal, "signal"), \
             patch.object(worker.signal, "alarm"), \
             patch.object(worker, "refine", side_effect=effect,
                          return_value=(self.structure, {"converged": False, "steps": 200})) as refine:
            worker.run_worker(self.output, records, index, workers, claims)
            return refine.call_count

    def test_claimed_samples_are_relaxed_once_across_workers(self):
        records = self.inventory(["a", "b", "c"])
        claims = self.root / "claims"
        claims.mkdir()
        (claims / "000001").touch()  # another worker holds sample 1
        self.assertEqual(self.run_worker(records, index=0, workers=3, claims=claims), 2)
        self.assertEqual(self.run_worker(records, index=1, workers=3, claims=claims), 0)
        self.assertEqual(sorted(p.name for p in claims.iterdir()), ["000000", "000001", "000002"])
        self.assertFalse((self.output / "samples/000001.json").exists())

    def test_workers_per_gpu_follows_cpus_and_override(self):
        with patch.object(protocol.os, "sched_getaffinity", return_value=set(range(64))), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CRYSTALGEN_RELAX_WORKERS_PER_GPU", None)
            self.assertEqual(protocol.workers_per_gpu(8), 4)
            self.assertEqual(protocol.workers_per_gpu(32), 1)
        with patch.dict(os.environ, {"CRYSTALGEN_RELAX_WORKERS_PER_GPU": "2"}):
            self.assertEqual(protocol.workers_per_gpu(8), 2)

    def test_failed_samples_preserve_original_bytes_and_finite_unconverged_are_kept(self):
        records = self.inventory(["raw", "unparseable"])
        self.assertEqual(self.run_worker(records), 1)
        self.assertEqual((self.output / "generated/structures/0.cif").read_text(), "relaxed")
        self.assertEqual((self.output / "generated/structures/1.cif").read_text(), "unparseable")
        for item in records:
            self.assertEqual(protocol.sha256(Path(item["source_path"])), item["source_sha256"])
        results = [json.loads(p.read_text()) for p in sorted((self.output / "samples").glob("*.json"))]
        self.assertFalse(results[0]["converged"])
        self.assertEqual(results[1]["status"], "fallback_original")

    def test_retry_reuses_only_hash_verified_outputs(self):
        records = self.inventory(["raw"])
        self.assertEqual(self.run_worker(records), 1)
        self.assertEqual(self.run_worker(records), 0)
        (self.output / "generated/structures/0.cif").write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "cached relaxed sample changed"):
            self.run_worker(records)

    def test_source_mutation_is_rejected(self):
        records = self.inventory(["raw"])
        (self.source / "0.cif").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "source CIF changed"):
            self.run_worker(records)

    def test_infrastructure_errors_stop_scoring(self):
        records = self.inventory(["raw"])
        for error in (RuntimeError("CUDA out of memory"), OSError("disk full"), ImportError("missing module")):
            with self.subTest(error=error), self.assertRaises(type(error)):
                self.run_worker(records, error)
        self.assertFalse((self.output / "samples").exists())

    def test_missing_model_stops_even_for_unparseable_candidates(self):
        records = self.inventory(["unparseable"])
        self.modules["lemat_genbench.models.uma.calculator"] = SimpleNamespace(
            create_uma_calculator=lambda **kw: (_ for _ in ()).throw(RuntimeError("checkpoint absent")))
        with self.assertRaisesRegex(RuntimeError, "checkpoint absent"):
            self.run_worker(records)

    def test_timeout_keeps_original_candidate(self):
        records = self.inventory(["raw"])
        self.run_worker(records, TimeoutError("sample time limit"))
        result = json.loads((self.output / "samples/000000.json").read_text())
        self.assertEqual(result["status"], "fallback_original")
        self.assertEqual(protocol.sha256(self.output / "generated/structures/0.cif"), records[0]["source_sha256"])

    def test_frozen_recipe_matches_completed_experiment(self):
        self.assertEqual((protocol.RECIPE["model"], protocol.RECIPE["task"]), ("uma-s-1", "omat"))
        self.assertEqual((protocol.RECIPE["optimizer"], protocol.RECIPE["cell_filter"]),
                         ("BFGSLineSearch", "FrechetCellFilter"))
        self.assertEqual(protocol.RECIPE["fmax_ev_per_angstrom"], 0.02)
        self.assertEqual(protocol.RECIPE["max_steps"], 200)
        self.assertEqual(protocol.RECIPE["sample_timeout_seconds"], 300)

    def test_frechet_guard_rejects_singletons_without_changing_two_sample_path(self):
        # Apply the real runtime patch to the upstream function's matching
        # prefix, with a lightweight stack stub in place of NumPy/LAPACK.
        source = self.root / "src/lemat_genbench/utils/distribution_utils.py"
        source.parent.mkdir(parents=True)
        source.write_text('\n' * 577 +
            'def compute_frechetdist_with_cache(mu1_cached, sigma1_cached, generated_crystals):\n'
            '    """Compute Fréchet distance using pre-computed reference statistics."""\n'
            '    Y = np.stack(generated_crystals, axis=0)\n'
            '    return Y\n')
        patch_path = Path(__file__).resolve().parent / "runtime/lemat-small-batch.patch"
        subprocess.run(["git", "apply", str(patch_path)], cwd=self.root, check=True, capture_output=True)
        calls = []
        def stack(values, axis):
            calls.append(values)
            return values
        namespace = {"np": SimpleNamespace(stack=stack)}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        calculate = namespace["compute_frechetdist_with_cache"]
        for values in ([], [[1.0]]):
            with self.assertRaisesRegex(ValueError, "at least two"):
                calculate(None, None, values)
        self.assertEqual(calls, [])
        values = [[1.0], [2.0]]
        self.assertIs(calculate(None, None, values), values)
        self.assertEqual(calls, [values])

    def test_empty_batch_needs_no_gpu_and_keeps_denominator(self):
        with patch.object(protocol, "run_workers") as run:
            records, summary = protocol.relax_records({}, 2500, self.output, Path("unused"))
        run.assert_not_called()
        self.assertEqual(records, {})
        self.assertEqual(summary["requested"], 2500)
        self.assertEqual(summary["missing_or_invalid_manifest"], 2500)
        with self.assertRaises(FileExistsError):
            protocol.relax_records({}, 2500, self.output, Path("unused"))

    def test_aggregation_keeps_missing_and_failed_candidates(self):
        inventory = self.inventory(["raw", "unparseable"])
        def launch(path, output, python, samples):
            self.run_worker(json.loads(path.read_text()))
        records = {("unconditional", item["sample_index"]): Path(item["source_path"]) for item in inventory}
        with patch.object(protocol, "run_workers", side_effect=launch):
            relaxed, summary = protocol.relax_records(records, 3, self.output, Path("unused"))
        self.assertEqual(len(relaxed), 2)
        self.assertEqual((summary["requested"], summary["relaxed"], summary["fallback_original"],
                          summary["missing_or_invalid_manifest"]), (3, 1, 1, 1))

    def test_relaxation_runs_before_metrics_by_default(self):
        requests = {"u": {"mode": "unconditional", "n_samples": 2500}}
        raw, refined = {("u", 0): self.source / "0.cif"}, {("u", 0): self.output / "0.cif"}
        with patch("scoring.evaluate.load_requests", return_value=requests), \
             patch("scoring.evaluate.load_output", return_value=(raw, ["missing"])), \
             patch("scoring.evaluate.relax_records", return_value=(refined, {"requested": 2500})) as relax, \
             patch("scoring.evaluate.run_lemat", return_value=({"sun": 0.1}, {"counts": {"expected": 2500}})) as score:
            rewards, report = evaluate(self.root, self.source, Path("scorer"), Path("python"))
        relax.assert_called_once()
        self.assertIs(score.call_args.args[0], refined)
        self.assertEqual(score.call_args.args[2], 2500)
        self.assertEqual(report["output_errors"], ["missing"])
        self.assertEqual(rewards["unconditional_sun"], 0.1)

    def test_dev_diagnostic_can_skip_but_final_has_no_bypass(self):
        with patch("scoring.evaluate.load_requests", return_value={"u": {"mode": "unconditional", "n_samples": 2}}), \
             patch("scoring.evaluate.load_output", return_value=({}, [])), \
             patch("scoring.evaluate.relax_records") as relax, \
             patch("scoring.evaluate.run_lemat", return_value=({}, {})):
            _, report = evaluate(self.root, self.source, Path("scorer"), Path("python"), skip_relaxation=True)
        relax.assert_not_called()
        self.assertEqual(report["relaxation"]["status"], "skipped_development_diagnostic")
        self.assertNotIn("skip_relaxation", Path(__file__).with_name("grader.py").read_text())

    def test_gpu_mapping_handles_explicit_and_implicit_masks(self):
        with patch.object(protocol.subprocess, "check_output", return_value="2\n"), \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}):
            self.assertEqual(protocol.gpu_tokens(Path("python")), ["GPU-a", "GPU-b"])
        with patch.object(protocol.subprocess, "check_output", return_value="0\n"):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                protocol.gpu_tokens(Path("python"))


if __name__ == "__main__":
    unittest.main()
