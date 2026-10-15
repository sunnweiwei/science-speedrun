"""Synthetic unit tests for the trusted metric and grading code."""
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import grader
import metrics

DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "environment" / "data"


def load_data_script(name):
    spec = importlib.util.spec_from_file_location(name, DATA_SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PretrainSourcePinTests(unittest.TestCase):
    def test_drifted_uniref50_fails_closed_unless_allowed(self):
        builder = load_data_script("build_pretrain_subset")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "uniref50.fasta.gz"
            source.write_bytes(b"not the pinned release")
            with self.assertRaises(SystemExit) as raised:
                builder.verify_pinned_source(source)
            self.assertIn(builder.PINNED_RELEASE, str(raised.exception))
            self.assertIn("--allow-drift", str(raised.exception))
            drifted = builder.verify_pinned_source(source, allow_drift=True)
            self.assertFalse(drifted["matches_pin"])
            self.assertEqual(drifted["size"], source.stat().st_size)
            self.assertEqual(len(drifted["sha256"]), 64)
            self.assertEqual(len(builder.PINNED_SHA256), 64)
            self.assertGreater(builder.PINNED_SIZE, 8_000_000_000)


class MetricTests(unittest.TestCase):
    def test_classification_perfect_and_uniform(self):
        truth = {f"e{i}": i % 2 for i in range(10)}
        perfect = {k: {"scores": [1.0 if c == v else 0.0 for c in range(2)]} for k, v in truth.items()}
        result = metrics.score_task("solubility", truth, perfect)
        self.assertEqual(result["reward"], 1.0)
        uniform = {k: {"scores": [1, 1]} for k in truth}
        result = metrics.score_task("solubility", truth, uniform)
        self.assertLessEqual(result["reward"], 0.6)

    def test_spearman_reward_clamps_at_zero(self):
        truth = {f"e{i}": float(i) for i in range(20)}
        forward = {k: {"value": v * 2 + 1} for k, v in truth.items()}
        self.assertAlmostEqual(metrics.score_task("fluorescence", truth, forward)["reward"], 1.0)
        reverse = {k: {"value": -v} for k, v in truth.items()}
        self.assertEqual(metrics.score_task("fluorescence", truth, reverse)["reward"], 0.0)

    def test_ppi_affinity_rmse_reward(self):
        truth = {f"e{i}": float(i) for i in range(5)}
        exact = {k: {"value": v} for k, v in truth.items()}
        result = metrics.score_task("ppi_affinity", truth, exact, ppi_affinity_std=2.0)
        self.assertAlmostEqual(result["reward"], 1.0)
        off = {k: {"value": v + 2.0} for k, v in truth.items()}
        result = metrics.score_task("ppi_affinity", truth, off, ppi_affinity_std=2.0)
        self.assertAlmostEqual(result["reward"], math.exp(-1.0))

    def test_ssp_masked_accuracy(self):
        truth = {"a": {"ss3": [0, 1, 2, 1], "valid_mask": [1, 1, 0, 1]}}
        prediction = {"a": {"ss3_scores": [[9, 0, 0], [0, 9, 0], [9, 0, 0], [9, 0, 0]]}}
        result = metrics.score_task("secondary_structure", truth, prediction)
        self.assertEqual(result["valid_residues"], 3)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)

    def test_malformed_predictions_raise(self):
        truth = {"a": 0, "b": 1}
        with self.assertRaises(ValueError):  # missing prediction
            metrics.score_task("solubility", truth, {"a": {"scores": [1, 0]}})
        with self.assertRaises(ValueError):  # wrong score length
            metrics.score_task("solubility", truth, {k: {"scores": [1, 0, 0]} for k in truth})
        with self.assertRaises(ValueError):  # negative scores
            metrics.score_task("solubility", truth, {k: {"scores": [-1, 2]} for k in truth})
        with self.assertRaises(ValueError):  # non-finite regression
            metrics.score_task("fluorescence", {"a": 1.0, "b": 2.0},
                               {"a": {"value": float("nan")}, "b": {"value": 0.0}})
        with self.assertRaises(ValueError):  # ss3 length mismatch
            metrics.score_task("secondary_structure",
                               {"a": {"ss3": [0, 1], "valid_mask": [1, 1]}},
                               {"a": {"ss3_scores": [[1, 0, 0]]}})


class GraderTests(unittest.TestCase):
    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))

    def test_grade_task_zeroes_only_bad_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.write_jsonl(tmp / "labels.jsonl",
                             [{"example_id": "a", "label": 0}, {"example_id": "b", "label": 1}])
            self.write_jsonl(tmp / "good.jsonl",
                             [{"example_id": "a", "prediction": {"scores": [1, 0]}},
                              {"example_id": "b", "prediction": {"scores": [0, 1]}}])
            good = grader.grade_task("solubility", tmp / "good.jsonl", tmp / "labels.jsonl", 1.0)
            self.assertEqual(good["reward"], 1.0)
            self.assertIsNone(good["invalid_reason"])
            # duplicate id fails loudly
            self.write_jsonl(tmp / "dup.jsonl",
                             [{"example_id": "a", "prediction": {"scores": [1, 0]}},
                              {"example_id": "a", "prediction": {"scores": [1, 0]}}])
            bad = grader.grade_task("solubility", tmp / "dup.jsonl", tmp / "labels.jsonl", 1.0)
            self.assertEqual(bad["reward"], 0.0)
            self.assertIsNotNone(bad["invalid_reason"])
            # unknown id fails loudly
            self.write_jsonl(tmp / "unknown.jsonl",
                             [{"example_id": "z", "prediction": {"scores": [1, 0]}}])
            bad = grader.grade_task("solubility", tmp / "unknown.jsonl", tmp / "labels.jsonl", 1.0)
            self.assertEqual(bad["reward"], 0.0)
            # missing file fails loudly
            bad = grader.grade_task("solubility", tmp / "absent.jsonl", tmp / "labels.jsonl", 1.0)
            self.assertEqual(bad["reward"], 0.0)

    def test_flatten_metrics(self):
        flat = {}
        metrics.flatten_metrics("solubility", {"accuracy": 0.5, "invalid_reason": None,
                                               "nested": {"x": 1.0}}, flat)
        self.assertEqual(flat, {"solubility_accuracy": 0.5, "solubility_nested_x": 1.0})

    def test_metrics_copies_match(self):
        """Development self-scoring must use the same code as the grader."""
        here = Path(__file__).resolve().parent
        sibling = here.parent / "environment" / "evaluation" / "metrics.py"
        if not sibling.is_file():
            self.skipTest("running inside the verifier image; sibling copy absent")
        self.assertEqual((here / "metrics.py").read_bytes(), sibling.read_bytes(),
                         "tests/metrics.py and environment/evaluation/metrics.py have diverged")


if __name__ == "__main__":
    unittest.main()
