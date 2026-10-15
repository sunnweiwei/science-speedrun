#!/usr/bin/env python3
"""Unit tests for the task layout, the metrics, the package checks and the sealed-prediction grader.
Standard library only (scipy is used when present). Run: bash tests/run_unit_tests.sh"""
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
TESTS = TASK / "tests"
sys.path.insert(0, str(TESTS)); sys.path.insert(0, str(TESTS / "protocol"))
import grader  # noqa: E402
import main_verifier  # noqa: E402
from tasks import GROUPS, NUM_TARGETS, TASKS, group_summary, target_metrics, task_metrics  # noqa: E402


class Layout(unittest.TestCase):
    def test_files(self):
        for rel in ["README.md", "task.toml", "instruction.md", "environment/Dockerfile", "environment/docker-compose.yaml",
                    "environment/contract.md", "environment/data/README.md", "environment/data/MATERIALIZE.md",
                    "environment/data/materialize.sh", "environment/data/CHECKSUMS.json",
                    "environment/data/build/SOURCES.sha256", "environment/baseline/unimol2-ref/README.md",
                    "environment/baseline/unimol2-ref/RESULTS.md", "environment/baseline/unimol2-ref/pretrain.sh",
                    "environment/baseline/unimol2-ref/publish.py", "environment/references/README.md",
                    "environment/evaluation/README.md", "environment/evaluation/dev_eval.py",
                    "environment/evaluation/check_submission.py", "tests/Dockerfile", "tests/docker-compose.yaml",
                    "tests/test.sh", "tests/main_verifier.py", "tests/grader.py", "docs/VALIDATION.md"]:
            self.assertTrue((TASK / rel).is_file(), rel)

    def test_instruction_sections_and_no_budget(self):
        text = (TASK / "instruction.md").read_text()
        heads = re.findall(r"^## (.+)$", text, re.M)
        self.assertEqual(heads, ["Task and goal", "Workspace", "Rules", "Submission", "Continuous research"])
        for word in ("H100", "GPU-hour", "hours", "48 h", "12 h", "deadline"):
            self.assertNotIn(word, text)

    def test_task_toml(self):
        t = tomllib.loads((TASK / "task.toml").read_text())
        self.assertEqual(t["artifacts"], ["/workspace/submission"])
        self.assertEqual(t["environment"]["network_mode"], "no-network")
        self.assertEqual(t["verifier"]["environment_mode"], "separate")
        self.assertEqual(t["verifier"]["environment"]["network_mode"], "no-network")
        self.assertTrue(t["task"]["authors"])

    def test_dev_compose_mounts_only_public(self):
        text = (TASK / "environment/docker-compose.yaml").read_text()
        body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("network_mode", body)
        self.assertIn("${TASK_RESEARCH_DIR", body)
        self.assertIn("/public/pretrain", body); self.assertIn("/public/downstream", body)
        for secret in ("verifier/", "private", "TASK_PRIVATE_DIR", "labels"):
            self.assertNotIn(secret, body)

    def test_verifier_compose_split(self):
        text = (TASK / "tests/docker-compose.yaml").read_text()
        main, grader_part = text.split("\n  grader:")
        self.assertNotIn("private/labels", main)
        self.assertIn("private/labels", grader_part)
        self.assertNotIn("gpus", grader_part)
        self.assertEqual(text.count("network_mode: none"), 2)

    def test_shared_runtime_identical(self):
        def block(p):
            s = (TASK / p).read_text()
            return s[s.index("# BEGIN SHARED RUNTIME"):s.index("# END SHARED RUNTIME")]
        self.assertEqual(block("environment/Dockerfile"), block("tests/Dockerfile"))

    def test_protocol_copies_identical(self):
        a, b = TASK / "environment/evaluation/protocol", TESTS / "protocol"
        names = lambda r: sorted(p.relative_to(r).as_posix() for p in r.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(names(a), names(b))
        for n in names(a):
            self.assertEqual((a / n).read_bytes(), (b / n).read_bytes(), n)

    def test_checksums_cover_views(self):
        table = json.loads((TASK / "environment/data/CHECKSUMS.json").read_text())
        for t in TASKS:
            for rel in (f"public/downstream/{t}/train.lmdb", f"public/downstream/{t}/valid.lmdb",
                        f"verifier/test-inputs/{t}/test.lmdb", f"private/labels/{t}.csv"):
                self.assertIn(rel, table)
        self.assertEqual(sum(k.startswith("public/pretrain/train_part_") for k in table), 12)

    def test_task_registry(self):
        self.assertEqual(len(TASKS), 24)
        self.assertEqual([len(v) for v in GROUPS.values()], [6, 4, 9, 5])
        self.assertEqual(sum(NUM_TARGETS.values()), 26)


class Metrics(unittest.TestCase):
    def test_values(self):
        y = [1.0, 2.0, 3.0, 4.0]; p = [1.5, 1.5, 3.5, 3.0]
        m = target_metrics(p, y)
        self.assertAlmostEqual(m["mae"], 0.625)
        self.assertAlmostEqual(m["rae"], 0.625 / 1.0)
        self.assertAlmostEqual(m["r2"], 1 - 1.75 / 5.0)
        self.assertAlmostEqual(m["spearman"], 3.5 / math.sqrt(22.5), places=9)

    def test_spearman_ties_and_constant(self):
        m = target_metrics([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(m["spearman"], 0.8660254, places=6)
        c = target_metrics([0.5, 0.5, 0.5], [1.0, 2.0, 3.0])
        self.assertEqual((c["spearman"], c["kendall"]), (0.0, 0.0))

    def test_multi_target_average_and_groups(self):
        truth = {"a": [1.0, 10.0], "b": [2.0, 20.0], "c": [3.0, 30.0]}
        pred = {"a": [1.0, 12.0], "b": [2.0, 18.0], "c": [4.0, 30.0]}
        m = task_metrics(pred, truth)
        self.assertAlmostEqual(m["mae"], (1 / 3 + 4 / 3) / 2)
        per = {t: {"rae": 0.5, "spearman": 0.25} for t in TASKS}
        g = group_summary(per)
        self.assertEqual(sorted(g), sorted(f"{x}_{y}" for x in GROUPS for y in ("ma_rae", "mean_spearman")))
        self.assertAlmostEqual(g["expansionrx_ma_rae"], 0.5)

    def test_matches_official_rae_formula(self):
        rng = random.Random(0)
        y = [rng.gauss(0, 1) for _ in range(200)]; p = [v + rng.gauss(0, 0.3) for v in y]
        mae = sum(abs(a - b) for a, b in zip(p, y)) / 200
        ybar = sum(y) / 200
        self.assertAlmostEqual(target_metrics(p, y)["rae"], mae / (sum(abs(v - ybar) for v in y) / 200))


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


class Grading(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.labels = self.tmp / "labels"; self.labels.mkdir()
        self.sealed = self.tmp / "sealed" / "run"; self.sealed.mkdir(parents=True)
        files = {}
        for i, t in enumerate(TASKS):
            k = NUM_TARGETS[t]
            smis = [f"C{'C' * j}" for j in range(5)]
            write_csv(self.labels / f"{t}.csv", ["smi"] + [f"target_{j}" for j in range(k)],
                      [[s] + [float(n + j) for j in range(k)] for n, s in enumerate(smis)])
            write_csv(self.sealed / f"{t}.csv", ["smi"] + [f"predict_{j}" for j in range(k)],
                      [[s] + [float(n + j) + 0.1 * (-1) ** n for j in range(k)] for n, s in enumerate(smis)])
        for name, obj in (("encoder_check.json", {"ok": True}), ("suite.json", {"minutes": 1})):
            (self.sealed / name).write_text(json.dumps(obj))
        for p in self.sealed.iterdir():
            files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        (self.sealed / "snapshot.json").write_text(json.dumps({"schema": "mpt-v2-predictions/v1", "files": files}))
        (self.sealed.parent / "READY").write_text(grader.sha256(self.sealed / "snapshot.json"))

    def test_scores(self):
        grader.validate_snapshot(self.sealed)
        reward, report = grader.score(self.sealed, self.labels)
        self.assertEqual(reward["valid"], 1.0)
        self.assertAlmostEqual(reward["qm9dft_v2_mae"], 0.1)
        self.assertEqual(len([k for k in reward if k.endswith("_rae") and not k.endswith("ma_rae")]), 24)
        self.assertTrue(all(isinstance(v, float) for v in reward.values()))

    def test_tampered_snapshot(self):
        with open(self.sealed / "exrx_logd.csv", "a") as f:
            f.write("CCCCCC,1.0\n")
        with self.assertRaises(ValueError):
            grader.validate_snapshot(self.sealed)

    def test_missing_molecule(self):
        rows = list(csv.reader(open(self.sealed / "asap_mlm.csv")))[:-1]
        write_csv(self.sealed / "asap_mlm.csv", rows[0], rows[1:])
        with self.assertRaises(ValueError):
            grader.score(self.sealed, self.labels)


class Package(unittest.TestCase):
    def make(self, **files):
        d = Path(tempfile.mkdtemp()) / "submission"; d.mkdir()
        base = {"encoder.json": '{"arch": "unimol2_84M"}', "checkpoint.pt": "x"}
        base.update(files)
        for k, v in base.items():
            if v is not None:
                (d / k).parent.mkdir(parents=True, exist_ok=True); (d / k).write_text(v)
        return d

    def test_valid(self):
        main_verifier.validate_package(self.make())
        main_verifier.validate_package(self.make(**{"encoder.json": '{"arch": "mine", "user_dir": "ud"}', "ud/__init__.py": ""}))

    def test_invalid(self):
        for bad in ({"checkpoint.pt": None}, {"encoder.json": "[1]"}, {"encoder.json": '{"arch": "x", "lr": 1}'},
                    {"extra.bin": "1"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                main_verifier.validate_package(self.make(**bad))
        d = self.make(); os.symlink("/etc/passwd", d / "README.md")
        with self.assertRaises(ValueError):
            main_verifier.validate_package(d)

    def test_prediction_file_checks(self):
        d = Path(tempfile.mkdtemp())
        write_csv(d / "p.csv", ["smi", "predict_0"], [["C", "1.0"], ["CC", "2.0"]])
        self.assertEqual(len(main_verifier.read_predictions(d / "p.csv", "exrx_logd", 2)), 3)
        with self.assertRaises(ValueError):
            main_verifier.read_predictions(d / "p.csv", "exrx_logd", 1)
        with self.assertRaises(ValueError):
            main_verifier.read_predictions(d / "p.csv", "qm9dft_v2", 5)
        write_csv(d / "n.csv", ["smi", "predict_0"], [["C", "nan"]])
        with self.assertRaises(ValueError):
            main_verifier.read_predictions(d / "n.csv", "exrx_logd", 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
