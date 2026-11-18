#!/usr/bin/env python3
"""CPU-runnable V4 unit & integration tests for #15 protein-sequence-modeling.

Covers:
1. Asymmetric order-bias dummy `p = sigmoid(len(seq_b) - len(seq_a))` on length-biased
   PPI pairs (`len(seq_b) > len(seq_a)` by ~100 aa on positives): achieves single-pass
   `ab_balanced_accuracy == 1.0`, yet collapses to `p_sym == 0.5`, `balanced_accuracy == 0.5`,
   and `skill == 0.0` under symmetric `(seq_a, seq_b)` / `(seq_b, seq_a)` averaging while
   retaining `p_ab`, `p_ba`, and `asymmetry_mae` in `report.json`.
2. End-to-end two-pass `main_verifier.py` -> `grader.py` CPU replay for both a legitimate
   sequence-content symmetric model and the order-bias length shortcut model.
3. `< 1e-12` parity between `grader.py` / `metrics.aggregate_pillars` and
   `harness/ssi_index_v3.py:sequence()` across normal, fold `K=136` observed-class chance,
   negative signed-mean, and partial invalid-task fixtures.
4. Byte-identical `metrics.py` copies, strict visibility boundary (`tests/` never in dev
   `/workspace`), and 3-part Literature Justification in `tests/README.md`.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

TESTS_DIR = Path(__file__).resolve().parent
TASK_DIR = TESTS_DIR.parent
REPO_ROOT = TASK_DIR.parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import common  # noqa: E402
import grader  # noqa: E402
import main_verifier  # noqa: E402
import metrics  # noqa: E402


def _load_v3_index():
    spec = importlib.util.spec_from_file_location(
        "ssi_index_v3", next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as sink:
        for row in rows:
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")


def _build_synthetic_benchmark(root: Path) -> tuple[Path, Path]:
    """Create a complete 11-task verifier + private-label fixture."""
    pub = root / "public" / "verifier"
    priv = root / "private"
    pub.mkdir(parents=True)
    priv.mkdir(parents=True)

    pub_files: dict[str, str] = {}
    priv_files: dict[str, str] = {}
    counts: dict[str, int] = {}

    for task in metrics.TASKS:
        pub_rows = []
        priv_rows = []
        if task in metrics.PPI_SYMMETRIC_TASKS:
            # 8 positive pairs where len(seq_b) > len(seq_a) by 100 aa ("W"-rich),
            # 8 negative pairs where len(seq_b) < len(seq_a) by 100 aa ("G"-rich).
            for idx in range(16):
                label = 1 if idx < 8 else 0
                aa = "W" if label == 1 else "G"
                if label == 1:
                    seq_a = aa * (50 + idx)
                    seq_b = aa * (150 + idx)
                else:
                    seq_a = aa * (160 + idx)
                    seq_b = aa * (60 + idx)
                eid = f"{task}_{idx:03d}"
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence_a": seq_a, "sequence_b": seq_b},
                })
                priv_rows.append({"example_id": eid, "label": label})
        elif task == "ppi_affinity":
            for idx in range(8):
                sign = 1.0 if idx % 2 == 0 else -1.0
                val = 6.0 + sign * metrics.PPI_AFFINITY_ANCHOR_RMSE
                aa = "W" if idx % 2 == 0 else "G"
                seq_a = aa * 20 + "A" * 10
                seq_b = aa * 20 + "L" * 10
                eid = f"{task}_{idx:03d}"
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence_a": seq_a, "sequence_b": seq_b},
                })
                priv_rows.append({"example_id": eid, "label": val})
        elif task == "fold":
            # Populate exactly 136 distinct observed classes in [0, 1194] to match SEQ_K["fold"] = 136!
            for idx in range(136):
                cls_id = (idx * 8) % 1195
                seq = "ACDEFGHIKLMNPQRSTVWY"[idx % 20] * (30 + (idx % 15))
                eid = f"{task}_{idx:04d}"
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence": seq},
                })
                priv_rows.append({"example_id": eid, "label": cls_id})
        elif task in metrics.CLASSES:
            n_cls = metrics.CLASSES[task]
            for idx in range(max(12, n_cls * 2)):
                cls_id = idx % n_cls
                aa = "ACDEFGHIKLMNPQRSTVWY"[cls_id % 20]
                eid = f"{task}_{idx:03d}"
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence": aa * (25 + idx)},
                })
                priv_rows.append({"example_id": eid, "label": cls_id})
        elif task in metrics.SPEARMAN_TASKS:
            for idx in range(10):
                eid = f"{task}_{idx:03d}"
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence": "W" * (idx + 1) + "A" * 20},
                })
                priv_rows.append({"example_id": eid, "label": float(idx)})
        elif task == "secondary_structure":
            for idx in range(6):
                eid = f"{task}_{idx:03d}"
                seq = "H" * 5 + "E" * 5 + "C" * 5
                ss3 = [0] * 5 + [1] * 5 + [2] * 5
                mask = [1] * 14 + [0]
                pub_rows.append({
                    "example_id": eid,
                    "task": task,
                    "features": {"sequence": seq},
                })
                priv_rows.append({
                    "example_id": eid,
                    "label": {"ss3": ss3, "valid_mask": mask},
                })
        else:
            raise AssertionError(f"Unhandled task {task}")

        pub_path = pub / f"{task}.jsonl"
        priv_path = priv / f"{task}.jsonl"
        _write_jsonl(pub_path, pub_rows)
        _write_jsonl(priv_path, priv_rows)
        pub_files[pub_path.name] = common.sha256(pub_path)
        priv_files[priv_path.name] = common.sha256(priv_path)
        counts[task] = len(pub_rows)

    common.write_json(
        pub / "manifest.json",
        {
            "schema": common.SCHEMA,
            "view": "verifier",
            "files": pub_files,
            "counts": counts,
        },
    )
    common.write_json(
        priv / "manifest.json",
        {
            "schema": common.PRIVATE_SCHEMA,
            "public_input_manifest_sha256": common.sha256(pub / "manifest.json"),
            "files": priv_files,
            "fit_standard_deviation": {"ppi_affinity": 2.5},
        },
    )
    return pub, priv


def _write_submission_package(pkg_dir: Path, mode: str) -> None:
    """Write a synthetic submission package (`order_bias` or `symmetric_content`)."""
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "README.md").write_text("# Synthetic Test Submission\n")
    predict_code = f"""#!/usr/bin/env python3
import json, math, sys
from pathlib import Path

MODE = {mode!r}
TASKS = ("fluorescence", "stability", "beta_lactamase", "solubility",
         "subcellular_localization", "binary_localization", "fold",
         "secondary_structure", "yeast_ppi", "human_ppi", "ppi_affinity")
CLASSES = {{"solubility": 2, "subcellular_localization": 10, "binary_localization": 2,
           "fold": 1195, "yeast_ppi": 2, "human_ppi": 2}}

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

inp, out = Path(sys.argv[1]), Path(sys.argv[2]) / "predictions"
out.mkdir(parents=True, exist_ok=False)

for task in TASKS:
    rows = [json.loads(line) for line in (inp / f"{{task}}.jsonl").open()]
    with (out / f"{{task}}.jsonl").open("w") as sink:
        for row in rows:
            feat = row["features"]
            if task in ("yeast_ppi", "human_ppi"):
                sa, sb = feat["sequence_a"], feat["sequence_b"]
                if MODE == "order_bias":
                    # Pure antisymmetric length-order shortcut: p(A,B) = sigmoid((|B|-|A|)/20)
                    p1 = sigmoid((len(sb) - len(sa)) / 20.0)
                else:
                    # Symmetric sequence-content predictor: fraction of 'W' across (A, B)
                    w_frac = (sa.count("W") + sb.count("W")) / max(len(sa) + len(sb), 1)
                    p1 = 0.9 if w_frac > 0.5 else 0.1
                pred = {{"scores": [1.0 - p1, p1]}}
            elif task in CLASSES:
                n = CLASSES[task]
                if MODE == "order_bias":
                    pred = {{"scores": [1.0] * n}}
                else:
                    seq = feat["sequence"]
                    aa_idx = "ACDEFGHIKLMNPQRSTVWY".find(seq[0])
                    scores = [1.0] * n
                    if task == "fold":
                        # Hit a valid observed class when possible
                        guess = ((len(seq) - 30) * 8) % n
                        scores[guess] = 10.0
                    elif aa_idx >= 0 and aa_idx < n:
                        scores[aa_idx] = 10.0
                    pred = {{"scores": scores}}
            elif task == "ppi_affinity":
                sa, sb = feat["sequence_a"], feat["sequence_b"]
                val = (6.0 + 3.083206134 if sa.count("W") > 10 else 6.0 - 3.083206134) if MODE != "order_bias" else 6.0
                pred = {{"value": float(val)}}
            elif task == "secondary_structure":
                seq = feat["sequence"]
                if MODE == "order_bias":
                    pred = {{"ss3_scores": [[0.0, 0.0, 1.0] for _ in seq]}}
                else:
                    mapping = {{"H": [10.0, 1.0, 1.0], "E": [1.0, 10.0, 1.0], "C": [1.0, 1.0, 10.0]}}
                    pred = {{"ss3_scores": [mapping.get(ch, [1.0, 1.0, 1.0]) for ch in seq]}}
            else:
                seq = feat["sequence"]
                val = float(seq.count("W")) if MODE != "order_bias" else 0.0
                pred = {{"value": val}}
            sink.write(json.dumps({{"example_id": row["example_id"], "prediction": pred}}, separators=(",", ":")) + "\\n")
"""
    (pkg_dir / "predict.py").write_text(predict_code)
    infer_sh = pkg_dir / "infer.sh"
    infer_sh.write_text(
        f'#!/usr/bin/env bash\nset -euo pipefail\nexec "{sys.executable}" "$(dirname "$0")/predict.py" "$1" "$2"\n'
    )
    infer_sh.chmod(0o755)


def _cpu_run_submission_unprivileged(submission, input_root, work, timeout):
    """Host CPU substitute for root-only `run_submission` in unit tests."""
    work.mkdir(parents=True, exist_ok=True)
    output = work / "untrusted-output"
    output.mkdir()
    agent_home = work / "agent-home"
    agent_tmp = work / "agent-tmp"
    agent_home.mkdir(mode=0o700)
    agent_tmp.mkdir(mode=0o700)
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(agent_home),
        "TMPDIR": str(agent_tmp),
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with (work / "inference.log").open("wb") as log:
        subprocess.run(
            ["bash", str(submission / "infer.sh"), str(input_root), str(output)],
            cwd=submission,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=True,
        )
    return output


class V4ProteinSequenceTests(unittest.TestCase):
    def test_order_bias_dummy_collapses_to_chance_under_symmetric_ppi(self) -> None:
        # Construct 20 pairs where positives ALWAYS have len(seq_b) = len(seq_a) + 100
        # and negatives ALWAYS have len(seq_b) = len(seq_a) - 100.
        truth = {}
        found_ab = {}
        found_ba = {}
        for idx in range(20):
            eid = f"pair_{idx:02d}"
            label = 1 if idx < 10 else 0
            len_a, len_b = (80 + idx, 180 + idx) if label == 1 else (180 + idx, 80 + idx)
            truth[eid] = label
            p_ab = 1.0 / (1.0 + math.exp(-(len_b - len_a)))
            p_ba = 1.0 / (1.0 + math.exp(-(len_a - len_b)))
            found_ab[eid] = {"scores": [1.0 - p_ab, p_ab]}
            found_ba[eid] = {"scores": [1.0 - p_ba, p_ba]}

        single_pass = metrics.score_task("human_ppi", truth, found_ab)
        self.assertAlmostEqual(single_pass["balanced_accuracy"], 1.0, places=12)
        self.assertAlmostEqual(single_pass["skill"], 1.0, places=12)

        sym_res = metrics.score_task(
            "human_ppi", truth, found_ab, found_swapped=found_ba
        )
        self.assertTrue(sym_res["symmetric_ppi"])
        self.assertAlmostEqual(sym_res["ab_balanced_accuracy"], 1.0, places=12)
        self.assertAlmostEqual(sym_res["ba_balanced_accuracy"], 0.0, places=12)
        self.assertAlmostEqual(sym_res["p_sym_mean"], 0.5, places=12)
        self.assertAlmostEqual(sym_res["balanced_accuracy"], 0.5, places=12)
        self.assertAlmostEqual(sym_res["skill"], 0.0, places=12)
        self.assertAlmostEqual(sym_res["reward"], 0.0, places=12)
        self.assertGreater(sym_res["asymmetry_mae"], 0.99)
        self.assertEqual(len(sym_res["pair_probabilities"]), 20)
        for info in sym_res["pair_probabilities"].values():
            self.assertAlmostEqual(info["p_sym"], 0.5, places=12)

    def test_v3_four_pillar_parity_across_normal_negative_and_fold_136_fixtures(self) -> None:
        v3 = _load_v3_index()

        # 1. Normal fixture: verify < 1e-12 parity on score, signed_score, and all 4 pillars
        synthetic_metrics = {
            "fluorescence": {"spearman": 0.68},
            "stability": {"spearman": 0.74},
            "beta_lactamase": {"spearman": 0.81},
            "solubility": {"balanced_accuracy": 0.71},
            "subcellular_localization": {"balanced_accuracy": 0.69},
            "binary_localization": {"balanced_accuracy": 0.91},
            "fold": {"balanced_accuracy": 0.24},
            "secondary_structure": {"accuracy": 0.73},
            "yeast_ppi": {"balanced_accuracy": 0.66},
            "human_ppi": {"balanced_accuracy": 0.77},
            "ppi_affinity": {"rmse": 2.15},
        }
        details = {}
        for task, m in synthetic_metrics.items():
            sk = metrics.task_skill(task, m)
            details[task] = {**m, "skill": sk, "reward": sk, "invalid_reason": None}
        agg = metrics.aggregate_pillars(details)
        ref = v3.sequence(synthetic_metrics)
        self.assertLess(abs(agg["score"] - ref["score"]), 1e-12)
        self.assertLess(abs(agg["signed_score"] - ref["signed_score"]), 1e-12)
        for pillar, val in ref["components"].items():
            self.assertLess(abs(agg["pillars"][pillar] - val), 1e-12)

        # 2. Negative signed-mean fixture: signed_score < 0 clips to 0.0 at top level
        neg_metrics = {
            "fluorescence": {"spearman": -0.40},
            "stability": {"spearman": -0.35},
            "beta_lactamase": {"spearman": -0.25},
            "solubility": {"balanced_accuracy": 0.45},
            "subcellular_localization": {"balanced_accuracy": 0.08},
            "binary_localization": {"balanced_accuracy": 0.44},
            "fold": {"balanced_accuracy": 0.002},
            "secondary_structure": {"accuracy": 0.35},
            "yeast_ppi": {"balanced_accuracy": 0.48},
            "human_ppi": {"balanced_accuracy": 0.49},
            "ppi_affinity": {"rmse": 4.20},
        }
        neg_details = {
            t: {**m, "skill": metrics.task_skill(t, m), "reward": metrics.task_skill(t, m), "invalid_reason": None}
            for t, m in neg_metrics.items()
        }
        neg_agg = metrics.aggregate_pillars(neg_details)
        neg_ref = v3.sequence(neg_metrics)
        self.assertLess(neg_agg["signed_score"], 0.0)
        self.assertEqual(neg_agg["score"], 0.0)
        self.assertLess(abs(neg_agg["score"] - neg_ref["score"]), 1e-12)
        self.assertLess(abs(neg_agg["signed_score"] - neg_ref["signed_score"]), 1e-12)

        # 3. Fold K = 136 observed-class uniform/constant chance calibration check
        fold_truth = {f"f_{i}": (i * 7) % 1195 for i in range(136)}
        self.assertEqual(len(set(fold_truth.values())), 136)
        fold_uniform = {k: {"scores": [1.0] * 1195} for k in fold_truth}
        fold_res = metrics.score_task("fold", fold_truth, fold_uniform)
        self.assertAlmostEqual(fold_res["balanced_accuracy"], 1.0 / 136.0, places=12)
        self.assertAlmostEqual(fold_res["skill"], 0.0, places=12)

        # 4. Partial invalid-task fixture zeros only the malformed task
        partial_details = dict(details)
        partial_details["solubility"] = {"reward": 0.0, "skill": 0.0, "invalid_reason": "missing file"}
        partial_agg = metrics.aggregate_pillars(partial_details)
        self.assertEqual(partial_agg["task_skills"]["solubility"], 0.0)
        self.assertGreater(partial_agg["score"], 0.0)

    def test_end_to_end_two_pass_main_verifier_and_grader_on_cpu(self) -> None:
        v3 = _load_v3_index()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            try:
                pub, priv = _build_synthetic_benchmark(root)

                for mode in ("order_bias", "symmetric_content"):
                    run_dir = root / f"run_{mode}"
                    pkg_dir = run_dir / "submission"
                    _write_submission_package(pkg_dir, mode)
                    sealed = run_dir / "sealed" / "run"
                    verifier_out = run_dir / "verifier_out"
                    grader_out = run_dir / "sealed" / "results"

                    with mock.patch.object(
                        main_verifier, "run_submission", side_effect=_cpu_run_submission_unprivileged
                    ), mock.patch.object(
                        sys,
                        "argv",
                        [
                            "main_verifier.py",
                            "--submission",
                            str(pkg_dir),
                            "--input",
                            str(pub),
                            "--sealed",
                            str(sealed),
                            "--output",
                            str(verifier_out),
                            "--timeout",
                            "60",
                            "--no-wait",
                        ],
                    ):
                        main_verifier.main()

                    self.assertTrue((sealed / "snapshot.json").is_file())
                    self.assertTrue((sealed / "yeast_ppi_swapped.jsonl").is_file())
                    self.assertTrue((sealed / "human_ppi_swapped.jsonl").is_file())

                    with mock.patch.object(
                        sys,
                        "argv",
                        [
                            "grader.py",
                            "--sealed",
                            str(sealed),
                            "--labels",
                            str(priv),
                            "--output",
                            str(grader_out),
                            "--wait-seconds",
                            "5",
                        ],
                    ):
                        grader.main()

                    report = json.loads((grader_out / "report.json").read_text())
                    reward = json.loads((grader_out / "reward.json").read_text())
                    self.assertIn("score", reward)
                    self.assertIn("reward", reward)
                    self.assertEqual(reward["score"], reward["reward"])
                    self.assertEqual(report["score"], reward["score"])

                    # Exact < 1e-12 parity with harness/ssi_index_v3.py:sequence()
                    v3_res = v3.sequence(report["tasks"])
                    self.assertLess(abs(reward["score"] - v3_res["score"]), 1e-12)
                    self.assertLess(abs(report["signed_score"] - v3_res["signed_score"]), 1e-12)

                    if mode == "order_bias":
                        for ppi_task in ("yeast_ppi", "human_ppi"):
                            t_rep = report["tasks"][ppi_task]
                            self.assertAlmostEqual(t_rep["ab_balanced_accuracy"], 1.0, places=12)
                            self.assertAlmostEqual(t_rep["balanced_accuracy"], 0.5, places=12)
                            self.assertAlmostEqual(t_rep["skill"], 0.0, places=12)
                            self.assertGreater(t_rep["asymmetry_mae"], 0.9)
                        # Constant/order-bias dummy collapses Protein interactions pillar & overall score to 0.0
                        self.assertAlmostEqual(report["pillars"]["Protein interactions"], 0.0, places=12)
                        self.assertAlmostEqual(reward["score"], 0.0, places=12)
                    else:
                        for ppi_task in ("yeast_ppi", "human_ppi"):
                            t_rep = report["tasks"][ppi_task]
                            self.assertAlmostEqual(t_rep["balanced_accuracy"], 1.0, places=12)
                            self.assertAlmostEqual(t_rep["skill"], 1.0, places=12)
                            self.assertAlmostEqual(t_rep["asymmetry_mae"], 0.0, places=12)
                        self.assertGreater(reward["score"], 0.85)

                    # Also verify development self-evaluation `score.py` matches `grader.py` to < 1e-12
                    dev_out = run_dir / "dev_score.json"
                    subprocess.run(
                        [
                            sys.executable,
                            str(TASK_DIR / "environment" / "evaluation" / "score.py"),
                            "--predictions",
                            str(sealed),
                            "--labels",
                            str(priv),
                            "--output",
                            str(dev_out),
                        ],
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    dev_report = json.loads(dev_out.read_text())
                    self.assertLess(abs(dev_report["score"] - reward["score"]), 1e-12)
                    self.assertLess(abs(dev_report["signed_score"] - report["signed_score"]), 1e-12)
            finally:
                for p in sorted(root.rglob("*"), reverse=True):
                    if p.is_dir() and not p.is_symlink():
                        p.chmod(0o700)
                    elif p.is_file() and not p.is_symlink():
                        p.chmod(0o600)

    def test_visibility_boundary_and_literature_justification(self) -> None:
        self.assertEqual(
            (TASK_DIR / "tests" / "metrics.py").read_bytes(),
            (TASK_DIR / "environment" / "evaluation" / "metrics.py").read_bytes(),
        )
        dev_dockerfile = (TASK_DIR / "environment" / "Dockerfile").read_text()
        dev_compose = (TASK_DIR / "environment" / "docker-compose.yaml").read_text()
        self.assertNotIn("COPY tests", dev_dockerfile)
        self.assertNotIn("/tests", dev_dockerfile)
        self.assertNotIn("tests/", dev_compose)
        self.assertNotIn("TASK_PRIVATE_DIR", dev_compose)

        readme = (TASK_DIR / "tests" / "README.md").read_text()
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, readme)
        self.assertIn("PEER", readme)
        self.assertIn("Park & Marcotte", readme)


if __name__ == "__main__":
    unittest.main()
