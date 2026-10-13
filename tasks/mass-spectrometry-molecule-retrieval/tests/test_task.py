"""Synthetic materialization, contract, scoring, and layout tests."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unittest
from unittest import mock


TASK_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
DATA_ROOT = TASK_ROOT / "environment/data"
sys.path.insert(0, str(TEST_ROOT))
import common
import grader
import main_verifier


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


materialize = import_path("massspecgym_materialize", DATA_ROOT / "materialize.py")
paired_bootstrap = import_path(
    "massspecgym_paired_bootstrap",
    TASK_ROOT / "environment/evaluation/paired_bootstrap.py",
)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_source(self):
        source = self.root / "source"
        dataset = source / materialize.DATASET_RELATIVE
        candidates = source / materialize.CANDIDATES_RELATIVE
        dataset.parent.mkdir(parents=True, exist_ok=True)
        candidates.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "identifier", "mzs", "intensities", "smiles", "inchikey", "formula",
            "precursor_mz", "adduct", "instrument_type", "collision_energy", "fold",
        ]
        rows = [
            {
                "identifier": "q-train", "mzs": "50,100", "intensities": "0.5,1",
                "smiles": "CCC", "inchikey": "TRAINKEY000001", "formula": "C3H8",
                "precursor_mz": "45", "adduct": "[M+H]+", "instrument_type": "QTOF",
                "collision_energy": "20", "fold": "train",
            },
            {
                "identifier": "q-val", "mzs": "60,110", "intensities": "0.3,1",
                "smiles": "CCN", "inchikey": "VALKEY00000001", "formula": "C2H7N",
                "precursor_mz": "46", "adduct": "[M+H]+", "instrument_type": "Orbitrap",
                "collision_energy": "", "fold": "val",
            },
            {
                "identifier": "q-test", "mzs": "70,120", "intensities": "0.2,1",
                "smiles": "CCO", "inchikey": "TESTKEY0000001", "formula": "C2H6O",
                "precursor_mz": "47", "adduct": "[M+H]+", "instrument_type": "",
                "collision_energy": "30", "fold": "test",
            },
        ]
        with dataset.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        pool = ["CCN", "CCO", "CCC"]
        candidates.write_text(json.dumps({"CCN": pool, "CCO": pool, "CCC": pool}))
        return source, dataset, candidates

    def prepare(self, suffix=""):
        source, dataset, candidates = self.make_source()
        public = self.root / f"public{suffix}"
        private = self.root / f"private{suffix}"
        args = argparse.Namespace(
            source_root=source,
            output_root=public,
            private_label_root=private,
            dataset_sha256=file_sha(dataset),
            candidates_sha256=file_sha(candidates),
        )
        report = materialize.materialize(args)
        return public, private, report

    def test_native_layout_and_declared_budget(self):
        config = tomllib.loads((TASK_ROOT / "task.toml").read_text())
        self.assertEqual(config["artifacts"], ["/workspace/submission"])
        self.assertEqual(config["agent"]["timeout_sec"], 172800.0)
        self.assertEqual(config["environment"]["gpus"], 8)
        self.assertEqual(config["environment"]["gpu_types"], ["H100"])
        self.assertEqual(config["environment"]["memory_mb"], 262144)
        self.assertEqual(config["environment"]["network_mode"], "no-network")
        self.assertEqual(config["verifier"]["environment_mode"], "separate")
        self.assertEqual(
            config["verifier"]["environment"]["network_mode"], "no-network"
        )
        self.assertEqual(config["task"]["authors"], [{"name": "T0-RSI contributors"}])
        self.assertEqual(
            config["metadata"]["reference_hit_rate_at_1"],
            0.4939052175894281,
        )
        for relative in (
            "instruction.md", "docs/baseline_score.json", "docs/compute-profiles.toml",
            "environment/contract.md",
            "environment/Dockerfile", "environment/entrypoint.sh",
            "environment/data/materialize.py", "environment/baseline/starter/train.py",
            "environment/references/README.md",
            "environment/references/reference_baseline.json",
            "tests/test.sh", "tests/run_unit_tests.sh", "tests/grader.py",
        ):
            self.assertTrue((TASK_ROOT / relative).is_file(), relative)

        profiles = tomllib.loads((TASK_ROOT / "docs/compute-profiles.toml").read_text())
        self.assertEqual(profiles["schema_version"], "agent-runner-compute-profiles/v1")
        self.assertEqual(profiles["default"], "default")
        self.assertEqual(profiles["profiles"]["default"]["gpus"], "task")

        reference = json.loads(
            (TASK_ROOT / "environment/references/reference_baseline.json").read_text()
        )
        self.assertEqual(reference["schema"], "massspecgym-retrieval-reference/v4")
        self.assertEqual(
            reference["name"],
            "GLACIER no-CF with strict-clean MIST fallback",
        )
        self.assertEqual(reference["primary_target"], 0.4939052175894281)
        self.assertEqual(
            reference["glacier_hybrid_replay"]["formal_full_test_replay"]
            ["metrics"]["hit_rate_at_1"],
            0.4939052175894281,
        )
        self.assertTrue(
            reference["glacier_hybrid_replay"]["determinism"]["byte_identical"]
        )
        baseline_score = json.loads((TASK_ROOT / "docs" / "baseline_score.json").read_text())
        self.assertEqual(baseline_score["reward"], 1.0)
        self.assertEqual(baseline_score["metrics"]["hit_rate_at_1"], 0.4939052175894281)
        self.assertEqual(
            reference["strict_clean_top5_max_replay"]["metrics"]["hit_rate_at_1"],
            0.39872408293460926,
        )
        reference_inference = (
            TASK_ROOT / "environment/references/scripts/predict_mist.py"
        ).read_text()
        self.assertIn('set_pooling="cls"', reference_inference)
        reference_runner = (
            TASK_ROOT / "environment/references/scripts/run_reference.sh"
        ).read_text()
        self.assertIn('MASS_SPEC_FORMULA_TOP_K:-5', reference_runner)
        self.assertIn('MASS_SPEC_FINGERPRINT_THRESHOLD:-0.20346222817897797',
                      reference_runner)
        self.assertIn('MASS_SPEC_MIST_CHECKPOINT:-', reference_runner)
        self.assertIn('--formula-top-k "$formula_top_k"', reference_runner)
        self.assertIn('--formula-aggregation "$formula_aggregation"', reference_runner)
        self.assertIn('--checkpoint "$mist_checkpoint"', reference_runner)

    def test_materializer_separates_labels_and_is_deterministic(self):
        public, private, report = self.prepare("-a")
        self.assertEqual(report["train_queries"], 1)
        self.assertEqual(report["validation_queries"], 1)
        self.assertEqual(report["test_queries"], 1)
        train = common.jsonl(public / "development/train.jsonl")[0]
        validation = common.jsonl(public / "development/validation.jsonl")[0]
        query = common.jsonl(public / "verifier/queries.jsonl")[0]
        label = common.jsonl(private / "labels.jsonl")[0]
        self.assertEqual(train["target_smiles"], "CCC")
        self.assertEqual(validation["target_smiles"], "CCN")
        self.assertNotIn("target_smiles", query)
        self.assertNotIn("formula", query)
        self.assertNotIn("inchikey", query)
        self.assertEqual(label["correct_candidate_id"], common.candidate_id("CCO"))
        val_pool = common.jsonl(public / "development/validation_candidate_pools.jsonl")[0]
        test_pool = common.jsonl(public / "verifier/candidate_pools.jsonl")[0]
        self.assertEqual(val_pool, test_pool)

        public_b, private_b, _ = self.prepare("-b")
        first = {
            str(path.relative_to(public)): path.read_bytes()
            for path in public.rglob("*") if path.is_file()
        }
        second = {
            str(path.relative_to(public_b)): path.read_bytes()
            for path in public_b.rglob("*") if path.is_file()
        }
        self.assertEqual(first, second)
        self.assertEqual((private / "labels.jsonl").read_bytes(),
                         (private_b / "labels.jsonl").read_bytes())

    def test_materializer_reuses_only_a_valid_complete_pair(self):
        public, private, expected = self.prepare("-reuse")
        source = self.root / "source"
        dataset = source / materialize.DATASET_RELATIVE
        candidates = source / materialize.CANDIDATES_RELATIVE
        args = argparse.Namespace(
            source_root=source,
            output_root=public,
            private_label_root=private,
            dataset_sha256=file_sha(dataset),
            candidates_sha256=file_sha(candidates),
            reuse_valid_existing=True,
        )
        self.assertEqual(materialize.materialize(args), expected)
        (public / "verifier/queries.jsonl").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            materialize.materialize(args)

    def test_input_validation_and_fixed_denominator_scoring(self):
        public, private, _ = self.prepare("-score")
        _, queries, pools = common.validate_input(public / "verifier")
        labels = grader.load_labels(private / "labels.jsonl", set(queries))
        query_id = next(iter(queries))
        pool = pools[queries[query_id]["candidate_pool_id"]]
        correct = labels[query_id]["correct_candidate_id"]
        ranking = [correct] + sorted(set(pool) - {correct})
        metrics = grader.score(
            {query_id: ranking}, queries, pools, labels, compute_mces=False
        )
        self.assertEqual(metrics, {
            "hit_rate_at_1": 1.0,
            "hit_rate_at_5": 1.0,
            "hit_rate_at_20": 1.0,
        })
        missing = grader.score({}, queries, pools, labels, compute_mces=False)
        self.assertEqual(missing["hit_rate_at_1"], 0.0)
        self.assertEqual(missing["hit_rate_at_20"], 0.0)

    def test_trusted_grader_publishes_normalized_reward_and_preserves_raw_metrics(self):
        public, private, _ = self.prepare("-normalized")
        _, queries, pools = common.validate_input(public / "verifier")
        labels = grader.load_labels(private / "labels.jsonl", set(queries))
        predictions = self.root / "predictions"
        predictions.mkdir()
        rows = []
        for query_id, query in queries.items():
            truth = labels[query_id]["correct_candidate_id"]
            candidates = list(pools[query["candidate_pool_id"]])
            rows.append({"query_id": query_id, "ranking": [truth] + [c for c in candidates if c != truth]})
        (predictions / "rankings.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        sealed = self.root / "seal/run"
        sealed.parent.mkdir()
        main_verifier.seal_output(predictions, sealed, public / "verifier/manifest.json")
        args = argparse.Namespace(sealed=sealed, input=public / "verifier", labels=private,
                                  output=self.root / "score", wait_seconds=1, mces_workers=1)
        report = grader.run(args, compute_mces=False)
        self.assertEqual(json.loads((args.output / "reward.json").read_text()), {"reward": 1.0})
        self.assertEqual(report["metrics"]["hit_rate_at_1"], 1.0)
        self.assertGreater(report["benchmark_score"]["unclipped_score"], 1.0)
        self.assertEqual(report["benchmark_score"]["saturation"], "upper")
        self.assertFalse(report["reference_comparison"]["comparable"])

    def test_unknown_or_partial_ranking_is_global_error(self):
        public, _, _ = self.prepare("-invalid")
        _, queries, pools = common.validate_input(public / "verifier")
        path = self.root / "rankings.jsonl"
        path.write_text('{"query_id":"unknown","ranking":[]}\n')
        rankings, error = grader.load_rankings(path, queries, pools)
        self.assertEqual(rankings, {})
        self.assertIn("unknown", error)
        query_id = next(iter(queries))
        path.write_text(json.dumps({"query_id": query_id, "ranking": []}) + "\n")
        rankings, error = grader.load_rankings(path, queries, pools)
        self.assertEqual(rankings, {})
        self.assertIn("exact candidate permutation", error)

    def test_mces_deduplicates_structure_pairs_without_changing_weights(self):
        queries = {
            "q1": {"candidate_pool_id": "pool"},
            "q2": {"candidate_pool_id": "pool"},
            "q3": {"candidate_pool_id": "pool"},
        }
        pools = {"pool": {"predicted": "CC", "correct": "CCC"}}
        labels = {
            query_id: {
                "correct_candidate_id": "correct",
                "true_smiles": "CCC",
            }
            for query_id in queries
        }
        rankings = {
            "q1": ["predicted", "correct"],
            "q2": ["predicted", "correct"],
        }

        class ImmediateExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def map(self, function, values, chunksize):
                return map(function, values)

        with (
            mock.patch.object(grader, "ProcessPoolExecutor", ImmediateExecutor),
            mock.patch.object(grader, "_mces", return_value=7.0) as compute,
        ):
            metrics = grader.score(
                rankings, queries, pools, labels, compute_mces=True, workers=2
            )
        compute.assert_called_once_with(("CCC", "CC"))
        self.assertEqual(metrics["mces_at_1"], (7.0 + 7.0 + 100.0) / 3)

    def test_paired_bootstrap_uses_query_level_outcomes(self):
        truth = {"q1": "a", "q2": "a", "q3": "a", "q4": "a"}
        baseline = {"q1": "a", "q2": "b", "q3": "a", "q4": "b"}
        challenger = {"q1": "b", "q2": "a", "q3": "a", "q4": "a"}
        report = paired_bootstrap.compare(
            truth, baseline, challenger,
            confidence=0.9, resamples=10_000, seed=7,
        )
        self.assertEqual(report["baseline_hit_rate"], 0.5)
        self.assertEqual(report["challenger_hit_rate"], 0.75)
        self.assertEqual(report["delta"], 0.25)
        self.assertEqual(report["paired_outcomes"], {
            "challenger_only": 2,
            "baseline_only": 1,
            "tie": 1,
        })

    def test_submission_manifest_is_relocation_safe_and_checksum_bound(self):
        package = self.root / "submission"
        package.mkdir()
        (package / "infer.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
        (package / "README.md").write_text("# Synthetic package\n")
        (package / "model.pt").write_bytes(b"checkpoint")
        (package / "model_config.json").write_text("{}\n")
        manifest = {
            "schema": "massspecgym-retrieval-model/v1",
            "checkpoint": {
                "path": "model.pt",
                "sha256": common.sha256(package / "model.pt"),
                "format": "pytorch",
            },
            "model_config": "model_config.json",
            "training_provenance": {
                "data": "synthetic",
                "pretrained_assets": [],
            },
        }
        common.write_json(package / "model_manifest.json", manifest)
        self.assertEqual(main_verifier.validate_submission(package), manifest)
        (package / "README.md").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "README.md"):
            main_verifier.validate_submission(package)
        (package / "README.md").write_text("# Synthetic package\n")
        (package / "model.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            main_verifier.validate_submission(package)

    def test_python_sources_compile_without_importing_heavy_dependencies(self):
        for path in TASK_ROOT.rglob("*.py"):
            py_compile.compile(str(path), doraise=True)

    def test_reference_patches_are_well_formed(self):
        patches = TASK_ROOT / "environment/references/patches"
        for path in sorted(patches.glob("*.patch")):
            subprocess.run(
                ["git", "apply", "--stat", str(path)],
                check=True,
                capture_output=True,
                text=True,
            )
        lock = json.loads(
            (TASK_ROOT / "environment/references/environment-lock.json").read_text()
        )
        asset_helper = (
            TASK_ROOT / "environment/references/scripts/prepare_reference_assets.sh"
        ).read_text()
        for relative, digest in lock["patched_source_sha256"].items():
            self.assertIn(relative.split("/", 1)[1], asset_helper)
            self.assertIn(digest, asset_helper)

    def test_reference_formula_loader_supports_published_top_five(self):
        module = import_path(
            "massspecgym_predict_mist",
            TASK_ROOT / "environment/references/scripts/predict_mist.py",
        )
        predictions = self.root / "formulae.jsonl"
        predictions.write_text(json.dumps({
            "identifier": "query",
            "formulas": ["C", "CH2", "CH4", "C2H2", "C2H4"],
        }) + "\n")
        self.assertEqual(module.load_formulas([predictions], 1), {"query": ["C"]})
        self.assertEqual(
            module.load_formulas([predictions], 5),
            {"query": ["C", "CH2", "CH4", "C2H2", "C2H4"]},
        )
        conditioned = {
            "query": [
                (1, module.np.asarray([0.2, 0.8], dtype=module.np.float32)),
                (0, module.np.asarray([0.9, 0.1], dtype=module.np.float32)),
            ]
        }
        maximum = module.aggregate_conditioned_fingerprints(conditioned, "max")
        mean = module.aggregate_conditioned_fingerprints(conditioned, "mean")
        self.assertEqual(maximum["query"].tolist(), [0.8999999761581421, 0.800000011920929])
        self.assertEqual(mean["query"].tolist(), [0.550000011920929, 0.45000001788139343])

    def test_contract_requires_full_candidate_permutation(self):
        contract = (TASK_ROOT / "environment/contract.md").read_text()
        self.assertIn("contain every candidate ID", contract)
        self.assertIn("Missing query rankings", contract)
        self.assertIn("MCES distance 100", contract)
        compose = (TASK_ROOT / "tests/docker-compose.yaml").read_text()
        main_section, grader_section = compose.split("  grader:", 1)
        self.assertNotIn("MASSSPECGYM_SUBMISSION_DIR", main_section)
        self.assertIn("/workspace/submission", (TASK_ROOT / "tests/test.sh").read_text())
        self.assertNotIn("/workspace/submission", grader_section)
        self.assertIn("network_mode: none", grader_section)

    def test_candidate_order_control_is_a_complete_permutation(self):
        public, _, _ = self.prepare("-order-control")
        output = self.root / "order-control.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/make_candidate_order_control.py"
                ),
                "--queries",
                str(public / "verifier/queries.jsonl"),
                "--candidate-pools",
                str(public / "verifier/candidate_pools.jsonl"),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        row = common.jsonl(output)[0]
        query = common.jsonl(public / "verifier/queries.jsonl")[0]
        pool = common.jsonl(public / "verifier/candidate_pools.jsonl")[0]
        self.assertEqual(row["query_id"], query["query_id"])
        self.assertEqual(
            row["ranking"],
            [candidate["candidate_id"] for candidate in pool["candidates"]],
        )

    def test_glacier_adapters_preserve_the_contract_and_published_domain(self):
        public, private, _ = self.prepare("-glacier")
        development = public / "development"
        query_ids = self.root / "validation-query-ids.txt"
        query_ids.write_text("q-val\n")
        staged = self.root / "staged-validation"
        subprocess.run(
            [
                sys.executable,
                str(TASK_ROOT / "environment/evaluation/stage_validation.py"),
                "--input-root",
                str(development),
                "--output-root",
                str(staged),
                "--query-id-file",
                str(query_ids),
            ],
            check=True,
            capture_output=True,
        )
        staged_query = common.jsonl(staged / "queries.jsonl")[0]
        self.assertEqual(staged_query["query_id"], "q-val")
        self.assertNotIn("target_smiles", staged_query)

        glacier_input = self.root / "glacier-input"
        glacier_input.mkdir()
        query = common.jsonl(public / "verifier/queries.jsonl")[0]
        query["instrument_type"] = "Orbitrap"
        common.write_json(glacier_input / "queries.jsonl", query)
        (glacier_input / "candidate_pools.jsonl").write_bytes(
            (public / "verifier/candidate_pools.jsonl").read_bytes()
        )
        candidate_table = self.root / "glacier-candidates.tsv"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/prepare_glacier_candidates.py"
                ),
                "--input-root",
                str(glacier_input),
                "--output",
                str(candidate_table),
                "--eligible-only",
            ],
            check=True,
            capture_output=True,
        )
        with candidate_table.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["spec"] for row in rows}, {"q-test"})
        self.assertFalse(any("target" in key for key in rows[0]))

        ranking = [row["candidate_id"] for row in rows]
        fallback = self.root / "fallback.jsonl"
        glacier = self.root / "glacier.jsonl"
        common.write_json(
            fallback, {"query_id": "q-test", "ranking": ranking[::-1]}
        )
        common.write_json(glacier, {"query_id": "q-test", "ranking": ranking})
        merged = self.root / "merged"
        merged.mkdir()
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/merge_glacier_fallback.py"
                ),
                "--input-root",
                str(glacier_input),
                "--glacier",
                str(glacier),
                "--fallback",
                str(fallback),
                "--output-root",
                str(merged),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(common.jsonl(merged / "rankings.jsonl")[0]["ranking"], ranking)

        protocol = subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/audit_glacier_protocol.py"
                ),
                "--labels",
                str(private / "labels.jsonl"),
                "--queries",
                str(glacier_input / "queries.jsonl"),
                "--candidate-pools",
                str(glacier_input / "candidate_pools.jsonl"),
                "--rankings",
                str(glacier),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        protocol_report = json.loads(protocol.stdout)
        self.assertEqual(protocol_report["queries"], 1)
        self.assertIn("exact_candidate_id", protocol_report["metrics"])

        # A production runner stages MIST only for the complement of GLACIER's
        # domain. An empty fallback is therefore valid for this all-eligible
        # one-query fixture and must not be indexed eagerly during the merge.
        fallback.write_text("", encoding="utf-8")
        partial_merged = self.root / "partial-merged"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/merge_glacier_fallback.py"
                ),
                "--input-root",
                str(glacier_input),
                "--glacier",
                str(glacier),
                "--fallback",
                str(fallback),
                "--output-root",
                str(partial_merged),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(
            common.jsonl(partial_merged / "rankings.jsonl")[0]["ranking"],
            ranking,
        )

        query["instrument_type"] = ""
        common.write_json(glacier_input / "queries.jsonl", query)
        staged_fallback = self.root / "staged-fallback"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/stage_glacier_fallback.py"
                ),
                "--input-root",
                str(glacier_input),
                "--output-root",
                str(staged_fallback),
            ],
            check=True,
            capture_output=True,
        )
        _, fallback_queries, fallback_pools = common.validate_input(staged_fallback)
        self.assertEqual(set(fallback_queries), {"q-test"})
        self.assertEqual(len(fallback_pools), 1)
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(
                [
                    sys.executable,
                    str(
                        TASK_ROOT
                        / "environment/references/scripts/merge_glacier_fallback.py"
                    ),
                    "--input-root",
                    str(glacier_input),
                    "--glacier",
                    str(glacier),
                    "--fallback",
                    str(fallback),
                    "--output-root",
                    str(self.root / "should-not-exist"),
                ],
                check=True,
                capture_output=True,
            )

        glacier_runner = (
            TASK_ROOT / "environment/references/scripts/run_glacier_reference.sh"
        ).read_text()
        self.assertIn("MASS_SPEC_GLACIER_FEATURE_CACHE_SIZE:-64", glacier_runner)
        self.assertIn("MASS_SPEC_GLACIER_PREFETCH_FACTOR:-1", glacier_runner)
        self.assertIn("MASS_SPEC_REFERENCE_PYTHON", glacier_runner)
        self.assertIn(
            "5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11",
            glacier_runner,
        )

    def test_synthetic_archive_packs_losslessly(self):
        source = self.root / "packed-source/msg_synth_filt"
        (source / "subformulae").mkdir(parents=True)
        labels = (
            "spec\tsmiles\tinchikey\tformula\n"
            "s1\tCCO\tKEY\tC2H6O\n"
            "s2\tnot-a-smiles\tOTHER\tC2H6O\n"
        )
        payload = {
            "cand_form": "C2H6O",
            "cand_ion": "[M+H]+",
            "output_tbl": None,
        }
        (source / "labels.csv").write_text(labels)
        (source / "subformulae/s1.json").write_text(json.dumps(payload))
        (source / "subformulae/s2.json").write_text(json.dumps(payload))
        archive = self.root / "synthetic.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(source, arcname="msg_synth_filt")
        output = self.root / "packed"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/pack_synthetic_subformulae.py"
                ),
                "--archive",
                str(archive),
                "--output-root",
                str(output),
                "--expected-rows",
                "2",
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual((output / "labels.csv").read_text(), labels)
        with sqlite3.connect(output / "subformulae.sqlite") as connection:
            stored = connection.execute(
                "SELECT payload FROM subformulae WHERE spec = 's1'"
            ).fetchone()[0]
        self.assertEqual(json.loads(stored), payload)

        experimental = self.root / "experimental.tsv"
        split = self.root / "split.tsv"
        experimental.write_text("spec\tinchikey\nheldout\tKEY-AAAA-B\n")
        split.write_text("name\tsplit\nheldout\ttest\n")
        filtered = self.root / "filtered.tsv"
        filter_report = self.root / "filter-report.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/filter_synthetic_heldout.py"
                ),
                "--labels",
                str(experimental),
                "--split",
                str(split),
                "--synthetic-labels",
                str(output / "labels.csv"),
                "--database",
                str(output / "subformulae.sqlite"),
                "--output-labels",
                str(filtered),
                "--output-report",
                str(filter_report),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(len(filtered.read_text().splitlines()), 1)
        report = json.loads(filter_report.read_text())
        self.assertEqual(report["removed_rows"], 2)
        self.assertEqual(report["removed_heldout_rows"], 1)
        self.assertEqual(report["removed_invalid_smiles_rows"], 1)
        with sqlite3.connect(output / "subformulae.sqlite") as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM subformulae").fetchone()[0],
                0,
            )

        # The operator-facing augmentation materializer uses the pinned
        # MassSpecGym TSV directly, without extracting the label-bearing
        # processed experimental archive into the agent-visible data root.
        output_tsv = self.root / "packed-tsv"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/pack_synthetic_subformulae.py"
                ),
                "--archive",
                str(archive),
                "--output-root",
                str(output_tsv),
                "--expected-rows",
                "2",
            ],
            check=True,
            capture_output=True,
        )
        boundary = self.root / "massspecgym.tsv"
        boundary.write_text(
            "identifier\tinchikey\tfold\nheldout\tKEY-AAAA-B\ttest\n"
        )
        filtered_tsv = self.root / "filtered-tsv.tsv"
        filter_report_tsv = self.root / "filter-report-tsv.json"
        subprocess.run(
            [
                sys.executable,
                str(
                    TASK_ROOT
                    / "environment/references/scripts/filter_synthetic_heldout.py"
                ),
                "--experimental-tsv",
                str(boundary),
                "--synthetic-labels",
                str(output_tsv / "labels.csv"),
                "--database",
                str(output_tsv / "subformulae.sqlite"),
                "--output-labels",
                str(filtered_tsv),
                "--output-report",
                str(filter_report_tsv),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(json.loads(filter_report_tsv.read_text())["removed_rows"], 2)
        augmentation_script = (
            TASK_ROOT / "environment/data/prepare_synthetic_augmentation.sh"
        ).read_text()
        self.assertIn("--experimental-tsv", augmentation_script)
        self.assertIn("--expected-synthetic-rows 2333485", augmentation_script)
        self.assertIn("verify_synthetic_augmentation.py", augmentation_script)


if __name__ == "__main__":
    unittest.main()
