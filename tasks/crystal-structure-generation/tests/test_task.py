"""Fast offline checks for the CrystalGen task contract and isolation."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import tomllib
import unittest
from unittest import mock


TESTS = Path(__file__).resolve().parent
TASK = TESTS.parent
sys.path.insert(0, str(TESTS))

from main_verifier import freeze_submission, seal_output, validate_final_suite  # noqa: E402
from scoring.contract import (  # noqa: E402
    load_output,
    load_requests,
    validate_submission,
)
from scoring.upstream import final_scores, normalize_result, writable_datasets_cache  # noqa: E402


def write_requests(root: Path, requests: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "requests.jsonl").write_text(
        "".join(json.dumps(request, sort_keys=True) + "\n" for request in requests)
    )


class CrystalGenTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.unconditional = {
            "schema_version": "crystalgen-request/v0",
            "request_id": "unconditional-dev",
            "mode": "unconditional",
            "condition": None,
            "n_samples": 2,
            "seed": 17,
        }
        self.conditional = {
            "schema_version": "crystalgen-request/v0",
            "request_id": "density-dev",
            "mode": "conditional",
            "condition": {
                "chemical_system": ["Cl", "Na"],
                "space_group": None,
                "target_property": {
                    "name": "density",
                    "value": 2.1,
                    "unit": "g/cm^3",
                    "tolerance": 0.2,
                },
            },
            "n_samples": 2,
            "seed": 23,
        }

    def test_native_layout_and_resources(self) -> None:
        config = tomllib.loads((TASK / "task.toml").read_text())
        self.assertEqual(config["artifacts"], ["/workspace/submission"])
        self.assertEqual(config["environment"]["gpus"], 8)
        self.assertEqual(config["verifier"]["environment_mode"], "separate")
        self.assertEqual(config["verifier"]["environment"]["gpus"], 8)
        for relative in (
            "README.md", "instruction.md", "environment/contract.md",
            "environment/Dockerfile", "environment/docker-compose.yaml",
            "environment/runtime/omatg-generator-runtime.patch",
            "environment/data/materialize.sh", "environment/data/scorer_assets/materialize.sh",
            "environment/data/README.md", "environment/data/MATERIALIZE.md",
            "environment/baseline/starter/README.md", "environment/baseline/starter/RESULTS.md",
            "environment/references/README.md", "docs/VALIDATION.md",
            "environment/evaluation/score.sh", "environment/baseline/starter/train.sh",
            "environment/evaluation/relax.sh", "environment/evaluation/relax.py",
            "environment/baseline/starter/omg_port/config.yaml",
            "environment/baseline/starter/omg_port/runtime.py",
            "environment/baseline/starter/omg_port/fit.sh",
            "environment/baseline/starter/omg_port/generation.py",
            "environment/baseline/starter/omg_port/publish.py",
            "tests/Dockerfile", "tests/docker-compose.yaml",
            "tests/runtime/omatg-generator-runtime.patch", "tests/test.sh",
        ):
            self.assertTrue((TASK / relative).is_file(), relative)

    def test_readme_cleanly_names_data_model_and_evaluation(self) -> None:
        readme = (TASK / "README.md").read_text()
        for phrase in (
            "### Data", "### Model starting point", "### Development evaluation",
            "### Untrusted model replay", "### Trusted grading", "## Metrics",
            "MP-20 train: 27,136 rows", "MP-20 test: 9,046 rows", "OmatG",
            "unconditional_sun_metasun", "SUN + MetaSUN", "2,500",
            "unconditional generation only", "fresh", "historical",
        ):
            if phrase == "OmatG":
                self.assertIn("OMatG", readme)
            else:
                self.assertIn(phrase, readme)
        self.assertIn("No | Not mounted", readme)

    def test_omatg_starter_is_unconditional_and_single_checkpoint(self) -> None:
        root = TASK / "environment" / "baseline" / "starter" / "omg_port"
        source = (root / "runtime.py").read_text()
        config = (root / "config.yaml").read_text()
        publisher = (root / "publish.py").read_text()
        generation = (root / "generation.py").read_text()
        self.assertFalse((root / "conditioning.py").exists())
        self.assertIn("class UnconditionalIndependentSampler(IndependentSampler)", source)
        for class_path in (
            "omg.model.model.Model", "omg.model.encoders.cspnet_full.CSPNetFull",
            "omg.datamodule.StructureDataset", "omg_port.runtime.UnconditionalIndependentSampler",
        ):
            self.assertIn(f"class_path: {class_path}", config)
        self.assertIn("prop: false", config)
        self.assertEqual(config.count("property_keys: []"), 3)
        self.assertNotIn("condition_dropout", config)
        self.assertNotIn("condition_encoder", config)
        self.assertIn("devices: 8", config)
        self.assertIn("strategy: ddp_find_unused_parameters_true", config)
        self.assertIn("num_workers: 0", config)
        self.assertIn("persistent_workers: false", config)
        self.assertIn("model.safetensors", publisher)
        self.assertIn("load_state_dict(state, strict=True)", generation)
        self.assertNotIn('        "predict",', generation)
        self.assertNotIn("enforce_request_condition", generation)
        self.assertNotIn("crystalgen_condition", generation)
        self.assertIn('"runtime.py"', publisher)
        self.assertNotIn("conditioning.py", publisher)
        self.assertIn("CUDA_VISIBLE_DEVICES", generation)

    def test_shared_runtime_and_scorer_are_identical(self) -> None:
        marker = "# END SHARED GENERATOR RUNTIME"
        # The development image is built from the task directory (it copies instruction.md), the
        # verifier's from tests/; the shared section is identical up to that prefix.
        development = (TASK / "environment/Dockerfile").read_text().split(marker)[0]
        development = development.replace("COPY environment/runtime/", "COPY runtime/")
        verifier = (TASK / "tests/Dockerfile").read_text().split(marker)[0]
        self.assertEqual(development, verifier)
        self.assertIn("ed80cb86fd69d4f4d14164c6635b0a8d4939b34f", development)
        self.assertIn("58e6eae3e4a6c87c22171cf069123ecc4e2fa7e6", development)
        for directory in ("environment/scoring", "tests/scoring"):
            self.assertFalse((TASK / directory / "conditions.py").exists())
        for name in ("__init__.py", "contract.py", "evaluate.py", "upstream.py", "relaxation.py", "relax_worker.py", "mace_assets.py", "displacement.py"):
            self.assertEqual(
                (TASK / "tests/scoring" / name).read_bytes(),
                (TASK / "environment/scoring" / name).read_bytes(),
            )
        self.assertEqual(
            (TASK / "environment/runtime/omatg-generator-runtime.patch").read_bytes(),
            (TASK / "tests/runtime/omatg-generator-runtime.patch").read_bytes(),
        )
        generator_patch = (TASK / "environment/runtime/omatg-generator-runtime.patch").read_text()
        self.assertIn('-    "torch-sim-atomistic ~= 0.5"', generator_patch)
        self.assertIn("Import it only when the command is used", generator_patch)
        self.assertEqual(
            (TASK / "environment/runtime/lemat-eight-gpu.patch").read_bytes(),
            (TASK / "tests/runtime/lemat-eight-gpu.patch").read_bytes(),
        )
        self.assertEqual(
            (TASK / "environment/runtime/lemat-small-batch.patch").read_bytes(),
            (TASK / "tests/runtime/lemat-small-batch.patch").read_bytes(),
        )
        self.assertEqual(
            (TASK / "environment/runtime/lemat-substitution-novelty.patch").read_bytes(),
            (TASK / "tests/runtime/lemat-substitution-novelty.patch").read_bytes(),
        )
        for patch in ("lemat-submitted-geometry.patch", "lemat-mace-reference-checkpoints.patch"):
            self.assertEqual((TASK / "environment/runtime" / patch).read_bytes(),
                             (TASK / "tests/runtime" / patch).read_bytes())
        model_patch = (TASK / "environment/runtime/lemat-mace-reference-checkpoints.patch").read_text()
        self.assertIn('MACE_MP_ENERGY_MODEL = "medium-0b3"', model_patch)
        self.assertIn('MACE_MP_EMBEDDING_MODEL = "medium-mpa-0"', model_patch)
        substitution_patch = (TASK / "environment/runtime/lemat-substitution-novelty.patch").read_text()
        # MatterGen equivalence and the element-substitution guard are installed in both scorers.
        self.assertIn("class DisorderedStructureMatcher(StructureMatcher):", substitution_patch)
        self.assertIn("self.disordered_matcher.fit(first, second)", substitution_patch)
        self.assertEqual(substitution_patch.count("DisorderAwareMatcher(self.fingerprinter)"), 2)
        self.assertIn("DILUTE_FRACTION = 0.25", substitution_patch)
        lemat_patch = (TASK / "environment/runtime/lemat-eight-gpu.patch").read_text()
        # Workers must bind GPUs by index; rewriting CUDA_VISIBLE_DEVICES after
        # CUDA initialisation leaves every worker on physical device 0.
        self.assertIn("gpu_index = worker_index % torch.cuda.device_count()", lemat_patch)
        self.assertIn("torch.cuda.set_device(gpu_index)", lemat_patch)
        # func_timeout runs each MLIP in a new thread; the device is thread-local.
        self.assertIn("def _run_on_worker_gpu(", lemat_patch)
        self.assertIn('"cuda" if mlip_name == "uma" else f"cuda:{gpu_index}"', lemat_patch)
        self.assertNotIn('os.environ["CUDA_VISIBLE_DEVICES"] =', lemat_patch)
        self.assertIn(").select_columns(", lemat_patch)
        verifier_docker = (TASK / "tests/Dockerfile").read_text()
        self.assertIn("FROM scorer-runtime AS grader", verifier_docker)
        self.assertIn("FROM generator-runtime AS main", verifier_docker)
        self.assertIn(
            "ln -s /scorer-assets/home/.cache /root/.cache",
            verifier_docker,
        )
        materializer = (TASK / "environment/data/scorer_assets/materialize.py").read_text()
        self.assertIn('"HOME": str(scorer_home)', materializer)
        self.assertIn('fairchem_cache.symlink_to(Path("../../huggingface/hub")', materializer)

    def test_final_generation_cannot_see_scorer_assets(self) -> None:
        compose = (TASK / "tests/docker-compose.yaml").read_text()
        main, grader = compose.split("  grader:", 1)
        self.assertNotIn("/scorer-assets", main)
        self.assertIn("/scorer-assets", grader)
        self.assertIn("network_mode: none", main)
        self.assertIn("network_mode: none", grader)
        self.assertNotIn("/data/mp20", compose)
        self.assertIn(
            "HF_DATASETS_CACHE: /scorer-assets/huggingface/datasets", grader
        )

    def test_development_mounts_follow_task_standard(self) -> None:
        compose = (TASK / "environment/docker-compose.yaml").read_text()
        for entry in ("mp20/train.csv", "mp20/val.csv", "mp20/manifest.json", "scorer-assets"):
            self.assertIn(f'/{entry}", target: /workspace/data/{entry}}}', compose)
        self.assertNotIn("target: /workspace/data/mp20/test.csv", compose)
        self.assertIn("target: /workspace/research", compose)
        self.assertNotIn("network_mode", compose)
        dockerfile = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("ln -s /workspace/data/scorer-assets /scorer-assets", dockerfile)
        self.assertIn("COPY instruction.md /workspace/instruction.md", dockerfile)
        self.assertNotIn("materialize", dockerfile.split("# END SHARED GENERATOR RUNTIME")[1])

    def test_operator_failures_cannot_be_published_as_model_zeroes(self):
        from main_verifier import InvalidSubmission, check_exit, validate_package
        with mock.patch("main_verifier.oom_kills", return_value=7):
            for code in (-9, 137, 1):
                with self.subTest(code=code), self.assertRaisesRegex(RuntimeError,"OOM kill"):
                    check_exit(code, 6, self.root)
            with self.assertRaisesRegex(RuntimeError,"unattributed SIGKILL"):
                check_exit(-9, 7, self.root)
            with self.assertRaises(InvalidSubmission):
                check_exit(2, 7, self.root)
        with mock.patch("main_verifier.validate_submission", side_effect=OSError("disk I/O failure")):
            with self.assertRaises(OSError):
                validate_package(self.root)

    def test_missing_trusted_launcher_yields_no_reward_and_notifies_grader(self):
        import main_verifier
        suite = self.build_suite("launcher-input")
        model = self.root / "model"
        output = self.root / "launcher-output"
        sealed = self.root / "launcher-phase" / "run"
        argv = ["main_verifier", "--input", str(suite), "--submission", str(model),
                "--output", str(output), "--sealed", str(sealed), "--allow-small-suite", "--no-wait"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(main_verifier, "validate_package"), \
             mock.patch.object(main_verifier, "freeze_submission", return_value=model), \
             mock.patch.object(main_verifier, "run_submission", side_effect=FileNotFoundError("trusted shell missing")):
            with self.assertRaises(FileNotFoundError):
                main_verifier.main()
        self.assertFalse((output / "reward.json").exists())
        self.assertEqual(json.loads((output / "error.json").read_text())["status"], "infrastructure_error")
        self.assertTrue((sealed.parent / "GENERATION_ERROR").is_file())

    def test_null_structure_counts_are_scorer_errors(self):
        for loaded, valid in [(1,None), (None,0), (1,2), (1,False), (1,-1)]:
            doc = {"run_info":{"n_structures":loaded},"validity_filtering":{"valid_structures":valid},"results":{}}
            with self.subTest(loaded=loaded,valid=valid), self.assertRaisesRegex(RuntimeError,"structure counts"):
                normalize_result(doc,expected=2)

    def test_invalid_submission_scores_zero(self) -> None:
        suite = self.build_suite("suite")
        submission = self.root / "submission"
        submission.mkdir()
        (submission / "README.md").write_text("no generate.sh\n")
        output = self.root / "verifier"
        sealed = self.root / "sealed" / "run"
        subprocess.run(
            [sys.executable, str(TESTS / "main_verifier.py"),
             "--submission", str(submission), "--input", str(suite),
             "--sealed", str(sealed), "--output", str(output)],
            check=True, capture_output=True,
        )
        reward = json.loads((output / "reward.json").read_text())
        self.assertEqual(reward["score"], 0.0)
        self.assertEqual(reward["reward"], 0.0)
        self.assertEqual(reward["invalid_submission"], 1.0)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["protocol"], "crystalgen-evaluation/v4-uma200-hull-aligned")
        self.assertEqual(report["evaluation_mode"], "sealed_replay_invalid_submission")
        self.assertEqual(report["requests_sha256"], hashlib.sha256((suite / "requests.jsonl").read_bytes()).hexdigest())
        self.assertIn("generate.sh", json.loads((output / "report.json").read_text())["error"])
        self.assertTrue((sealed.parent / "INVALID").is_file())
        graded = subprocess.run(
            [sys.executable, str(TESTS / "grader.py"),
             "--sealed", str(sealed), "--input", str(suite),
             "--assets", str(self.root / "unused-assets"), "--output", str(self.root / "grade"),
             "--wait-seconds", "5"],
            check=True, capture_output=True, text=True,
        )
        self.assertIn("nothing to score", graded.stdout)

    def test_submission_replay_copy_is_read_only(self) -> None:
        source = self.root / "submission"
        source.mkdir()
        (source / "generate.sh").write_text("#!/bin/sh\n")
        nested = source / "code"
        nested.mkdir()
        (nested / "model.py").write_text("pass\n")
        frozen = freeze_submission(source, self.root / "frozen")
        try:
            self.assertEqual(frozen.stat().st_mode & 0o777, 0o555)
            self.assertEqual((frozen / "generate.sh").stat().st_mode & 0o777, 0o444)
            self.assertEqual((frozen / "code").stat().st_mode & 0o777, 0o555)
        finally:
            for path in sorted(
                (item for item in frozen.rglob("*") if item.is_dir()), reverse=True
            ):
                path.chmod(0o755)
            frozen.chmod(0o755)

    def test_request_schema_rejects_mutations(self) -> None:
        suite = self.root / "suite"
        write_requests(suite, [self.unconditional])
        loaded = load_requests(suite)
        self.assertEqual(set(loaded), {"unconditional-dev"})
        for patch in (
            {"mode": "conditional"}, {"condition": self.conditional["condition"]},
            {"request_id": "../escape"}, {"extra": 1}, {"n_samples": True},
            {"n_samples": 0}, {"n_samples": 10001}, {"seed": True}, {"seed": -1},
        ):
            with self.subTest(patch=patch):
                write_requests(suite, [{**self.unconditional, **patch}])
                with self.assertRaises(ValueError):
                    load_requests(suite)

    def test_generator_rejects_legacy_conditional_inputs(self) -> None:
        for relative in ("generate.py", "omg_port/generation.py"):
            with self.subTest(relative=relative):
                spec = importlib.util.spec_from_file_location(
                    "generator_under_test", TASK / "environment/baseline/starter" / relative
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                suite = self.root / "suite"
                argument = suite / "requests.jsonl" if relative == "generate.py" else suite
                write_requests(suite, [self.unconditional])
                self.assertEqual(module.load_requests(argument), [self.unconditional])
                write_requests(suite, [self.conditional])
                with self.assertRaisesRegex(ValueError, "unconditional"):
                    module.load_requests(argument)

    def test_published_module_imports_without_condition_adapters(self) -> None:
        subprocess.run(
            [sys.executable, "-m", "omg_port.generation", "--help"],
            cwd=TASK / "environment/baseline/starter", check=True, capture_output=True,
        )

    def test_suite_builder_does_not_overwrite_existing_suite(self) -> None:
        suite = self.build_suite("suite")
        before = (suite / "requests.jsonl").read_bytes()
        result = subprocess.run(
            [sys.executable, str(TESTS / "requests/build_suite.py"),
             "--seed-file", str(self.root / "seed.bin"), "--output-root", str(suite)],
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, (suite / "requests.jsonl").read_bytes())

    def test_examples_are_unconditional_only(self) -> None:
        for directory, expected in (("baseline/starter", 2), ("evaluation", 100)):
            requests = load_requests(TASK / "environment" / directory / "examples/input")
            self.assertEqual(len(requests), 1)
            self.assertEqual(sum(r["n_samples"] for r in requests.values()), expected)

    def test_missing_and_duplicate_outputs_remain_failures(self) -> None:
        suite = self.root / "suite"
        write_requests(suite, [self.unconditional])
        output = self.root / "output"
        output.mkdir()
        cif = output / "one.cif"
        cif.write_text("data_one\n")
        (output / "manifest.csv").write_text(
            "request_id,sample_index,structure_path\n"
            "unconditional-dev,0,one.cif\n"
            "unconditional-dev,0,one.cif\n"
        )
        records, errors = load_output(output, load_requests(suite))
        self.assertEqual(records, {})
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))

    def test_submission_manifest_binds_checkpoint_and_training_data(self) -> None:
        submission = self.root / "submission"
        submission.mkdir()
        (submission / "generate.sh").write_text("#!/bin/sh\nexit 0\n")
        checkpoint = submission / "model.json"
        checkpoint.write_text(json.dumps({
            "schema": "crystalgen-smoke-model/v0",
            "element_counts": {},
            "atom_counts": {},
        }))
        manifest = {
            "schema": "crystalgen-model/v0",
            "checkpoint": {
                "path": "model.json",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "format": "json",
            },
            "parameter_count": 2,
            "one_parameter_set": True,
            "trained_from_scratch": True,
            "training_data": {
                "name": "MP-20", "split": "train",
                "source_sha256": "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d",
            },
            "model_config": {"architecture": "aggregate-statistics-smoke-generator"},
        }
        (submission / "model_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(validate_submission(submission)["parameter_count"], 2)
        checkpoint.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_submission(submission)

    def test_safetensors_parameter_count_is_computed_from_shapes(self) -> None:
        submission = self.root / "safe-submission"
        submission.mkdir()
        (submission / "generate.sh").write_text("#!/bin/sh\nexit 0\n")
        header = json.dumps({
            "weight": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]},
        }, separators=(",", ":")).encode()
        checkpoint = submission / "model.safetensors"
        checkpoint.write_bytes(struct.pack("<Q", len(header)) + header + bytes(24))
        manifest = {
            "schema": "crystalgen-model/v0",
            "checkpoint": {
                "path": checkpoint.name,
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "format": "safetensors",
            },
            "parameter_count": 6,
            "one_parameter_set": True,
            "trained_from_scratch": True,
            "training_data": {
                "name": "MP-20", "split": "train",
                "source_sha256": "133bab58dba02d316f9f91174839f17802dce648be4e8659d1a42b32b97fa01d",
            },
            "model_config": {"architecture": "test-network"},
        }
        (submission / "model_manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(validate_submission(submission)["parameter_count"], 6)
        manifest["parameter_count"] = 5
        (submission / "model_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "tensor shapes"):
            validate_submission(submission)

    def test_partial_hull_ensemble_is_not_a_scientific_zero(self):
        from scoring.upstream import check_hull_ensemble
        row = {"value":0.04, "value_orb":0.05, "value_mace":0.03, "value_uma":0.04}
        def document():
            result = {"evaluator_results":{"mean_e_above_hull":{"metric_results":{"mean_e_above_hull":{"individual_values":[row]}}}}}
            return {"results":{"stability":repr(result)}}
        check_hull_ensemble(document(), 1)
        row["value_uma"] = None
        with self.assertRaisesRegex(RuntimeError, "MLIP hull energy"):
            check_hull_ensemble(document(), 1)

    def test_wrong_scorer_cannot_publish_the_hull_aligned_protocol(self):
        from scoring.upstream import verify_scorer
        with tempfile.TemporaryDirectory() as path, self.assertRaisesRegex(RuntimeError, "Unexpected submitted-geometry scorer"):
            verify_scorer(Path(path))

    def test_leaderboard_normalization_uses_sun_plus_metasun(self) -> None:
        serialized = (
            "BenchmarkResult(final_scores={'stable_count': 4, 'metastable_count': 3, "
            "'sun_count': np.int64(2), 'msun_count': 3})"
        )
        self.assertEqual(final_scores(serialized)["sun_count"], 2)
        document = {
            "run_info": {"n_structures": 8},
            "validity_filtering": {"valid_structures": 7},
            "results": {
                "sun_new": serialized,
                "novelty_new": "BenchmarkResult(final_scores={'novel_structures_count': 6})",
                "uniqueness_new": "BenchmarkResult(final_scores={'unique_structures_count': 5})",
            },
        }
        rates, details = normalize_result(document, expected=10)
        self.assertEqual(rates["sun"], 0.2)
        self.assertEqual(rates["metasun"], 0.3)
        self.assertEqual(rates["sun_metasun"], 0.5)
        self.assertAlmostEqual(rates["ssi_score"], 0.35, places=12)
        self.assertEqual(rates["validity"], 0.7)
        self.assertEqual(details["counts"]["expected"], 10)

    def test_upstream_embedded_asset_failures_cannot_be_zero_scores(self) -> None:
        base = {
            "run_info": {"n_structures": 1},
            "validity_filtering": {"valid_structures": 1},
            "results": {
                "sun": "BenchmarkResult(final_scores={'sun_count': 0, 'msun_count': 0})",
                "novelty": "BenchmarkResult(final_scores={'novel_structures_count': 0})",
                "uniqueness": "BenchmarkResult(final_scores={'unique_structures_count': 1})",
            },
        }
        rates, _ = normalize_result(base, expected=1)
        self.assertEqual(rates["sun_metasun"], 0)
        failures = (
            ("novelty", "BenchmarkResult(final_scores={'novel_structures_count': 0, 'novelty_score': nan})"),
            ("novelty", "BenchmarkResult(final_scores={'novel_structures_count': 0, 'total_structures_evaluated': 0})"),
            ("distribution", "BenchmarkResult(warnings=['No embeddings found for model uma'])"),
            ("uniqueness", "BenchmarkResult(final_scores={'unique_structures_count': 2})"),
        )
        for name, serialized in failures:
            with self.subTest(name=name, serialized=serialized):
                document = {**base, "results": {**base["results"], name: serialized}}
                with self.assertRaises(RuntimeError):
                    normalize_result(document, expected=1)

    def test_read_only_datasets_cache_gets_a_writable_symlink_view(self) -> None:
        # `datasets.load_dataset` locks HF_DATASETS_CACHE even for cached offline
        # loads; on the read-only /scorer-assets mount that used to fail silently
        # and zero every hull-based metric.
        source = self.root / "assets" / "datasets"
        (source / "LeMaterial___le_mat-bulk-mlip-hull").mkdir(parents=True)
        (source / "LeMaterial___le_mat-bulk").mkdir()
        (source / "stale.lock").write_text("")
        work = self.root / "work"
        work.mkdir()
        with mock.patch.dict(os.environ, {"HF_DATASETS_CACHE": str(source)}):
            self.assertIsNone(writable_datasets_cache(work))
            source.chmod(0o555)
            try:
                if os.access(source, os.W_OK):
                    self.skipTest("cannot make the fixture read-only as this user")
                cache = writable_datasets_cache(work)
            finally:
                source.chmod(0o755)
        self.assertEqual(cache, work / "datasets-cache")
        self.assertTrue(os.access(cache, os.W_OK))
        links = sorted(path.name for path in cache.iterdir())
        self.assertEqual(links, ["LeMaterial___le_mat-bulk", "LeMaterial___le_mat-bulk-mlip-hull"])
        for link in cache.iterdir():
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), (source / link.name).resolve())

    def build_suite(self, name: str, seed_bytes: bytes = bytes(range(32))) -> Path:
        seed = self.root / "seed.bin"
        seed.write_bytes(seed_bytes)
        suite = self.root / name
        subprocess.run(
            [sys.executable, str(TESTS / "requests/build_suite.py"),
             "--seed-file", str(seed),
             "--output-root", str(suite)],
            check=True,
        )
        return suite

    def test_final_suite_builder_uses_only_2500_unconditional_samples(self) -> None:
        suite = self.build_suite("final-suite")
        requests = validate_final_suite(suite)
        self.assertEqual(sum(r["n_samples"] for r in requests.values()), 2_500)
        self.assertEqual(len(requests), 1)
        self.assertTrue(all(r["mode"] == "unconditional" for r in requests.values()))
        self.assertNotIn("cif", (suite / "requests.jsonl").read_text().lower())

    def test_final_suite_is_seeded_and_reproducible(self) -> None:
        first = self.build_suite("first")
        repeat = self.build_suite("repeat")
        fresh = self.build_suite("fresh", bytes(range(32, 64)))
        self.assertEqual((first / "requests.jsonl").read_bytes(), (repeat / "requests.jsonl").read_bytes())
        self.assertNotEqual((first / "requests.jsonl").read_bytes(), (fresh / "requests.jsonl").read_bytes())

    def test_final_suite_rejects_legacy_manifest_and_mismatched_count(self) -> None:
        suite = self.build_suite("suite")
        path = suite / "suite_manifest.json"
        manifest = json.loads(path.read_text())
        for patch in (
            {"schema": "crystalgen-final-suite/v0"}, {"conditional_samples": 0},
            {"unconditional_samples": 2499}, {"requests_sha256": "0" * 64},
        ):
            with self.subTest(patch=patch):
                path.write_text(json.dumps({**manifest, **patch}))
                with self.assertRaises(ValueError):
                    validate_final_suite(suite)

    def test_trusted_fixture_grader_reads_only_a_sealed_snapshot(self) -> None:
        suite = self.root / "suite"
        write_requests(suite, [self.unconditional])
        request_path = suite / "requests.jsonl"
        (suite / "suite_manifest.json").write_text(json.dumps({
            "schema": "crystalgen-final-suite/v1",
            "requests_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
            "unconditional_samples": 2,
        }))
        with self.assertRaises(ValueError):
            validate_final_suite(suite)
        self.assertEqual(len(validate_final_suite(suite, allow_small=True)), 1)
        untrusted = self.root / "untrusted"
        untrusted.mkdir()
        (untrusted / "manifest.csv").write_text(
            "request_id,sample_index,structure_path\n"
        )
        sealed = self.root / "sealed" / "run"
        seal_output(untrusted, sealed, request_path)
        fixture = self.root / "fixture.json"
        result = {
            "run_info": {"n_structures": 2},
            "validity_filtering": {"valid_structures": 2},
            "results": {
                "sun": "BenchmarkResult(final_scores={'stable_count': 1, "
                       "'metastable_count': 1, 'sun_count': 1, 'msun_count': 1, "
                       "'total_structures_evaluated': 2, 'metasun_count': 2, 'metasun_structure_indices': [0, 1]})",
                "novelty": "BenchmarkResult(final_scores={'novel_structures_count': 2})",
                "uniqueness": "BenchmarkResult(final_scores={'unique_structures_count': 2})",
                # Positive formation energy: low-energy by the hull window, but less stable than its elements.
                # Retained as diagnostic ONLY; never gates primary score = (SUN + 0.5 * MetaSUN) / expected.
                "stability": "BenchmarkResult(evaluator_results={'formation_energy': {'metric_results': "
                             "{'formation_energy': MetricResult(individual_values=[{'value': 0.05}, {'value': 0.02}])}}})",
            },
        }
        fixture.write_text(json.dumps({"unconditional": result}))
        output = self.root / "grade"
        try:
            subprocess.run(
                [sys.executable, str(TESTS / "grader.py"),
                 "--sealed", str(sealed), "--input", str(suite),
                 "--assets", str(self.root / "unused-assets"),
                 "--output", str(output), "--fixture-results", str(fixture),
                 "--allow-small-suite", "--wait-seconds", "0"],
                check=True,
            )
            reward = json.loads((output / "reward.json").read_text())
            self.assertEqual(reward["unconditional_sun"], 0.5)
            self.assertEqual(reward["unconditional_metasun"], 0.5)
            self.assertEqual(reward["unconditional_sun_metasun"], 1.0)
            self.assertAlmostEqual(reward["unconditional_ssi_score"], 0.75, places=12)
            self.assertEqual(reward["unconditional_negative_formation_unique_novel"], 0.0)
            self.assertAlmostEqual(reward["score"], 0.75, places=12)
            self.assertAlmostEqual(reward["reward"], 0.75, places=12)
            self.assertTrue(
                all(key.startswith("unconditional_") for key in reward if key not in {"reward", "score"})
            )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["schema"], "crystalgen-report/v4-uma200-hull-aligned")
            self.assertTrue(report["pre_scoring_relaxation"])
            self.assertEqual(set(report["tracks"]), {"unconditional"})
            self.assertNotIn("conditional", report)
            self.assertEqual(report["diagnostics"]["gating_policy"], "diagnostics_only_no_adhoc_gates")
            self.assertEqual(report["diagnostics"]["strict_sun_yield"], 0.5)
            self.assertEqual(report["diagnostics"]["sun_plus_metasun_yield"], 1.0)
            self.assertEqual(report["diagnostics"]["negative_formation_energy_fraction"], 0.0)
            self.assertIn("n_elements", report["diagnostics"])
            self.assertIn("as_submitted", report["diagnostics"])
            self.assertIn("relaxation_rmsd", report["diagnostics"])
        finally:
            for path in sorted(
                (item for item in sealed.rglob("*") if item.is_dir()), reverse=True
            ):
                path.chmod(0o755)
            sealed.chmod(0o755)

    def test_ssi_v3_crystal_parity_and_visibility_boundary(self) -> None:
        from scoring.evaluate import evaluate

        harness_path = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness" / "ssi_index_v3.py"
        spec = importlib.util.spec_from_file_location("ssi_index_v3", harness_path)
        ssi_v3 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ssi_v3)

        suite = self.build_suite("parity-suite")
        generated = self.root / "parity-gen"
        generated.mkdir()
        (generated / "manifest.csv").write_text("request_id,sample_index,structure_path\n")

        fixtures = [
            (0, 0, 0),
            (100, 0, 100),
            (0, 100, 100),
            (8, 430, 2400),
            (2500, 0, 2500),
            (0, 2500, 2500),
        ]
        for sun_c, msun_c, valid_c in fixtures:
            with self.subTest(sun=sun_c, metasun=msun_c, valid=valid_c):
                fixture_doc = {
                    "unconditional": {
                        "run_info": {"n_structures": valid_c},
                        "validity_filtering": {"valid_structures": valid_c},
                        "results": {
                            "sun": (
                                f"BenchmarkResult(final_scores={{'stable_count': {sun_c}, "
                                f"'metastable_count': {msun_c}, 'sun_count': {sun_c}, "
                                f"'msun_count': {msun_c}, 'total_structures_evaluated': {valid_c}, "
                                f"'metasun_count': {sun_c + msun_c}, "
                                f"'metasun_structure_indices': list(range({sun_c + msun_c}))}})"
                                .replace(f"list(range({sun_c + msun_c}))", str(list(range(sun_c + msun_c))))
                            ),
                            "novelty": f"BenchmarkResult(final_scores={{'novel_structures_count': {sun_c + msun_c}}})",
                            "uniqueness": f"BenchmarkResult(final_scores={{'unique_structures_count': {sun_c + msun_c}}})",
                            "stability": (
                                "BenchmarkResult(evaluator_results={'formation_energy': {'metric_results': "
                                "{'formation_energy': MetricResult(individual_values="
                                + repr([{"value": -0.1}] * valid_c)
                                + ")}}})"
                            ),
                        },
                    }
                }
                fixture_file = self.root / f"fixture_{sun_c}_{msun_c}.json"
                fixture_file.write_text(json.dumps(fixture_doc))
                rewards, _ = evaluate(
                    suite,
                    generated,
                    Path("unused-scorer"),
                    Path("unused-python"),
                    fixture_results=fixture_file,
                )
                v3_res = ssi_v3.crystal({"expected": 2500, "sun": sun_c, "metasun": msun_c})
                self.assertLess(abs(rewards["score"] - v3_res["score"]), 1e-12)
                self.assertLess(abs(rewards["reward"] - v3_res["score"]), 1e-12)

        tests_readme = (TESTS / "README.md").read_text()
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, tests_readme)
        for doc_rel in ("instruction.md", "environment/contract.md"):
            doc_text = (TASK / doc_rel).read_text()
            self.assertIn("0.5 * MetaSUN", doc_text)

        dev_dockerfile = (TASK / "environment/Dockerfile").read_text()
        dev_compose = (TASK / "environment/docker-compose.yaml").read_text()
        self.assertNotIn("COPY tests", dev_dockerfile)
        self.assertNotIn("/tests", dev_compose)


if __name__ == "__main__":
    unittest.main()
