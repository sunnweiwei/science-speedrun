"""Small offline interface tests; no model training or benchmark data download."""

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import unittest
from unittest.mock import patch

import grade


TASK = Path(__file__).resolve().parents[1]
MATERIALIZE = TASK / "environment/data/materialize.sh"
CIF = """data_synthetic
loop_
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
CA ALA A 0.0 0.0 0.0
#
"""


def csv_file(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_feature_file(path, digest, length):
    value_bytes = length * grade.FEATURE_WIDTH
    scale_bytes = length * (grade.FEATURE_WIDTH // grade.FEATURE_GROUP_SIZE) * 2
    header = {
        "values": {
            "dtype": "I8", "shape": [length, grade.FEATURE_WIDTH],
            "data_offsets": [0, value_bytes],
        },
        "scales": {
            "dtype": "BF16",
            "shape": [length, grade.FEATURE_WIDTH // grade.FEATURE_GROUP_SIZE],
            "data_offsets": [value_bytes, value_bytes + scale_bytes],
        },
        "__metadata__": {
            "sequence_sha256": digest,
            "esmc_repo": "biohub/ESMC-6B",
            "esmc_revision": grade.ESMC_REVISION,
            "boundary": grade.FEATURE_BOUNDARY,
            "kernel_backend": grade.KERNEL_BACKEND,
            "schema_version": grade.FEATURE_VERSION,
            "quantization": "symmetric-int8",
            "group_size": str(grade.FEATURE_GROUP_SIZE),
            "scale_dtype": "bfloat16",
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded
                     + bytes(value_bytes + scale_bytes))


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.request = {
            "schema_version": "foldbench-folding-request/v0", "request_id": "brand-new-id",
            "target": {"entities": [{"id": "E1", "type": "protein", "sequence": "AAA"}]},
            "run": {"seeds": [7, 11], "num_samples": 2},
        }

    def suite(self):
        suite = self.root / "suite"
        (suite / "requests").mkdir(parents=True)
        request_path = suite / "requests/item.json"
        request_path.write_text(json.dumps(self.request))
        (suite / "suite_manifest.json").write_text(json.dumps({
            "contains_ground_truth_coordinates": False,
            "requests": [{"request_id": self.request["request_id"], "path": "requests/item.json",
                          "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest()}],
        }))
        feature_dir = self.root / "features/esmc6b-final-int8"
        feature_dir.mkdir(parents=True, exist_ok=True)
        self.feature_manifest = self.root / "features.esmc6b_final_int8.jsonl"
        rows = []
        sequences = {entity["sequence"] for entity in self.request["target"]["entities"]
                     if entity["type"] == "protein"}
        for sequence in sorted(sequences):
            digest = hashlib.sha256(sequence.encode()).hexdigest()
            synthetic_feature_file(
                feature_dir / f"{digest}.safetensors", digest, len(sequence)
            )
            rows.append(json.dumps({
                "sequence_sha256": digest,
                "length": len(sequence),
                "shape": [len(sequence), 2560],
                "dtype": "int8",
                "scale_shape": [len(sequence), 40],
                "scale_dtype": "bfloat16",
                "quantization": "symmetric-int8",
                "group_size": 64,
                "feature_version": grade.FEATURE_VERSION,
                "path": f"features/esmc6b-final-int8/{digest}.safetensors",
            }))
        self.feature_manifest.write_text("\n".join(rows) + ("\n" if rows else ""))
        return suite

    def prediction(self, **overrides):
        return dict(pdb_id="brand-new-id", seed=7, sample=0, ranking_score=0,
                    prediction_path="structure.cif", **overrides)

    def test_native_layout_and_standard_agent_budget(self):
        config = tomllib.loads((TASK / "task.toml").read_text())
        self.assertEqual(config["verifier"]["environment_mode"], "separate")
        self.assertEqual(config["artifacts"], ["/workspace/submission"])
        self.assertEqual(config["agent"]["timeout_sec"], 172800.0)  # TASK_STANDARD default, not told to the agent
        for path in ("instruction.md", "environment/contract.md", "environment/Dockerfile",
                     "tests/Dockerfile", "tests/test.sh", "environment/baseline/starter/infer.sh",
                     "environment/data/assemble_local_release.py",
                     "environment/data/finalize_local_features.py",
                     "environment/data/validate_release.py"):
            self.assertTrue((TASK / path).is_file(), path)

    def test_prompt_defines_continuous_loop_and_from_scratch_boundary(self):
        instruction = (TASK / "instruction.md").read_text()
        contract = (TASK / "environment/contract.md").read_text()
        self.assertIn("continuing research task", instruction)
        self.assertIn("Do not return a final answer", instruction)
        self.assertIn("immediately begin the next experiment", instruction)
        self.assertIn("train from scratch", instruction)
        self.assertIn("must not copy, initialize", instruction)
        self.assertIn("final-layer groupwise-INT8", instruction)
        self.assertIn("Learn any feature projection", instruction)
        self.assertIn("45M--55M", instruction)
        self.assertIn("serious starting point", instruction)
        self.assertIn("full model R&D", instruction)
        self.assertIn("45,000,000--55,000,000", contract)
        self.assertIn("reproducible parameter-audit command", contract)
        self.assertNotIn("Supplied pretrained weights may be used", instruction)
        self.assertNotIn("Select ONE final model", instruction)
        self.assertNotIn("external operator", instruction)
        self.assertNotIn("killed", instruction)

    def test_canonical_feature_codec_round_trip(self):
        import torch
        from safetensors.torch import save_file

        module_path = TASK / "environment/baseline/starter/reference/features.py"
        spec = importlib.util.spec_from_file_location("foldbench_features", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        values = torch.arange(-127, 128, dtype=torch.int8).repeat(2, 11)[:, :2560].contiguous()
        scales = torch.full((2, 40), 0.25, dtype=torch.bfloat16)
        path = self.root / "feature.safetensors"
        save_file({"values": values, "scales": scales}, str(path))
        decoded = module.load_esmc_final(path)
        expected = values.float() * 0.25
        self.assertEqual(decoded.shape, (2, 2560))
        self.assertTrue(torch.equal(decoded, expected))

    def test_development_mount_excludes_hf_cache(self):
        compose = (TASK / "environment/docker-compose.yaml").read_text()
        self.assertNotIn("target: /workspace/data\n", compose)
        self.assertNotIn("/hf\n", compose)
        for subtree in ("manifests", "objects", "features", "foldbench", "models"):
            self.assertIn(f"target: /workspace/data/{subtree}", compose)

    def test_runtime_build_contexts_share_dependency_recipe(self):
        marker = "# END SHARED RUNTIME"
        development = (TASK / "environment/Dockerfile").read_text().split(marker)[0]
        verifier = (TASK / "tests/Dockerfile").read_text().split(marker)[0]
        self.assertEqual(development, verifier)
        self.assertNotIn("auth2.json", development)
        self.assertIn("/opt/esmfold2-assets", development)
        for source in ("Protenix", "OpenDDE", "dplm"):
            self.assertIn(source, development)
        self.assertIn('git init "/opt/references/$1"', development)
        self.assertIn("COPY . /tests", (TASK / "tests/Dockerfile").read_text())

    def test_runnable_50m_starter_is_primary_and_from_scratch(self):
        starter = TASK / "environment/baseline/starter"
        for path in (
            "train.sh", "publish.py", "publish_latest.py", "model.py",
            "esmfold2_50m/architecture.py", "esmfold2_50m/audit.py",
            "esmfold2_50m/adapter.py", "esmfold2_50m/config.py",
            "esmfold2_50m/runtime.py", "esmfold2_50m/train.py",
        ):
            self.assertTrue((starter / path).is_file(), path)
        config = (starter / "esmfold2_50m/config.py").read_text()
        architecture = (starter / "esmfold2_50m/architecture.py").read_text()
        publisher = (starter / "publish.py").read_text()
        self.assertIn("EXPECTED_PARAMETER_RANGE = (45_000_000, 55_000_000)", config)
        self.assertIn("lm_num_layers=0", config)
        self.assertIn("final_hidden.unsqueeze(2)", architecture)
        self.assertNotIn("initialize_frozen_lm_projection", architecture)
        self.assertIn('"pretrained_parameters": False', publisher)
        self.assertIn('"audit.py"', publisher)
        self.assertIn('os.replace(next_link, args.submission / "current")', publisher)
        trainer = (starter / "esmfold2_50m/train.py").read_text()
        indexer = (starter / "reference/build_reference_index.py").read_text()
        self.assertIn('dtype=torch.bfloat16', trainer)
        self.assertIn('model.set_kernel_backend(None)', trainer)
        self.assertIn('actual_tokens > max_tokens or actual_atoms > max_atoms', trainer)
        # Polymer-less pool entries are skipped before the adapter, and a
        # model-side failure on one rank becomes a zero-gradient step that
        # still joins the collective instead of ending the run.
        self.assertIn('if not record.get("sequences"):', trainer)
        self.assertIn('record_skipped_example(args.output_dir, rank, step_index + 1, record, error)', trainer)
        self.assertIn('parameter.grad = torch.zeros_like(parameter)', trainer)
        self.assertNotIn('"msa": msa', indexer)

    def test_prior_reference_architectures_are_mounted(self):
        references = TASK / "environment/references"
        self.assertTrue((references / "README.md").is_file())
        for architecture in ("esmfold2", "protenix", "opendde", "dplm2"):
            self.assertTrue(
                (references / "prior_workspace/official_baselines" / architecture / "README.md").is_file(),
                architecture,
            )

    def test_materializer_placeholder_fails_before_creation(self):
        env = dict(os.environ, FOLDBENCH_HF_REPO="__HF_DATASET_REPO__",
                   FOLDBENCH_HF_REVISION="__HF_COMMIT_SHA__")
        destination = self.root / "not-created"
        result = subprocess.run(["bash", str(MATERIALIZE), str(destination)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("not published yet", result.stderr)
        self.assertFalse(destination.exists())

    def test_mutable_revision_rejected(self):
        env = dict(os.environ, FOLDBENCH_HF_REPO="owner/dataset", FOLDBENCH_HF_REVISION="main")
        result = subprocess.run(["bash", str(MATERIALIZE), str(self.root / "data")],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("immutable commit", result.stderr)

    def test_download_failure_propagates_without_verification(self):
        fake_hf = self.root / "hf"
        fake_hf.write_text("#!/bin/sh\nexit 17\n")
        fake_hf.chmod(0o755)
        env = dict(os.environ, PATH=f"{self.root}:{os.environ['PATH']}",
                   FOLDBENCH_HF_REPO="owner/dataset", FOLDBENCH_HF_REVISION="a" * 40)
        result = subprocess.run(["bash", str(MATERIALIZE), str(self.root / "data with spaces")],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 17)
        self.assertNotIn("verified", result.stdout)

    def test_suite_uses_unknown_ids_and_dynamic_seed_policy(self):
        requests = grade.suite_requests(self.suite())
        self.assertEqual(requests["brand-new-id"]["run"], {"seeds": [7, 11], "num_samples": 2})

    def test_suite_checksum_is_checked(self):
        suite = self.suite()
        (suite / "requests/item.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "checksum"):
            grade.suite_requests(suite)

    def test_suite_rejects_extra_inference_fields(self):
        self.request["native_path"] = "/answers/answer.cif"
        with self.assertRaisesRegex(ValueError, "fields"):
            grade.suite_requests(self.suite())

    def test_fixed_feature_manifest_requires_exact_suite_coverage(self):
        suite = self.suite()
        requests = grade.suite_requests(suite)
        records = grade.suite_features(self.feature_manifest, requests)
        self.assertEqual(len(records), 1)
        self.feature_manifest.write_text("")
        with self.assertRaisesRegex(ValueError, "coverage"):
            grade.suite_features(self.feature_manifest, requests)

    def test_development_subset_stager_is_self_contained(self):
        suite = self.suite()
        output = self.root / "staged"
        result = subprocess.run([
            sys.executable,
            str(TASK / "environment/evaluation/stage_inputs.py"),
            "--suite", str(suite),
            "--features", str(self.feature_manifest),
            "--output", str(output),
            "--request-id", "brand-new-id",
        ], check=True, capture_output=True, text=True)
        self.assertIn("brand-new-id", result.stdout)
        row = json.loads((output / "features.esmc6b_final_int8.jsonl").read_text())
        feature = output / row["path"]
        self.assertTrue(feature.is_file())
        self.assertFalse(feature.is_symlink())

    def test_corrupt_feature_payload_is_rejected(self):
        suite = self.suite()
        record = json.loads(self.feature_manifest.read_text())
        (self.feature_manifest.parent / record["path"]).write_bytes(b"not-safetensors")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            grade.suite_features(self.feature_manifest, grade.suite_requests(suite))

    def test_forbidden_supplied_checkpoint_is_rejected(self):
        model = self.root / "submission"
        model.mkdir()
        weight = model / "renamed.safetensors"
        weight.write_bytes(b"official checkpoint fixture")
        digest = hashlib.sha256(weight.read_bytes()).hexdigest()
        with patch.object(grade, "FORBIDDEN_WEIGHT_SHA256", {digest}):
            with self.assertRaisesRegex(ValueError, "forbidden supplied checkpoint"):
                grade.audit_submission(model)

    def test_historical_candidate_and_optimizer_state_are_rejected(self):
        model = self.root / "submission"
        (model / "candidate-v3").mkdir(parents=True)
        (model / "candidate-v3/model.safetensors").write_bytes(b"not official")
        with self.assertRaisesRegex(ValueError, "historical candidate"):
            grade.audit_submission(model)
        shutil.rmtree(model / "candidate-v3")
        (model / "optimizer.pt").write_bytes(b"training state")
        with self.assertRaisesRegex(ValueError, "training or historical"):
            grade.audit_submission(model)

    def test_output_validation_and_duplicate_rejection(self):
        (self.root / "structure.cif").write_text(CIF)
        row = self.prediction()
        csv_file(self.root / "prediction_reference.csv", [row])
        accepted, invalid = grade.predictions(self.root, {"brand-new-id": self.request})
        self.assertEqual(len(accepted), 1)
        self.assertEqual(invalid, [])
        csv_file(self.root / "prediction_reference.csv", [row, row])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            grade.predictions(self.root, {"brand-new-id": self.request})

    def test_output_path_cannot_escape(self):
        output = self.root / "output"
        output.mkdir()
        (self.root / "outside.cif").write_text(CIF)
        row = self.prediction()
        row["prediction_path"] = "../outside.cif"
        csv_file(output / "prediction_reference.csv", [row])
        with self.assertRaisesRegex(ValueError, "escaping"):
            grade.predictions(output, {"brand-new-id": self.request})

    def test_missing_structure_counts_as_invalid(self):
        csv_file(self.root / "prediction_reference.csv", [self.prediction()])
        accepted, invalid = grade.predictions(self.root, {"brand-new-id": self.request})
        self.assertFalse(accepted)
        self.assertEqual(len(invalid), 1)

    def test_nonfinite_structure_is_invalid_not_dropped_from_denominator(self):
        (self.root / "structure.cif").write_text(CIF.replace("0.0 0.0 0.0", "nan 0.0 0.0"))
        csv_file(self.root / "prediction_reference.csv", [self.prediction()])
        accepted, invalid = grade.predictions(self.root, {"brand-new-id": self.request})
        self.assertEqual(accepted, [])
        self.assertEqual(len(invalid), 1)
        metric = grade.track_score("monomer_protein", [{"pdb_id": "new", "chain_id": "A"}], [])
        self.assertEqual(metric["score"], 0)
        self.assertEqual(metric["targets"], 1)

    def test_fixed_denominator_and_ranking_not_oracle(self):
        targets = [{"pdb_id": "x", "chain_id": "A"}, {"pdb_id": "y", "chain_id": "A"}]
        raw = [{"pdb_id": "x", "chain_id": "A", "ranking_score": "0", "lddt": "0.4"},
               {"pdb_id": "x", "chain_id": "A", "ranking_score": "-1", "lddt": "0.9"}]
        metric = grade.track_score("monomer_protein", targets, raw)
        self.assertEqual(metric["score"], 0.2)
        self.assertEqual(metric["missing_or_invalid"], 1)

    def test_nonfinite_highest_rank_does_not_select_lower_rank_answer(self):
        targets = [{"pdb_id": "x", "chain_id": "A"}]
        raw = [{"pdb_id": "x", "chain_id": "A", "ranking_score": "1", "lddt": "nan"},
               {"pdb_id": "x", "chain_id": "A", "ranking_score": "0", "lddt": "0.9"}]
        self.assertEqual(grade.track_score("monomer_protein", targets, raw)["score"], 0)

    def test_collapsed_interface_fails_capri_geometry(self):
        spec = importlib.util.spec_from_file_location("foldbench_grade_capri", TASK / "tests/grade.py")
        grade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grade)
        dev_spec = importlib.util.spec_from_file_location(
            "foldbench_track_scores", TASK / "environment/evaluation/track_scores.py")
        dev = importlib.util.module_from_spec(dev_spec)
        dev_spec.loader.exec_module(dev)
        targets = [{"pdb_id": f"t{i}", "interface_chain_id_1": "A", "interface_chain_id_2": "B"} for i in range(4)]
        rows = [
            {"dockq_score": "0.36", "irmsd": "15.0", "lrmsd": "30.0"},  # collapsed: fnat alone passes DockQ
            {"dockq_score": "0.30", "irmsd": "3.5", "lrmsd": "12.0"},   # acceptable by interface RMSD
            {"dockq_score": "0.25", "irmsd": "6.0", "lrmsd": "9.0"},    # acceptable by ligand RMSD
            {"dockq_score": "0.20", "irmsd": "1.0", "lrmsd": "2.0"},    # below the DockQ threshold
        ]
        raw = [dict(target, ranking_score="1", **row) for target, row in zip(targets, rows)]
        self.assertAlmostEqual(grade.track_score("interface_protein_protein", targets, raw)["score"], 0.5)
        self.assertAlmostEqual(dev.track_score("interface_protein_protein", targets, raw), 0.5)

    def test_interface_and_ligand_thresholds(self):
        target = {"pdb_id": "x", "interface_chain_id_1": "A", "interface_chain_id_2": "B"}
        raw = dict(target, ranking_score="0", dockq_score="0.23", irmsd="4.0", lrmsd="11")
        self.assertEqual(grade.track_score("interface_protein_dna", [target], [raw])["score"], 1)
        raw = dict(target, ranking_score="0", dockq_score="0.23")  # no RMSD: geometry unknown, not a success
        self.assertEqual(grade.track_score("interface_protein_dna", [target], [raw])["score"], 0)
        ligand = {"pdb_id": "x", "native_chain_id_1": "A", "native_chain_id_2": "B"}
        raw = dict(target, ranking_score="0", rmsd="1.99", **{"lddt-pli": "0.81"})
        self.assertEqual(grade.track_score("interface_protein_ligand", [ligand], [raw])["score"], 1)
        raw["rmsd"] = "2"
        self.assertEqual(grade.track_score("interface_protein_ligand", [ligand], [raw])["score"], 0)

    def test_starter_from_foreign_cwd_and_relocated_model_directory(self):
        model = self.root / "relocated model"
        model.mkdir()
        for name in ("infer.sh", "predict.py"):
            (model / name).write_text((TASK / "environment/baseline/starter" / name).read_text())
        (model / "model.py").write_text(
            "from pathlib import Path\n"
            "def load_model(model_dir):\n"
            "    assert (Path(model_dir) / 'weights.txt').read_text() == 'test'\n"
            "    return Model()\n"
            "class Model:\n"
            "    def predict(self, **kwargs):\n"
            f"        return {{'mmcif': {CIF!r}, 'ranking_score': 0}}\n"
        )
        (model / "weights.txt").write_text("test")
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "requests.jsonl").write_text(json.dumps(self.request) + "\n")
        digest = hashlib.sha256(b"AAA").hexdigest()
        feature_dir = inputs / "features/esmc6b-final-int8"
        feature_dir.mkdir(parents=True)
        synthetic_feature_file(feature_dir / f"{digest}.safetensors", digest, 3)
        (inputs / "features.esmc6b_final_int8.jsonl").write_text(json.dumps({
            "sequence_sha256": digest, "length": 3, "shape": [3, 2560],
            "dtype": "int8", "scale_shape": [3, 40], "scale_dtype": "bfloat16",
            "quantization": "symmetric-int8", "group_size": 64,
            "feature_version": grade.FEATURE_VERSION,
            "path": f"features/esmc6b-final-int8/{digest}.safetensors",
        }) + "\n")
        output = self.root / "output"
        subprocess.run(["bash", str(model / "infer.sh"), str(inputs), str(output)],
                       cwd="/tmp", check=True, capture_output=True)
        accepted, invalid = grade.predictions(output, {"brand-new-id": self.request})
        self.assertEqual(len(accepted), 4)
        self.assertFalse(invalid)

    def test_inference_timeout_kills_the_process_group(self):
        spec = importlib.util.spec_from_file_location("foldbench_grade", TASK / "tests/grade.py")
        grade = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(grade)
        self.assertEqual(grade.INFERENCE_SECONDS, 8 * 3600)
        marker = self.root / "survivor"
        script = self.root / "slow.sh"
        script.write_text(f"(sleep 3; touch {marker}) &\nsleep 30\n")
        with (self.root / "log").open("w") as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                grade.run_inference(["bash", str(script)], self.root, log, 1)
        import time
        time.sleep(4)
        self.assertFalse(marker.exists(), "a background worker outlived the timeout")
        with (self.root / "log").open("w") as log:
            with self.assertRaises(subprocess.CalledProcessError):
                grade.run_inference(["bash", "-c", "exit 3"], self.root, log, 10)

    def test_starter_records_one_failed_request_without_aborting_the_batch(self):
        model = self.root / "model"
        model.mkdir()
        for name in ("infer.sh", "predict.py"):
            (model / name).write_text((TASK / "environment/baseline/starter" / name).read_text())
        (model / "model.py").write_text(
            "def load_model(model_dir):\n"
            "    return Model()\n"
            "class Model:\n"
            "    def predict(self, request, **kwargs):\n"
            "        if request['request_id'] == 'too-large':\n"
            "            raise RuntimeError('CUDA out of memory. Tried to allocate 3.27 GiB')\n"
            f"        return {{'mmcif': {CIF!r}, 'ranking_score': 0}}\n"
        )
        inputs = self.root / "inputs"
        inputs.mkdir()
        large = dict(self.request, request_id="too-large")
        (inputs / "requests.jsonl").write_text(
            json.dumps(self.request) + "\n" + json.dumps(large) + "\n")
        digest = hashlib.sha256(b"AAA").hexdigest()
        feature_dir = inputs / "features/esmc6b-final-int8"
        feature_dir.mkdir(parents=True)
        synthetic_feature_file(feature_dir / f"{digest}.safetensors", digest, 3)
        (inputs / "features.esmc6b_final_int8.jsonl").write_text(json.dumps({
            "sequence_sha256": digest, "length": 3, "shape": [3, 2560],
            "dtype": "int8", "scale_shape": [3, 40], "scale_dtype": "bfloat16",
            "quantization": "symmetric-int8", "group_size": 64,
            "feature_version": grade.FEATURE_VERSION,
            "path": f"features/esmc6b-final-int8/{digest}.safetensors",
        }) + "\n")
        output = self.root / "output"
        completed = subprocess.run(["bash", str(model / "infer.sh"), str(inputs), str(output)],
                                   cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        requests = {"brand-new-id": self.request, "too-large": large}
        accepted, invalid = grade.predictions(output, requests)
        self.assertEqual(len(accepted), 4)
        self.assertFalse(invalid)
        failures = [json.loads(line) for line in (output / "failures.jsonl").read_text().splitlines()]
        self.assertEqual([entry["request_id"] for entry in failures], ["too-large"] * 4)
        self.assertEqual(failures[0]["error_type"], "RuntimeError")

    def test_replay_and_reward_with_mock_official_scorer(self):
        suite = self.suite()
        csv_file(suite / "targets/monomer_protein.csv", [{"pdb_id": "brand-new-id", "chain_id": "A"}])
        natives = self.root / "natives"
        natives.mkdir()
        (natives / "brand-new-id.cif").write_text(CIF)
        model = self.root / "model"
        model.mkdir()
        (model / "infer.sh").write_text('exec python3 "$(dirname "$0")/infer.py" "$@"\n')
        (model / "infer.py").write_text(textwrap.dedent(f"""
            import csv, json, sys
            from pathlib import Path
            request = json.loads((Path(sys.argv[1]) / 'requests.jsonl').read_text())
            output = Path(sys.argv[2])
            (output / 'prediction.cif').write_text({CIF!r})
            with (output / 'prediction_reference.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames={grade.FIELDS!r})
                writer.writeheader()
                writer.writerow(dict(pdb_id=request['request_id'], seed=7, sample=0,
                                     ranking_score=0, prediction_path='prediction.cif'))
        """))
        output = self.root / "verifier"
        output.mkdir()
        real_run = subprocess.run

        def score_or_run(command, **kwargs):
            if command[0] == "/opt/foldbench-env/bin/python":
                csv_file(output / "evaluation/submission/raw/monomer_protein_ost.csv", [
                    dict(pdb_id="brand-new-id", chain_id="A", ranking_score="0", lddt="0.75")
                ])
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)

        args = type("Args", (), dict(suite=suite, features=self.feature_manifest,
                                     ground_truth=natives, submission=model, output=output))()
        with patch.object(grade.subprocess, "run", side_effect=score_or_run):
            grade.run(args)
        self.assertEqual(json.loads((output / "reward.json").read_text()), {"reward": 0.75, "monomer_protein": 0.75})
        self.assertFalse(json.loads((output / "report.json").read_text())["label_isolated_sandbox"])

    @unittest.skipUnless(os.environ.get("FOLDBENCH_NATIVE_SMOKE_DIR"), "optional native-scorer fixture")
    def test_native_scorer_fixture_not_a_model_benchmark(self):
        # Plumbing only: replay a bundled native as both prediction and reference.
        # This is intentionally NOT a trained submission or a benchmark result.
        fixture = Path(os.environ["FOLDBENCH_NATIVE_SMOKE_DIR"])
        self.request = json.loads((fixture / "suite/requests/7pv5-assembly1.json").read_text())
        self.request["request_id"] = "unseen-native-fixture"
        self.request["run"] = {"seeds": [101], "num_samples": 1}
        suite = self.suite()
        csv_file(suite / "targets/monomer_protein.csv", [{"pdb_id": self.request["request_id"], "chain_id": "A"}])
        natives, model, output = (self.root / name for name in ("natives", "model", "verifier"))
        for path in (natives, model, output):
            path.mkdir()
        shutil.copyfile(fixture / "ground_truths/7pv5-assembly1.cif", natives / "unseen-native-fixture.cif")
        shutil.copyfile(natives / "unseen-native-fixture.cif", model / "fixture.cif")
        (model / "infer.sh").write_text('exec python3 "$(dirname "$0")/infer.py" "$@"\n')
        (model / "infer.py").write_text(textwrap.dedent(f"""
            import csv, json, shutil, sys
            from pathlib import Path
            request = json.loads((Path(sys.argv[1]) / 'requests.jsonl').read_text())
            output = Path(sys.argv[2])
            shutil.copyfile(Path(__file__).parent / 'fixture.cif', output / 'prediction.cif')
            with (output / 'prediction_reference.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames={grade.FIELDS!r})
                writer.writeheader()
                writer.writerow(dict(pdb_id=request['request_id'], seed=101, sample=0,
                                     ranking_score=0, prediction_path='prediction.cif'))
        """))
        args = type("Args", (), dict(suite=suite, features=self.feature_manifest,
                                     ground_truth=natives, submission=model, output=output))()
        try:
            grade.run(args)
        except subprocess.CalledProcessError:
            self.fail((output / "scoring.log").read_text()[-8000:])
        reward = json.loads((output / "reward.json").read_text())
        self.assertAlmostEqual(reward["monomer_protein"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
