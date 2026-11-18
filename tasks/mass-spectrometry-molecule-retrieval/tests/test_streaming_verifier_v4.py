"""V4 tests for persistent single-spectrum streaming (`massspecgym-retrieval-stream/v2`),
128-query repeated-pool cold-start statelessness spot-check, legacy batch fallback,
and V3 Hit@1 metric parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())
STARTER_ROOT = TASK_ROOT / "environment/baseline/starter"

sys.path.insert(0, str(TEST_ROOT))
import common
import grader
import main_verifier


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ssi_index_v3 = _load_module("ssi_index_v3", REPO_ROOT / "harness/ssi_index_v3.py")


def build_synthetic_dataset(root: Path, pool_specs: list[tuple[str, list[str], int]]):
    """Create a valid public verifier input + private labels directory with repeated/isomeric pools."""
    public = root / "public"
    private = root / "private"
    public.mkdir(parents=True)
    private.mkdir(parents=True)

    pools = []
    members = []
    pool_truth = {}
    for pool_id, smiles_list, query_count in pool_specs:
        candidates = [
            {"candidate_id": common.candidate_id(s), "smiles": s}
            for s in smiles_list
        ]
        pools.append({"candidate_pool_id": pool_id, "candidates": candidates})
        pool_truth[pool_id] = smiles_list[0]
        members.append([pool_id] * query_count)

    # Interleave queries across pools as in MassSpecGym v1.5.
    interleaved = []
    while any(members):
        for queue in members:
            if queue:
                interleaved.append(queue.pop(0))

    queries = []
    labels = []
    for idx, pool_id in enumerate(interleaved):
        qid = f"MassSpecGymID{idx:07d}"
        true_smiles = pool_truth[pool_id]
        queries.append({
            "query_id": qid,
            "mzs": [41.0386 + (idx % 7), 73.0284 + (idx % 11), 119.0491 + (idx % 13)],
            "intensities": [0.35, 1.0, 0.62],
            "precursor_mz": 180.063388 + idx * 0.0103,
            "adduct": "[M+H]+" if idx % 3 else "[M+Na]+",
            "instrument_type": "Orbitrap" if idx % 2 == 0 else "QTOF",
            "collision_energy": 30.0 if idx % 4 else None,
            "candidate_pool_id": pool_id,
        })
        labels.append({
            "query_id": qid,
            "correct_candidate_id": common.candidate_id(true_smiles),
            "true_smiles": true_smiles,
        })

    with (public / "queries.jsonl").open("w", encoding="utf-8") as stream:
        for row in queries:
            stream.write(common.compact_json(row) + "\n")
    with (public / "candidate_pools.jsonl").open("w", encoding="utf-8") as stream:
        for row in pools:
            stream.write(common.compact_json(row) + "\n")
    common.write_json(public / "manifest.json", {
        "schema": "massspecgym-retrieval-input/v1",
        "source": {"dataset": "MassSpecGym v1.5 synthetic fixture"},
        "counts": {"queries": len(queries), "candidate_pools": len(pools)},
        "files": {
            "queries.jsonl": common.sha256(public / "queries.jsonl"),
            "candidate_pools.jsonl": common.sha256(public / "candidate_pools.jsonl"),
        },
    })

    with (private / "labels.jsonl").open("w", encoding="utf-8") as stream:
        for row in labels:
            stream.write(common.compact_json(row) + "\n")
    common.write_json(private / "manifest.json", {
        "schema": "massspecgym-retrieval-private-labels/v1",
        "counts": {"queries": len(labels)},
        "files": {"labels.jsonl": common.sha256(private / "labels.jsonl")},
    })
    return public, private


def make_mock_package(destination: Path, server_code: str, *, include_serve_sh: bool = True):
    destination.mkdir(parents=True)
    (destination / "README.md").write_text("# Synthetic test package\n", encoding="utf-8")
    (destination / "model.pt").write_bytes(b"synthetic-weights")
    (destination / "model_config.json").write_text("{}\n", encoding="utf-8")
    (destination / "infer.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'exec python3 "$(dirname "$0")/batch_infer.py" "$1" "$2"\n',
        encoding="utf-8",
    )
    (destination / "infer.sh").chmod(0o755)
    (destination / "batch_infer.py").write_text(textwrap.dedent("""\
        import hashlib, json, pathlib, sys
        inp, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        pools = {
            row["candidate_pool_id"]: row["candidates"]
            for row in (json.loads(l) for l in (inp / "candidate_pools.jsonl").read_text().splitlines() if l.strip())
        }
        with (out / "rankings.jsonl").open("w", encoding="utf-8") as stream:
            for line in (inp / "queries.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                q = json.loads(line)
                cands = pools[q["candidate_pool_id"]]
                scored = sorted(
                    cands,
                    key=lambda c: (
                        hashlib.sha256(f"{q['precursor_mz']:.6f}:{len(q['mzs'])}:{c['smiles']}".encode()).hexdigest(),
                        c["candidate_id"],
                    ),
                )
                stream.write(json.dumps({"query_id": q["query_id"], "ranking": [c["candidate_id"] for c in scored]}) + "\\n")
    """), encoding="utf-8")
    if include_serve_sh:
        (destination / "predict_server.py").write_text(server_code, encoding="utf-8")
        (destination / "predict_server.py").chmod(0o755)
        (destination / "serve.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"\n'
            'exec python "$root/predict_server.py" "$@"\n',
            encoding="utf-8",
        )
        (destination / "serve.sh").chmod(0o755)
    common.write_json(destination / "model_manifest.json", {
        "schema": "massspecgym-retrieval-model/v1",
        "checkpoint": {
            "path": "model.pt",
            "sha256": common.sha256(destination / "model.pt"),
            "format": "pytorch",
        },
        "model_config": "model_config.json",
        "training_provenance": {"data": "synthetic", "pretrained_assets": []},
    })
    return destination


class RootlessReplayMixin:
    """Allow real unprivileged subprocess execution of main_verifier.evaluate() in CPU unit tests."""

    def run_full_verifier_and_grader(
        self, submission: Path, public: Path, private: Path, run_dir: Path
    ) -> tuple[dict, dict]:
        agent = mock.Mock(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(run_dir))
        logs_dir = run_dir / "logs"
        sealed_dir = run_dir / "sealed/run"
        score_dir = run_dir / "sealed/results"

        with (
            mock.patch.object(main_verifier.pwd, "getpwnam", return_value=agent),
            mock.patch.object(main_verifier, "readable_by", return_value=False),
            mock.patch.object(main_verifier, "isolate_agent_files", return_value={}),
            mock.patch.object(main_verifier, "purge_agent_state"),
            mock.patch.object(main_verifier, "kill_uid_processes"),
            mock.patch.object(main_verifier, "_make_demote", return_value=None),
            mock.patch.object(main_verifier.os, "chown"),
        ):
            v_args = argparse.Namespace(
                submission=submission,
                input=public,
                sealed=sealed_dir,
                output=logs_dir,
                timeout=120,
                no_wait=True,
            )
            main_verifier.evaluate(v_args)

        if (sealed_dir.parent / "READY").is_file():
            g_args = argparse.Namespace(
                sealed=sealed_dir,
                input=public,
                labels=private,
                output=score_dir,
                wait_seconds=2,
                mces_workers=1,
            )
            grader.run(g_args, compute_mces=False)
            for name in ("report.json", "reward.json"):
                shutil.copyfile(score_dir / name, logs_dir / name)

        report = json.loads((logs_dir / "report.json").read_text(encoding="utf-8"))
        reward = json.loads((logs_dir / "reward.json").read_text(encoding="utf-8"))
        return report, reward


class StreamingVerifierV4Tests(unittest.TestCase, RootlessReplayMixin):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.default_pools = [
            # Shared constitutional/stereoisomer pools with exact Morgan-fingerprint ties ([C@H] vs [C@@H])
            ("pool-shared-chiral", ["C[C@H](O)CC", "C[C@@H](O)CC", "CCCCO", "CCOCC"], 4),
            ("pool-shared-ketone", ["CC(=O)CC", "CCCC=O", "C1CCCO1"], 3),
            ("pool-shared-amine", ["CCNCC", "CCCCN", "CC(C)NC"], 3),
            ("pool-singleton-iso", ["c1ccccc1O", "C1=CC(=O)C=CC1"], 1),
            ("pool-singleton-2", ["CC(=O)O", "HCOOCH3".replace("HCOOCH3", "COC=O")], 1),
        ]

    def test_compliant_streaming_server_enforces_one_at_a_time_and_matches_batch_exactly(self):
        public, private = build_synthetic_dataset(self.root / "data", self.default_pools)
        audit_file = self.root / "dispatch_audit.jsonl"

        compliant_server = textwrap.dedent(f"""\
            import hashlib, json, os, pathlib, select, sys
            out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
            os.dup2(2, 1)
            audit_path = pathlib.Path({str(audit_file)!r})
            stdin_fd = sys.stdin.fileno()
            step = 0
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                req = json.loads(line)
                # Verify strict 1-at-a-time dispatch: query i+1 MUST NOT be waiting in stdin
                # before we emit our ranking response for query i!
                readable, _, _ = select.select([stdin_fd], [], [], 0.01)
                assert not readable, f"Lookahead violation at step {{step}}: unread bytes in stdin!"
                assert "candidate_pool_id" not in req, "candidate_pool_id must be stripped in stream mode"
                assert req["query_id"].startswith("sq_"), "query_id must be anonymized"
                cands = req["candidates"]
                scored = sorted(
                    cands,
                    key=lambda c: (
                        hashlib.sha256(f"{{req['precursor_mz']:.6f}}:{{len(req['mzs'])}}:{{c['smiles']}}".encode()).hexdigest(),
                        c["candidate_id"],
                    ),
                )
                with audit_path.open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps({{
                        "pid": os.getpid(),
                        "step": step,
                        "query_id": req["query_id"],
                        "precursor_mz": req["precursor_mz"],
                        "cand_order": [c["candidate_id"] for c in cands],
                    }}) + "\\n")
                step += 1
                out.write(json.dumps({{"query_id": req["query_id"], "ranking": [c["candidate_id"] for c in scored]}}) + "\\n")
                out.flush()
        """)

        stream_pkg = make_mock_package(self.root / "pkg-stream", compliant_server, include_serve_sh=True)
        batch_pkg = make_mock_package(self.root / "pkg-batch", compliant_server, include_serve_sh=False)

        stream_report, stream_reward = self.run_full_verifier_and_grader(
            stream_pkg, public, private, self.root / "run-stream"
        )
        batch_report, batch_reward = self.run_full_verifier_and_grader(
            batch_pkg, public, private, self.root / "run-batch"
        )

        self.assertEqual(stream_report["status"], "complete")
        self.assertEqual(stream_report["protocol_mode"], "massspecgym-retrieval-stream/v2")
        self.assertEqual(stream_report["evaluation_protocol"], "massspecgym-retrieval-stream/v2")
        self.assertEqual(batch_report["protocol_mode"], "massspecgym-per-spectrum-batches/v2")

        # Spot check verified all min(128, 12) = 12 queries including all 10 repeated-pool queries.
        spot = stream_report["statelessness_spot_check"]
        self.assertEqual(spot["checked_queries"], 12)
        self.assertEqual(spot["repeated_pool_queries"], 10)
        self.assertEqual(spot["mismatched_queries"], 0)
        self.assertTrue(spot["passed"])

        # Exact Hit@1, Hit@5, Hit@20, reward, score, and full ranking parity (diff == 0.0).
        self.assertEqual(stream_report["metrics"], batch_report["metrics"])
        self.assertEqual(abs(stream_reward["score"] - batch_reward["score"]), 0.0)
        self.assertEqual(stream_reward["score"], stream_reward["reward"])
        v3_expected = ssi_index_v3.massspec(stream_reward)["score"]
        v3_from_metrics = ssi_index_v3.massspec(stream_report["metrics"])["score"]
        self.assertLess(abs(stream_reward["score"] - v3_expected), 1e-12)
        self.assertLess(abs(stream_reward["score"] - v3_from_metrics), 1e-12)

        stream_rankings = common.jsonl(self.root / "run-stream/sealed/run/rankings.jsonl")
        batch_rankings = common.jsonl(self.root / "run-batch/sealed/run/rankings.jsonl")
        self.assertEqual(stream_rankings, batch_rankings)

        # Verify worker process lifecycle (1 persistent PID for Pass 1, 1 fresh cold-start PID for Pass 2)
        # and byte-identical precursor_mz preservation (zero synthetic noise).
        audit_rows = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(audit_rows), 24)
        pass1_rows, pass2_rows = audit_rows[:12], audit_rows[12:]
        self.assertEqual(len({r["pid"] for r in pass1_rows}), 1)
        self.assertEqual(len({r["pid"] for r in pass2_rows}), 1)
        self.assertNotEqual(pass1_rows[0]["pid"], pass2_rows[0]["pid"])
        orig_queries = common.jsonl(public / "queries.jsonl")
        self.assertEqual(
            [r["precursor_mz"] for r in pass1_rows],
            [q["precursor_mz"] for q in orig_queries],
        )

    def test_real_published_fourier_deepsets_starter_streams_and_matches_batch_mode(self):
        starter_model = _load_module("starter_model_v4", STARTER_ROOT / "model.py")
        starter_common = _load_module("starter_common_v4", STARTER_ROOT / "common.py")
        publication = _load_module("starter_pub_v4", STARTER_ROOT / "publication.py")

        torch.manual_seed(7)
        config = {
            "hidden_size": 32,
            "fingerprint_size": 256,
            "layers_per_mlp": 2,
            "dropout": 0.0,
            "max_peaks": 60,
            "mz_min": 10.0,
            "mz_max": 1000.0,
            "fingerprint_radius": 2,
        }
        net = starter_model.DeepSetsFingerprint(
            hidden_size=config["hidden_size"],
            fingerprint_size=config["fingerprint_size"],
            layers_per_mlp=config["layers_per_mlp"],
            dropout=config["dropout"],
        )
        state_dict = {k: v.cpu() for k, v in net.state_dict().items()}

        published_stream = self.root / "starter-published"

        def build(stage: Path):
            for name in (
                "infer.sh", "serve.sh", "predict.py", "predict_server.py",
                "model.py", "common.py", "publication.py",
            ):
                shutil.copyfile(STARTER_ROOT / name, stage / name)
            (stage / "infer.sh").chmod(0o755)
            (stage / "serve.sh").chmod(0o755)
            ckpt = stage / "model.pt"
            torch.save({"state_dict": state_dict, "model_config": config}, ckpt)
            (stage / "model_config.json").write_text(starter_common.compact_json(config) + "\n")
            (stage / "README.md").write_text("# Starter V4 streaming test\n")
            (stage / "model_manifest.json").write_text(starter_common.compact_json({
                "schema": "massspecgym-retrieval-model/v1",
                "checkpoint": {"path": "model.pt", "sha256": starter_common.sha256(ckpt), "format": "pytorch"},
                "model_config": "model_config.json",
                "training_provenance": {"data": "synthetic", "pretrained_assets": []},
            }) + "\n")

        publication.publish_directory(published_stream, build)
        self.assertTrue((published_stream / "serve.sh").is_file())
        published_batch = self.root / "starter-batch-only"
        shutil.copytree(published_stream, published_batch)
        (published_batch / "serve.sh").unlink()

        public, private = build_synthetic_dataset(self.root / "starter-data", self.default_pools)
        stream_report, stream_reward = self.run_full_verifier_and_grader(
            published_stream, public, private, self.root / "starter-run-stream"
        )
        batch_report, batch_reward = self.run_full_verifier_and_grader(
            published_batch, public, private, self.root / "starter-run-batch"
        )
        self.assertEqual(stream_report["protocol_mode"], "massspecgym-retrieval-stream/v2")
        self.assertTrue(stream_report["statelessness_spot_check"]["passed"])
        self.assertEqual(batch_report["protocol_mode"], "massspecgym-per-spectrum-batches/v2")
        self.assertEqual(stream_report["metrics"], batch_report["metrics"])
        self.assertEqual(stream_reward, batch_reward)
        self.assertEqual(
            common.jsonl(self.root / "starter-run-stream/sealed/run/rankings.jsonl"),
            common.jsonl(self.root / "starter-run-batch/sealed/run/rankings.jsonl"),
        )

    def test_stateful_cheating_servers_caught_and_scored_zero(self):
        public, private = build_synthetic_dataset(self.root / "cheat-data", self.default_pools)

        # Cheater 1: caches seen candidate pools in memory and alters ranking on later occurrences.
        pool_cache_cheater = textwrap.dedent("""\
            import json, sys
            seen_pools = {}
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                cids = sorted(c["candidate_id"] for c in req["candidates"])
                pool_key = tuple(cids)
                occ = seen_pools.get(pool_key, 0)
                seen_pools[pool_key] = occ + 1
                # Rotate top candidate on later repeated-pool occurrences (transductive elimination cheat).
                shift = occ % len(cids)
                ranking = cids[shift:] + cids[:shift]
                sys.stdout.write(json.dumps({"query_id": req["query_id"], "ranking": ranking}) + "\\n")
                sys.stdout.flush()
        """)
        pkg1 = make_mock_package(self.root / "cheater-pool-cache", pool_cache_cheater)
        report1, reward1 = self.run_full_verifier_and_grader(
            pkg1, public, private, self.root / "run-cheater-1"
        )
        self.assertEqual(report1["status"], "stateful_query_leakage")
        self.assertEqual(report1["protocol_mode"], "massspecgym-retrieval-stream/v2")
        self.assertFalse(report1["model_quality_measured"])
        self.assertFalse(report1["statelessness_spot_check"]["passed"])
        self.assertGreater(report1["statelessness_spot_check"]["mismatched_queries"], 0)
        self.assertEqual(reward1, {"reward": 0.0, "score": 0.0})
        self.assertEqual(report1["score"], 0.0)

        # Cheater 2: depends on input candidate list order (candidate-position shortcut).
        order_cheater = textwrap.dedent("""\
            import json, sys
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                ranking = [c["candidate_id"] for c in req["candidates"]]
                sys.stdout.write(json.dumps({"query_id": req["query_id"], "ranking": ranking}) + "\\n")
                sys.stdout.flush()
        """)
        pkg2 = make_mock_package(self.root / "cheater-order", order_cheater)
        report2, reward2 = self.run_full_verifier_and_grader(
            pkg2, public, private, self.root / "run-cheater-2"
        )
        self.assertEqual(report2["status"], "stateful_query_leakage")
        self.assertEqual(reward2, {"reward": 0.0, "score": 0.0})

    def test_extra_scores_field_in_streaming_response_is_rejected(self):
        public, private = build_synthetic_dataset(self.root / "extra-field-data", self.default_pools[:2])
        bad_schema_server = textwrap.dedent("""\
            import json, sys
            for line in sys.stdin:
                if not line.strip():
                    continue
                req = json.loads(line)
                cids = sorted(c["candidate_id"] for c in req["candidates"])
                sys.stdout.write(json.dumps({
                    "query_id": req["query_id"],
                    "ranking": cids,
                    "scores": [1.0] * len(cids),
                }) + "\\n")
                sys.stdout.flush()
        """)
        pkg = make_mock_package(self.root / "pkg-bad-schema", bad_schema_server)
        report, reward = self.run_full_verifier_and_grader(
            pkg, public, private, self.root / "run-bad-schema"
        )
        self.assertEqual(report["status"], "invalid_output")
        self.assertEqual(reward, {"reward": 0.0, "score": 0.0})

    def test_spot_check_stratification_caps_at_128_and_prioritizes_repeated_pools(self):
        pool_specs = (
            [(f"rep-{i:03d}", [f"C{'C' * (i % 8)}O", f"C{'C' * (i % 8)}N"], 3) for i in range(45)]
            + [(f"single-{i:03d}", [f"C{'C' * (i % 6)}S", f"C{'C' * (i % 6)}F"], 1) for i in range(45)]
        )
        public, _ = build_synthetic_dataset(self.root / "large-data", pool_specs)
        _, queries, pools = common.validate_input(public)
        self.assertEqual(len(queries), 180)  # 45 * 3 + 45 * 1 = 180 > 128

        selected, repeated_count = main_verifier.select_spot_check_queries(queries, pools)
        self.assertEqual(len(selected), 128)
        self.assertEqual(len(set(selected)), 128)
        # All 135 repeated-pool queries (90 later occurrences + 45 first occurrences) outrank singletons,
        # so all 128 sampled queries come from repeated pools!
        self.assertEqual(repeated_count, 128)
        # Every one of the 45 distinct repeated pools has at least one later occurrence included.
        by_pool = {}
        for qid, row in queries.items():
            by_pool.setdefault(row["candidate_pool_id"], []).append(qid)
        later_occurrences = {
            qid for members in by_pool.values() if len(members) >= 2 for qid in members[1:]
        }
        self.assertEqual(len(set(selected) & later_occurrences), 90)
        self.assertNotEqual(selected, list(queries)[:128])

    def test_v3_metric_alignment_and_visibility_boundary(self):
        for raw_hit1 in (0.0, 0.04004329004329004, 0.40453406242879927, 0.4939052175894281, 1.0):
            rew_payload = {"reward": raw_hit1, "score": raw_hit1}
            v3 = ssi_index_v3.massspec(rew_payload)
            self.assertLess(abs(v3["score"] - raw_hit1), 1e-12)
            self.assertLess(abs(rew_payload["score"] - v3["score"]), 1e-12)

        dockerfile = (TASK_ROOT / "environment/Dockerfile").read_text(encoding="utf-8")
        dev_compose = (TASK_ROOT / "environment/docker-compose.yaml").read_text(encoding="utf-8")
        self.assertNotIn("COPY tests", dockerfile)
        self.assertNotIn("/tests", dev_compose)

        sealed_readme = (TEST_ROOT / "README.md").read_text(encoding="utf-8")
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, sealed_readme)


if __name__ == "__main__":
    unittest.main()
