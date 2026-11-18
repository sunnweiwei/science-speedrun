"""Per-pool batching: no invocation sees two spectra of one candidate pool."""

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import main_verifier


POOL_SIZES = [5, 1, 3, 3, 2, 1, 4]


def make_input(root: Path, sizes=POOL_SIZES):
    """Queries interleaved across pools, as in the real release (pools are not contiguous)."""
    root.mkdir(parents=True)
    pools, members = [], []
    for index, size in enumerate(sizes):
        smiles = [f"C{'C' * index}O{'O' * k}" for k in range(3)]
        candidates = [{"candidate_id": common.candidate_id(s), "smiles": s} for s in smiles]
        pools.append({"candidate_pool_id": f"pool-{index}", "candidates": candidates})
        members.append([f"pool-{index}"] * size)
    order = []
    while any(members):
        for queue in members:
            if queue:
                order.append(queue.pop())
    queries = [{
        "query_id": f"MassSpecGymID{number:07d}", "mzs": [50.0, 60.0 + number],
        "intensities": [0.5, 1.0], "precursor_mz": 100.0 + number, "adduct": "[M+H]+",
        "instrument_type": "Orbitrap", "collision_energy": None, "candidate_pool_id": pool_id,
    } for number, pool_id in enumerate(order)]
    with (root / "queries.jsonl").open("w") as stream:
        stream.writelines(common.compact_json(row) + "\n" for row in queries)
    with (root / "candidate_pools.jsonl").open("w") as stream:
        stream.writelines(common.compact_json(row) + "\n" for row in pools)
    common.write_json(root / "manifest.json", {
        "schema": "massspecgym-retrieval-input/v1", "source": {"revision": "test"},
        "counts": {"queries": len(queries), "candidate_pools": len(pools)},
        "files": {name: common.sha256(root / name) for name in main_verifier.INPUT_FILES},
    })
    return common.validate_input(root)


class BatchPlanTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest, self.queries, self.pools = make_input(self.root / "input")

    def test_no_pool_twice_every_query_once_and_fewest_batches(self):
        batches = main_verifier.plan_batches(self.queries)
        self.assertEqual(len(batches), max(POOL_SIZES))
        for batch in batches:
            pools = [self.queries[query_id]["candidate_pool_id"] for query_id in batch]
            self.assertEqual(len(pools), len(set(pools)))
        flat = [query_id for batch in batches for query_id in batch]
        self.assertEqual(sorted(flat), sorted(self.queries))
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual([len(batch) for batch in batches],
                         [sum(size > j for size in POOL_SIZES) for j in range(max(POOL_SIZES))])

    def test_plan_is_fixed_and_keeps_input_order_inside_a_batch(self):
        batches = main_verifier.plan_batches(self.queries)
        # The assignment depends on query IDs only, not on the input order.
        self.assertEqual([set(batch) for batch in batches],
                         [set(batch) for batch in main_verifier.plan_batches(dict(reversed(self.queries.items())))])
        position = {query_id: index for index, query_id in enumerate(self.queries)}
        for batch in batches:
            self.assertEqual(batch, sorted(batch, key=position.__getitem__))

    def test_batch_inputs_are_valid_and_hold_only_their_queries_and_pools(self):
        query_lines = main_verifier.keyed_lines(self.root / "input/queries.jsonl", "query_id")
        pool_lines = main_verifier.keyed_lines(self.root / "input/candidate_pools.jsonl", "candidate_pool_id")
        seen = []
        for index, batch in enumerate(main_verifier.plan_batches(self.queries)):
            destination = self.root / f"batch-{index}"
            main_verifier.write_batch_input(destination, self.manifest, batch, self.queries,
                                            query_lines, pool_lines)
            manifest, queries, pools = common.validate_input(destination)
            self.assertEqual(list(queries), batch)
            self.assertEqual(set(pools), {self.queries[q]["candidate_pool_id"] for q in batch})
            self.assertEqual(manifest["source"], self.manifest["source"])
            for query_id, row in queries.items():
                self.assertEqual(row, self.queries[query_id])
                self.assertEqual(pools[row["candidate_pool_id"]], self.pools[row["candidate_pool_id"]])
            seen.extend(queries)
        self.assertEqual(sorted(seen), sorted(self.queries))


class BatchOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def write(self, *rows):
        (self.output / "rankings.jsonl").write_text(
            "".join((row if isinstance(row, str) else json.dumps(row)) + "\n" for row in rows))

    def test_rows_outside_the_batch_duplicates_and_garbage_invalidate_the_output(self):
        batch = ["a", "b"]
        self.assertIsNone(main_verifier.read_batch_rankings(self.output, batch))
        self.write({"query_id": "b", "ranking": ["x"]}, "", {"query_id": "a", "ranking": ["y"]})
        self.assertEqual(set(main_verifier.read_batch_rankings(self.output, batch)), {"a", "b"})
        for rows in ([{"query_id": "c", "ranking": []}],
                     [{"query_id": "a", "ranking": []}, {"query_id": "a", "ranking": []}],
                     [{"query_id": ["a"], "ranking": []}], [["a"]], ["{not json"]):
            with self.subTest(rows=rows):
                self.write(*rows)
                with self.assertRaises(main_verifier.InvalidBatchOutput):
                    main_verifier.read_batch_rankings(self.output, batch)

    def test_symlinked_ranking_file_is_not_read(self):
        (self.output / "elsewhere.jsonl").write_text('{"query_id":"a","ranking":[]}\n')
        (self.output / "rankings.jsonl").symlink_to(self.output / "elsewhere.jsonl")
        self.assertIsNone(main_verifier.read_batch_rankings(self.output, ["a"]))


class RunBatchesTests(unittest.TestCase):
    """run_batches with the process launch replaced by an in-process fake package."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest, self.queries, self.pools = make_input(self.root / "input")
        self.work = self.root / "work"
        self.work.mkdir()
        self.calls = []
        agent = mock.Mock(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(self.root))
        for patch in (mock.patch.object(main_verifier.pwd, "getpwnam", return_value=agent),
                      mock.patch.object(main_verifier, "readable_by", return_value=False),
                      mock.patch.object(main_verifier, "isolate_agent_files", return_value={}),
                      mock.patch.object(main_verifier, "purge_agent_state"),
                      mock.patch.object(main_verifier.os, "chown")):
            patch.start()
            self.addCleanup(patch.stop)

    def fake_package(self, skip=(), sleep=0.0):
        def run(submission, input_root, work, timeout, home=None):
            _, queries, pools = common.validate_input(input_root)
            self.calls.append({"queries": list(queries), "home": home, "input": input_root,
                               "timeout": timeout})
            output = work / "untrusted-output"
            output.mkdir()
            (work / "inference.log").write_text(f"{len(queries)} queries\n")
            if len(self.calls) in skip:
                return output
            with (output / "rankings.jsonl").open("w") as stream:
                for query_id, row in reversed(list(queries.items())):
                    stream.write(json.dumps({"query_id": query_id,
                                             "ranking": list(pools[row["candidate_pool_id"]])}) + "\n")
            return output
        return run

    def test_one_fresh_invocation_per_batch_and_rankings_merged_in_input_order(self):
        with mock.patch.object(main_verifier, "run_submission", side_effect=self.fake_package()):
            merged = main_verifier.run_batches(Path("/package"), self.root / "input", self.manifest,
                                               self.queries, self.work, 100)
        self.assertEqual(len(self.calls), max(POOL_SIZES))
        self.assertEqual(len({str(call["home"]) for call in self.calls}), 1)  # recreated per batch
        for call in self.calls:
            pools = [self.queries[q]["candidate_pool_id"] for q in call["queries"]]
            self.assertEqual(len(pools), len(set(pools)))
        rows = [json.loads(line) for line in (merged / "rankings.jsonl").read_text().splitlines()]
        self.assertEqual([row["query_id"] for row in rows], list(self.queries))
        self.assertFalse((self.work / "batch").exists())
        self.assertEqual(sorted(path.name for path in (self.work / "logs").iterdir()),
                         [f"{index:04d}.log" for index in range(max(POOL_SIZES))])
        self.assertEqual(stat.S_IMODE((self.work / "logs").stat().st_mode), 0o700)
        record = json.loads((self.work / "batches.json").read_text())
        self.assertEqual(record["batches"], max(POOL_SIZES))
        self.assertEqual(sum(item["queries"] for item in record["invocations"]), len(self.queries))
        self.assertEqual(main_verifier.purge_agent_state.call_count, max(POOL_SIZES))

    def test_a_batch_without_ranking_file_leaves_the_merged_file_missing(self):
        with mock.patch.object(main_verifier, "run_submission", side_effect=self.fake_package(skip={2})):
            merged = main_verifier.run_batches(Path("/package"), self.root / "input", self.manifest,
                                               self.queries, self.work, 100)
        self.assertFalse((merged / "rankings.jsonl").exists())
        self.assertEqual(len(self.calls), 2)

    def test_budget_is_compute_timeout_plus_a_startup_allowance_per_invocation(self):
        allowance = main_verifier.STARTUP_SECONDS_PER_INVOCATION
        self.assertEqual(main_verifier.inference_budget(18000, 373), 18000 + 373 * allowance)
        with mock.patch.object(main_verifier, "run_submission", side_effect=self.fake_package()):
            main_verifier.run_batches(Path("/package"), self.root / "input", self.manifest,
                                      self.queries, self.work, 100)
        record = json.loads((self.work / "batches.json").read_text())
        self.assertEqual(record["inference_budget_seconds"], 100 + max(POOL_SIZES) * allowance)
        self.assertGreater(self.calls[0]["timeout"], 100)

    def test_the_timeout_covers_all_invocations(self):
        clock = iter(range(0, 1000, 30))
        with mock.patch.object(main_verifier, "run_submission", side_effect=self.fake_package()), \
             mock.patch.object(main_verifier, "STARTUP_SECONDS_PER_INVOCATION", 0), \
             mock.patch.object(main_verifier.time, "monotonic", side_effect=lambda: next(clock)):
            with self.assertRaises(subprocess.TimeoutExpired):
                main_verifier.run_batches(Path("/package"), self.root / "input", self.manifest,
                                          self.queries, self.work, 100)
        self.assertLess(len(self.calls), max(POOL_SIZES))
        self.assertTrue(all(call["timeout"] <= 100 for call in self.calls))
        self.assertEqual(main_verifier.purge_agent_state.call_count, len(self.calls) + 1)

    def test_readable_full_input_is_an_infrastructure_error(self):
        with mock.patch.object(main_verifier, "readable_by", return_value=True), \
             mock.patch.object(main_verifier, "run_submission") as run:
            with self.assertRaisesRegex(RuntimeError, "full test input"):
                main_verifier.run_batches(Path("/package"), self.root / "input", self.manifest,
                                          self.queries, self.work, 100)
        run.assert_not_called()


class IsolationHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_readable_by_follows_directory_traversal_bits(self):
        self.root.chmod(0o755)   # mkdtemp is private; the test controls traversal below it
        hidden = self.root / "hidden"
        hidden.mkdir(mode=0o755)
        (hidden / "queries.jsonl").write_text("{}\n")
        (hidden / "queries.jsonl").chmod(0o644)
        other_uid, other_gid = os.getuid() + 12345, os.getgid() + 12345
        self.assertTrue(main_verifier.readable_by(hidden / "queries.jsonl", other_uid, other_gid))
        hidden.chmod(0o700)
        self.addCleanup(hidden.chmod, 0o755)
        self.assertFalse(main_verifier.readable_by(hidden / "queries.jsonl", other_uid, other_gid))
        self.assertTrue(main_verifier.readable_by(hidden / "queries.jsonl", os.getuid(), os.getgid()))

    def test_purge_removes_everything_new_or_owned_by_the_inference_uid(self):
        shared = self.root / "tmp"
        shared.mkdir()
        (shared / "image-file").write_text("kept")
        directories = {shared: {"image-file"}}
        (shared / "state.txt").write_text("answer")
        (shared / "nested").mkdir()
        (shared / "nested/deep").write_text("answer")
        (shared / "link").symlink_to(shared / "image-file")
        other_uid = os.getuid() + 12345   # image files are not owned by the inference UID
        with mock.patch.object(main_verifier, "remove_ipc_objects") as remove_ipc:
            main_verifier.purge_agent_state(other_uid, os.getgid(), directories)
            self.assertEqual(sorted(path.name for path in shared.iterdir()), ["image-file"])
            main_verifier.purge_agent_state(os.getuid(), os.getgid(), directories)   # now owned by the UID
            self.assertEqual(list(shared.iterdir()), [])
        self.assertEqual(remove_ipc.call_args_list, [mock.call(other_uid, os.getgid()),
                                                     mock.call(os.getuid(), os.getgid())])

    def test_isolation_finds_writable_directories_and_removes_rewrite_access(self):
        tree = self.root / "tree"
        (tree / "open").mkdir(parents=True)
        (tree / "open").chmod(0o777)
        (tree / "closed").mkdir()
        (tree / "closed").chmod(0o755)
        (tree / "closed/writable-file").write_text("x")
        (tree / "closed/writable-file").chmod(0o666)
        (tree / "closed/plain").write_text("x")
        (tree / "closed/plain").chmod(0o644)
        other_uid, other_gid = os.getuid() + 12345, os.getgid() + 12345
        walk = os.walk
        with mock.patch.object(main_verifier.os, "walk", side_effect=lambda top: walk(tree)), \
             mock.patch.object(main_verifier.os, "lchown") as lchown:
            directories = main_verifier.isolate_agent_files(other_uid, other_gid)
        lchown.assert_not_called()
        self.assertEqual(set(directories), {tree / "open"})
        self.assertTrue((tree / "open").stat().st_mode & stat.S_ISVTX)
        self.assertEqual(stat.S_IMODE((tree / "closed/writable-file").stat().st_mode), 0o644)
        self.assertEqual(stat.S_IMODE((tree / "closed/plain").stat().st_mode), 0o644)
        # Entries owned by the inference UID are handed to root with read access kept.
        (tree / "closed/own").write_text("x")
        (tree / "closed/own").chmod(0o600)
        with mock.patch.object(main_verifier.os, "walk", side_effect=lambda top: walk(tree / "closed")), \
             mock.patch.object(main_verifier.os, "lchown") as lchown:
            main_verifier.isolate_agent_files(os.getuid(), other_gid)
        self.assertIn(mock.call(tree / "closed/own", 0, 0), lchown.call_args_list)
        self.assertEqual(stat.S_IMODE((tree / "closed/own").stat().st_mode), 0o644)


if __name__ == "__main__":
    unittest.main()
