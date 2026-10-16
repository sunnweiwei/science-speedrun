"""split_suite.py: disjoint, grouped, quota-respecting and deterministic validation/test parts."""
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "environment/data/split_suite.py"
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def protein(rng, length=120):
    return "".join(rng.choice(ALPHABET) for _ in range(length))


def build_suite(root: Path) -> dict:
    """60 single-chain assemblies over two tracks, with a planted same-entry pair, an identical-chain pair
    and a near-identical pair (one substitution)."""
    rng = random.Random(7)
    suite = root / "foldbench/suite"
    (suite / "requests").mkdir(parents=True)
    (suite / "targets").mkdir()
    (root / "foldbench/ground_truths").mkdir()
    feature_dir = root / "foldbench/features/v1"
    feature_dir.mkdir(parents=True)
    sequences = {f"{i:04d}-assembly1": protein(rng) for i in range(60)}
    sequences["0001-assembly2"] = protein(rng)                      # same PDB entry as 0001-assembly1
    sequences["0100-assembly1"] = sequences["0002-assembly1"]       # identical chain
    near = list(sequences["0003-assembly1"]); near[10] = "W" if near[10] != "W" else "Y"
    sequences["0101-assembly1"] = "".join(near)                      # near-identical chain
    items, tracks, rows = [], {}, {"monomer_protein": [], "interface_protein_ligand": []}
    features = []
    for n, (rid, seq) in enumerate(sorted(sequences.items())):
        track = "monomer_protein" if n % 3 else "interface_protein_ligand"
        tracks[rid] = track
        request = {"request_id": rid, "target": {"entities": [{"id": "E1", "type": "protein", "sequence": seq, "copies": 1}]}}
        (suite / "requests" / f"{rid}.json").write_text(json.dumps(request))
        (root / "foldbench/ground_truths" / f"{rid}.cif").write_text(f"data_{rid}\n")
        items.append({"request_id": rid, "path": f"requests/{rid}.json", "tracks": [track]})
        rows[track].append([rid, "A"])
        digest = hashlib.sha256(seq.encode()).hexdigest()
        if not (feature_dir / f"{digest}.safetensors").exists():
            (feature_dir / f"{digest}.safetensors").write_text(digest)
            features.append({"sequence_sha256": digest, "path": f"features/v1/{digest}.safetensors"})
    for track, table in rows.items():
        with (suite / "targets" / f"{track}.csv").open("w", newline="") as stream:
            csv.writer(stream).writerows([["pdb_id", "chain_id"]] + table)
    by_track = {t: sum(1 for v in tracks.values() if v == t) for t in rows}
    (suite / "suite_manifest.json").write_text(json.dumps({
        "requests": items, "assemblies": len(items), "assemblies_by_track": by_track,
        "targets_by_track": by_track, "seed": 101}))
    (root / "foldbench/features.esmc6b_final_int8.jsonl").write_text("".join(json.dumps(r) + "\n" for r in features))
    return tracks


def split(root: Path, out: Path, *extra) -> dict:
    run = subprocess.run([sys.executable, str(SCRIPT), str(root / "foldbench"), str(out), *extra],
                         capture_output=True, text=True, check=True)
    return json.loads(run.stdout)


def ids(part: Path) -> set:
    return {item["request_id"] for item in json.loads((part / "suite/suite_manifest.json").read_text())["requests"]}


class SplitSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tracks = build_suite(self.root)

    def test_parts_are_disjoint_complete_and_grouped(self):
        split(self.root, self.root)
        val, test = ids(self.root / "foldbench-val"), ids(self.root / "foldbench-test")
        self.assertFalse(val & test)
        self.assertEqual(val | test, set(self.tracks))
        for pair in (("0001-assembly1", "0001-assembly2"), ("0002-assembly1", "0100-assembly1"),
                     ("0003-assembly1", "0101-assembly1")):
            self.assertEqual(pair[0] in val, pair[1] in val, pair)

    def test_validation_quota_per_track(self):
        report = split(self.root, self.root, "--validation-fraction", "0.25", "--max-per-track", "6")
        totals = {t: sum(1 for v in self.tracks.values() if v == t) for t in set(self.tracks.values())}
        for track, count in report["validation"]["assemblies_by_track"].items():
            self.assertLessEqual(count, min(6, math.ceil(0.25 * totals[track])))
            self.assertGreater(count, 0)

    def test_files_follow_their_assemblies(self):
        split(self.root, self.root)
        for name in ("foldbench-val", "foldbench-test"):
            part = self.root / name
            keep = ids(part)
            self.assertEqual({p.stem for p in (part / "ground_truths").glob("*.cif")}, keep)
            for table in (part / "suite/targets").glob("*.csv"):
                with table.open() as stream:
                    self.assertTrue({row["pdb_id"] for row in csv.DictReader(stream)} <= keep)
            rows = [json.loads(l) for l in (part / "features.esmc6b_final_int8.jsonl").read_text().splitlines()]
            for row in rows:
                self.assertTrue((part / row["path"]).is_file())
            manifest = json.loads((part / "suite/suite_manifest.json").read_text())
            self.assertEqual(manifest["assemblies"], len(keep))
            self.assertEqual(manifest["split"]["name"], "validation" if name.endswith("val") else "test")

    def test_deterministic(self):
        split(self.root, self.root / "a")
        split(self.root, self.root / "b")
        self.assertEqual(ids(self.root / "a/foldbench-val"), ids(self.root / "b/foldbench-val"))


if __name__ == "__main__":
    unittest.main()
