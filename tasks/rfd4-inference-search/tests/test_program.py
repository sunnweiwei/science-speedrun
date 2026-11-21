import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import gemmi
import zstandard

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment/evaluation"))
from program import collect_outputs, invoke_program
from reference import NAMES, POLICIES, historical_points, package_reference
from requests_io import suite_requests, write_monomer_request
from run import compare
from cost_curve import points as curve_points


class ProgramContractTest(unittest.TestCase):
    def test_publish_replaces_the_whole_package_and_preserves_it_on_failure(self):
        publish = TASK / "environment/baseline/publish.sh"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "submission"
            out.mkdir()
            candidate = root / "candidate"
            candidate.mkdir()
            (candidate / "infer.py").write_text("from helper import model\n")
            (candidate / "helper.py").write_text("model = 1\n")
            subprocess.run([publish, candidate, out], check=True)
            self.assertEqual({p.name for p in out.iterdir()}, {"infer.py", "helper.py"})
            (candidate / "helper.py").unlink()
            (candidate / "infer.py").write_text("model = 2\n")
            subprocess.run([publish, candidate, out], check=True)
            self.assertEqual([p.name for p in out.iterdir()], ["infer.py"])
            (candidate / "infer.py").unlink()
            (candidate / "policy.json").write_text("invalid")
            self.assertNotEqual(subprocess.run([publish, candidate, out], capture_output=True).returncode, 0)
            self.assertEqual((out / "infer.py").read_text(), "model = 2\n")
            legacy = TASK / "environment/baseline/stochastic-tsr"
            subprocess.run([publish, legacy, out], check=True)
            self.assertEqual([p.name for p in out.iterdir()], ["policy.json"])
            self.assertEqual((out / "policy.json").read_bytes(), (legacy / "policy.json").read_bytes())
            other = root / "notes"
            other.mkdir()
            (other / "experiment.md").write_text("working notes")
            self.assertNotEqual(subprocess.run([publish, legacy, other], capture_output=True).returncode, 0)
            self.assertEqual((other / "experiment.md").read_text(), "working notes")

    def test_unseen_request_and_slot_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = root / "suite.json"
            item = {"kind": "unconditional_monomer", "id": "future/molecule 7",
                    "input": {"length": 117}, "metadata": {"experiment": "held-out length"}}
            suite.write_text(json.dumps({"version": 1, "requests": [item]}))
            cases = suite_requests(suite)
            assets = root / "assets"
            (assets / "monomers").mkdir(parents=True)
            tags = ("_atom_site.label_comp_id", "_atom_site.auth_comp_id", "_entity_poly_seq.mon_id",
                    "_chem_comp.id", "_chem_comp.name", "_rfd_design.sampled_contig", "_chem_comp_atom.comp_id")
            template = "data_short\n" + "".join(f"{tag} 100-100\n" for tag in tags)
            (assets / "monomers/monomer-short.cif.zst").write_bytes(zstandard.ZstdCompressor().compress(template.encode()))
            request = write_monomer_request(root / "input", assets, 117, 3, 7, 108, cases[117])
            data = json.loads(request.read_bytes())
            self.assertEqual((data["id"], data["metadata"], data["num_outputs"]), (item["id"], item["metadata"], 3))
            blob = (request.parent / data["input"]["file"]).read_bytes()
            block = gemmi.cif.read_string(zstandard.ZstdDecompressor().decompress(blob).decode()).sole_block()
            self.assertEqual(block.find_value("_rfd_design.sampled_contig"), "117-117")

            package, output = root / "submission", root / "output"
            package.mkdir()
            output.mkdir()
            (package / "helper.py").write_text("def process(n):\n    return n + 3\n")
            (package / "infer.py").write_text(
                "import argparse,json,pathlib\nfrom helper import process\n"
                "p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--output'); a=p.parse_args()\n"
                "r=json.loads(pathlib.Path(a.request).read_text())\n"
                "(pathlib.Path(a.output)/'slot-000001.cif').write_text(str(process(r['seed'])))\n")
            timing = invoke_program(package, request, output, assets, 5, root / "run.log", private=False)
            self.assertEqual(timing["program_exit_code"], 0)
            (output / "slot-000002.cif").symlink_to(suite)
            paths = collect_outputs(output, 3, 117, root / "accepted")
            self.assertEqual(paths, [None, root / "accepted/slot-000001.cif", None])
            self.assertEqual(paths[1].read_text(), "10")

    def test_timeout_and_unsupported_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = root / "request.json"
            request.write_text(json.dumps({"version": 1, "kind": "unimplemented_kind"}))
            timing = invoke_program(TASK / "environment/baseline/program", request, root, root,
                                    5, root / "unsupported.log", private=False)
            self.assertEqual(timing["program_exit_code"], 64)
            (root / "infer.py").write_text("import time\ntime.sleep(30)\n")
            timing = invoke_program(root, request, root, root, .1, root / "timeout.log", private=False)
            self.assertTrue(timing["timed_out"])

    def test_references_and_measured_tradeoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in NAMES:
                package = package_reference(name, Path(tmp) / name)
                if name in POLICIES:
                    self.assertEqual((package / "policy.json").read_bytes(),
                                     (TASK / "environment/baseline" / name / "policy.json").read_bytes())
                else:
                    self.assertEqual(json.loads((package / "reference-preset.json").read_text()),
                                     name.removeprefix("native-"))
                self.assertTrue((package / "infer.py").is_file())
        points = historical_points([100, 150, 250], 50)
        self.assertEqual([points[name]["passing_clusters"] for name in POLICIES], [82, 71, 46])
        self.assertEqual(historical_points([117], 50), {})
        ref = {"reward": .5, "total_gpu_seconds": 100, "clusters_per_allocated_gpu_hour": 90,
               "by_length": {100: {"slots": 10, "refolded": 10, "timed_out": False}}}
        self.assertEqual(compare(ref, ref)["dominates_reference"], 0)
        self.assertEqual(compare(dict(ref, reward=.6, total_gpu_seconds=110), ref)["dominates_reference"], 0)
        self.assertEqual(compare(dict(ref, reward=.6, total_gpu_seconds=90), ref)["dominates_reference"], 1)

    def test_cost_curve_uses_matching_requests_and_device(self):
        point = {"reward": .5, "total_gpu_seconds": 100, "generation_attempts_per_gpu_hour": 400,
                 "attempts_per_gpu_hour": 360, "clusters_per_allocated_gpu_hour": 180}
        report = dict(point, protocol="rfd4-program-v1", workload_sha256="demo", device={"uuid": "first"},
                      submission_sha256="candidate", reference_sha256="baseline",
                      seconds_per_slot=36, reference_name="stochastic-tsr", reference=dict(point, reward=.4),
                      comparison={"program_complete": 1, "reference_complete": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.json"
            path.write_text(json.dumps(report))
            self.assertEqual([r["on_observed_frontier"] for r in curve_points([path])], [True, False])
            second = Path(tmp) / "two.json"
            second.write_text(json.dumps(dict(report, workload_sha256="different")))
            with self.assertRaises(ValueError):
                curve_points([path, second])
