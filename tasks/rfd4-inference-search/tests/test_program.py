import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

import gemmi
import zstandard

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / "environment/evaluation"))
sys.path.insert(0, str(TASK / "environment/baseline/program"))
sys.path.insert(0, str(TASK / "environment/references/overall"))
sys.path.insert(0, str(TASK / "tests"))
from budget_scorecard import build as build_budget_scorecard
from infer import detect_s8_schedule
from program import collect_outputs, invoke_program
from reference import NAMES, POLICIES, historical_points, package_reference
from requests_io import DEV_4TRACK_TARGETS, conditioned_suite_requests, suite_requests, write_conditioned_request, write_monomer_request
from rfd4_private import VERIFY_4TRACK_TARGETS, four_track_seeds
from run import compare
from cost_curve import points as curve_points
from evaluate_tracks import (
    PAIRED_TIMING_PROFILE,
    TRACK_SPECS,
    WARM_CAMPAIGN_PROFILE,
    aggregate_repeat_summaries,
    four_track_harbor_metrics,
    load_archive_context_for_targets,
    pareto_summary_across_tracks,
    parse_references,
    render_paired_pareto_plot,
    resolve_gpus,
    resolve_target_roster,
    summarize_archive_200_quality,
    summarize_paired_baselines,
    write_paired_points_csv,
)
from benchmark_curve import Measurement


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

    def test_four_track_conditioned_contract_and_strict_pareto_grader(self):
        manifest = json.loads((TASK / "environment/data/figure2_benchmarks.json").read_text())
        self.assertEqual(len(manifest["targets"]), 141)
        dev_roster = resolve_target_roster("dev")
        verify_roster = resolve_target_roster("verify")
        full_roster = resolve_target_roster("full")
        self.assertEqual(set(dev_roster), set(TRACK_SPECS))
        self.assertEqual(set(verify_roster), set(TRACK_SPECS))
        self.assertEqual(sum(len(v) for v in dev_roster.values()), 12)
        self.assertEqual(sum(len(v) for v in verify_roster.values()), 16)
        self.assertEqual([len(verify_roster[ds]) for ds in TRACK_SPECS], [4, 5, 4, 3])
        self.assertEqual(sum(len(v) for v in full_roster.values()), 141)
        # Held-out verify targets are disjoint from dev targets on all multi-target suites.
        for ds in ("ppi_backbone", "studio_figure_2c_unfixed", "mcsa_fixed"):
            self.assertTrue(set(dev_roster[ds]).isdisjoint(verify_roster[ds]))
        for ds, tgts in full_roster.items():
            for t in tgts:
                self.assertIn(f"{ds}/{t}", manifest["targets"])
        for ds, tgts in verify_roster.items():
            for t in tgts:
                self.assertIn(f"{ds}/{t}", manifest["targets"])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dummy_cif = root / "rfd3_pdl1.cif.zst"
            dummy_cif.write_bytes(b"cif-payload")
            req_path = write_conditioned_request(root / "req", dummy_cif, "ppi_backbone/rfd3_pdl1", 4, 727201, 120.0)
            req = json.loads(req_path.read_text())
            self.assertEqual(req["kind"], "conditioned_design")
            self.assertEqual(req["input"], {"file": "rfd3_pdl1.cif.zst"})
            self.assertEqual(req["num_outputs"], 4)

            # Custom four-track suite parsing and private HMAC seed derivation
            suite_file = root / "four_track_suite.json"
            suite_file.write_text(json.dumps({"version": 1, "targets": verify_roster}))
            self.assertEqual(conditioned_suite_requests(suite_file), verify_roster)
            seed_file = root / "seed"
            seed_file.write_bytes(b"\x17" * 32)
            gen_off, af3_s, digest = four_track_seeds(seed_file)
            self.assertTrue(0 <= gen_off < 2**29 and 0 <= af3_s < 2**31 and len(digest) == 64)

        # Build matched 20-attempt paired Low and High baselines across all 4 tracks
        paired_refs = {"native-low": {}, "native-high": {}}
        for ds, tgts in dev_roster.items():
            track = TRACK_SPECS[ds]["track"]
            paired_refs["native-low"][track] = {t: Measurement(20, 20, 4, 2, 40.0, 30.0) for t in tgts}   # 10% clustered, fast
            paired_refs["native-high"][track] = {t: Measurement(20, 20, 6, 4, 90.0, 30.0) for t in tgts}  # 20% clustered, slower
        paired_pts = summarize_paired_baselines(paired_refs, timing_profile=PAIRED_TIMING_PROFILE)

        # 1. Exact clone of Low is weakly nondominated (on_total_pareto=True) MUST NOT strictly advance Pareto!
        clone_low = {
            TRACK_SPECS[ds]["track"]: {t: Measurement(20, 20, 4, 2, 40.0, 30.0) for t in tgts}
            for ds, tgts in dev_roster.items()
        }
        clone_summary = pareto_summary_across_tracks(clone_low, paired_pts, method_name="clone-of-low")
        self.assertEqual(clone_summary["tracks_on_total_pareto"], 4)
        self.assertEqual(clone_summary["tracks_strictly_advancing_total_pareto"], 0)
        self.assertEqual(clone_summary["tracks_dominating_any_reference_total"], 0)
        self.assertEqual(clone_summary["tracks_dominating_strongest_reference_total"], 0)
        self.assertEqual(clone_summary["macro_total_hypervolume_gain_pct"], 0.0)
        self.assertEqual(clone_summary["paired_comparisons"]["Low"]["dominates_reference_macro"], 0)

        # 2. Trivial fast sub-Low-quality emitter MUST NOT count as advancing the frontier!
        fast_bad = {
            TRACK_SPECS[ds]["track"]: {t: Measurement(20, 20, 1, 1, 5.0, 10.0) for t in tgts}  # 5% < 10% Low floor
            for ds, tgts in dev_roster.items()
        }
        bad_summary = pareto_summary_across_tracks(fast_bad, paired_pts, method_name="fast-degraded")
        self.assertEqual(bad_summary["tracks_strictly_advancing_total_pareto"], 0)
        self.assertEqual(bad_summary["macro_total_hypervolume_gain_pct"], 0.0)

        # 3. Genuine frontier-advancing strategy beating High at faster speed strictly advances & dominates!
        better = {
            TRACK_SPECS[ds]["track"]: {t: Measurement(20, 20, 8, 5, 70.0, 30.0) for t in tgts}  # 25% > 20% High, 100s < 120s High
            for ds, tgts in dev_roster.items()
        }
        better_summary = pareto_summary_across_tracks(better, paired_pts, method_name="fast-search-v1")
        self.assertTrue(better_summary["all_four_tracks_evaluated"])
        self.assertEqual(better_summary["tracks_strictly_advancing_total_pareto"], 4)
        self.assertEqual(better_summary["tracks_dominating_any_reference_total"], 4)
        self.assertEqual(better_summary["tracks_dominating_strongest_reference_total"], 4)
        self.assertGreater(better_summary["macro_total_hypervolume_gain_pct"], 0.0)
        self.assertAlmostEqual(better_summary["reward"], 0.25)

        # 4. Reject comparing 20-attempt candidates against 200-attempt archive baselines or mismatched timing boundaries!
        archive_ctx = load_archive_context_for_targets(dev_roster)
        with self.assertRaises(ValueError):
            pareto_summary_across_tracks(better, archive_ctx, method_name="cross-k-illegal")
        mismatched_boundary = {
            tr: [dict(b, timing_profile=WARM_CAMPAIGN_PROFILE) for b in bpts]
            for tr, bpts in paired_pts.items()
        }
        with self.assertRaises(ValueError):
            pareto_summary_across_tracks(better, mismatched_boundary, method_name="cross-boundary-illegal")

    def test_multi_repeat_sd_harbor_reward_and_s8_schedule_detector(self):
        dev_roster = resolve_target_roster("dev")
        rep_summaries = []
        for r_idx, extra_clusters in enumerate((4, 5, 6)):
            paired_refs = {"native-low": {}, "native-high": {}}
            cand = {}
            for ds, tgts in dev_roster.items():
                track = TRACK_SPECS[ds]["track"]
                paired_refs["native-low"][track] = {t: Measurement(20, 20, 4, 2, 40.0 + r_idx, 30.0) for t in tgts}
                paired_refs["native-high"][track] = {t: Measurement(20, 20, 6, 4, 90.0 + r_idx, 30.0) for t in tgts}
                cand[track] = {t: Measurement(20, 20, 8, extra_clusters, 75.0 + r_idx, 30.0) for t in tgts}
            rep_summaries.append(
                pareto_summary_across_tracks(
                    cand,
                    summarize_paired_baselines(paired_refs, timing_profile=PAIRED_TIMING_PROFILE),
                    method_name="cand-repeats",
                )
            )
        agg = aggregate_repeat_summaries(rep_summaries)
        self.assertEqual(agg["repeats"], 3)
        self.assertAlmostEqual(agg["macro_clustered_pct"], 25.0)
        self.assertAlmostEqual(agg["macro_clustered_pct_sd"], 5.0)
        self.assertEqual(agg["tracks_strictly_advancing_total_pareto"], 4)

        flat = four_track_harbor_metrics(agg)
        self.assertAlmostEqual(flat["reward"], 0.25)
        self.assertAlmostEqual(flat["macro_clustered_pct_sd"], 5.0)
        self.assertEqual(flat["tracks_strictly_advancing_total_pareto"], 4)
        self.assertTrue(all(type(v) in (int, float) for v in flat.values()))
        self.assertEqual(four_track_harbor_metrics({"reward": 0.0, "invalid_submission": "bad"}),
                         {"reward": 0.0, "invalid_submission": 1})

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plot_rows = agg["plot_rows"]
            write_paired_points_csv(plot_rows, root / "points.csv")
            with (root / "points.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 12)
            render_paired_pareto_plot(
                plot_rows, root / "pareto_including_af3.png", include_af3=True,
                evaluated_tracks=agg["evaluated_tracks"], attempts_per_target=20, repeats=3, panel_name="verify",
            )
            self.assertTrue((root / "pareto_including_af3.png").is_file())

            # Verify chemical-composition S8.3 schedule detector across all four modalities (including expandable HETATM ranges)
            ppi_cif = root / "ppi.cif"
            ppi_cif.write_text("data_x\nloop_\n_atom_site.group_PDB\n_atom_site.label_comp_id\nATOM ALA\nHETATM 50-120\n")
            sm_cif = root / "sm.cif"
            sm_cif.write_text("data_x\nloop_\n_atom_site.group_PDB\n_atom_site.label_comp_id\nATOM GLY\nHETATM LIG\nHETATM 80-110\n")
            dna_cif = root / "dna.cif.zst"
            dna_cif.write_bytes(zstandard.ZstdCompressor().compress(
                b"data_x\nloop_\n_atom_site.group_PDB\n_atom_site.label_comp_id\nATOM DA\nATOM DT\nHETATM 60-90\n"
            ))
            for lvl in ("low", "medium", "high"):
                self.assertEqual(detect_s8_schedule(ppi_cif, lvl), f"{lvl}/front_loaded")
                self.assertEqual(detect_s8_schedule(sm_cif, lvl), f"{lvl}/midpeak")
            self.assertEqual(detect_s8_schedule(dna_cif, "low"), "low/front_loaded")
            self.assertEqual(detect_s8_schedule(dna_cif, "medium"), "medium/front_loaded")
            self.assertEqual(detect_s8_schedule(dna_cif, "high"), "high/flat")

    def test_official_verifier_and_task_metadata_target_four_tracks(self):
        meta = tomllib.loads((TASK / "task.toml").read_text())
        self.assertEqual(meta["metadata"]["primary_metric"], "four_track_mean_clustered_pass_rate")
        self.assertEqual(meta["metadata"]["budget_curve_protocol"], "rfd4-budget-curve-v1")
        self.assertEqual(meta["metadata"]["budget_curve_seconds_per_slot"], [2.5, 6.25, 25])
        self.assertIn("Four-track", meta["metadata"]["live_score_scope"])
        self.assertEqual(meta["environment"]["gpus"], 8)
        self.assertEqual(meta["verifier"]["environment"]["gpus"], 8)
        self.assertEqual(meta["verifier"]["timeout_sec"], 14400.0)
        self.assertNotIn("gpu_types", meta["environment"])
        self.assertNotIn("gpu_types", meta["verifier"]["environment"])
        test_sh = (TASK / "tests/test.sh").read_text()
        self.assertIn("/workspace/evaluation/evaluate_tracks.py", test_sh)
        self.assertIn("${RFD4_REFERENCES:-none}", test_sh)
        self.assertIn("${RFD4_VERIFY_ATTEMPTS:-200}", test_sh)
        self.assertIn("${RFD4_SECONDS_PER_SLOT:-25}", test_sh)
        self.assertIn("--reward-output", test_sh)
        self.assertIn("/logs/verifier/report.json", test_sh)
        self.assertIn("native-medium", NAMES)
        self.assertEqual(parse_references(["none"]), ())

        # Verify candidate-only K=200 verify summary and target-matched archive quality deltas.
        verify_roster = resolve_target_roster("verify")
        cand_200 = {
            TRACK_SPECS[ds]["track"]: {t: Measurement(200, 200, 50, 40, 2400.0, 460.0) for t in tgts}
            for ds, tgts in verify_roster.items()
        }
        cand_only = pareto_summary_across_tracks(cand_200, {}, method_name="search-200")
        self.assertAlmostEqual(cand_only["reward"], 0.20)
        self.assertEqual(cand_only["paired_comparisons"], {})
        archive_ctx = load_archive_context_for_targets(verify_roster)
        self.assertEqual(summarize_archive_200_quality(cand_only, archive_ctx, attempts_per_target=20), {})
        arch_q = summarize_archive_200_quality(cand_only, archive_ctx, attempts_per_target=200)
        self.assertEqual(set(arch_q), {"Low", "Medium", "High", "Search"})
        self.assertAlmostEqual(arch_q["Search"]["archive_macro_clustered_pct"], 14.25, places=1)
        self.assertAlmostEqual(arch_q["Search"]["quality_delta_pp"], 20.0 - arch_q["Search"]["archive_macro_clustered_pct"], places=4)
        cand_only["archive_200attempt_quality_comparisons"] = arch_q
        flat = four_track_harbor_metrics(cand_only)
        self.assertAlmostEqual(flat["reward"], 0.20)
        self.assertAlmostEqual(flat["archive_200_search_archive_macro_clustered_pct"], arch_q["Search"]["archive_macro_clustered_pct"])
        self.assertAlmostEqual(flat["archive_200_search_quality_delta_pp"], arch_q["Search"]["quality_delta_pp"])
        self.assertTrue(all(type(v) in (int, float) for v in flat.values()))
        with tempfile.TemporaryDirectory() as tmp:
            render_paired_pareto_plot(
                cand_only["plot_rows"], Path(tmp) / "cand_only.png", include_af3=True,
                evaluated_tracks=cand_only["evaluated_tracks"], attempts_per_target=200, repeats=1, panel_name="verify",
            )
            self.assertTrue((Path(tmp) / "cand_only.png").is_file())

    def test_multi_gpu_resolution_and_budget_scorecard_compatibility(self):
        fake_gpus = [
            {"index": i, "name": "NVIDIA RTX A6000", "uuid": f"GPU-{i:04d}", "driver": "550.0", "device_file": f"/dev/nvidia{i}"}
            for i in range(8)
        ]
        self.assertEqual(len(resolve_gpus("auto", available=fake_gpus)), 8)
        self.assertEqual(len(resolve_gpus("single", available=fake_gpus)), 1)
        self.assertEqual([g["index"] for g in resolve_gpus("0,3,7", available=fake_gpus)], [0, 3, 7])
        with self.assertRaises(ValueError):
            resolve_gpus("9", available=fake_gpus)

        # Verify evaluate_tracks.py summaries integrate natively with budget_scorecard.py as status="complete"
        dev_roster = resolve_target_roster("dev")
        trial_fields = ("repeat", "dataset", "criterion", "method", "target", "attempt_id",
                        "scored", "passed", "cluster_id", "generation_gpu_s", "af3_gpu_s")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tier_dirs = {}
            for tier, cap, c_hits in (("tenth", 6.0, 2), ("quarter", 15.0, 3), ("full", 60.0, 4)):
                tdir = root / tier
                tdir.mkdir()
                tier_dirs[tier] = tdir
                cand_by_track = {}
                refs_by_name = {"native-low": {}}
                trials = []
                for ds, tgts in dev_roster.items():
                    spec = TRACK_SPECS[ds]
                    track = spec["track"]
                    cand_by_track[track] = {}
                    refs_by_name["native-low"][track] = {}
                    for tgt in tgts:
                        cand_by_track[track][tgt] = Measurement(4, 4, c_hits, c_hits, 4 * (cap * 0.5), 12.0)
                        refs_by_name["native-low"][track][tgt] = Measurement(4, 4, 1, 1, 4 * (cap * 0.5), 12.0)
                        for label, hits in (("candidate-x", c_hits), ("Low", 1)):
                            for idx in range(4):
                                ok = int(idx < hits)
                                trials.append({
                                    "repeat": 0, "dataset": ds, "criterion": spec["criterion"], "method": label,
                                    "target": tgt, "attempt_id": f"slot{idx:04d}", "scored": 1, "passed": ok,
                                    "cluster_id": f"c{idx}" if ok else "", "generation_gpu_s": cap * 0.5, "af3_gpu_s": 3.0,
                                })
                trials_path = tdir / "trials.csv"
                with trials_path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=trial_fields, lineterminator="\n")
                    w.writeheader()
                    w.writerows(trials)
                paired_pts = summarize_paired_baselines(refs_by_name, timing_profile=PAIRED_TIMING_PROFILE)
                summary = pareto_summary_across_tracks(cand_by_track, paired_pts, method_name="candidate-x")
                summary.pop("plot_rows", None)
                summary.update({
                    "panel": "dev", "attempts_per_target": 4, "repeats": 1, "seconds_per_slot": cap,
                    "paired_references": ["native-low"], "reference_sha256": {"native-low": "b" * 64},
                    "submission_sha256": "a" * 64, "suite_sha256": "c" * 64,
                    "device": {"name": "NVIDIA RTX A6000", "uuid": "GPU-0000", "driver": "550.0", "gpu_count": 8},
                    "generation_timing_boundary": "per_request_infer_subprocess_wall_clock",
                    "af3_timing_boundary": "shared_persistent_af3_server_warm_fold",
                    "trials_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
                })
                (tdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            _, scorecard = build_budget_scorecard(tier_dirs, Decimal("60"))
            self.assertEqual(scorecard["status"], "complete")
            self.assertAlmostEqual(scorecard["primary_full_budget_quality"], 1.0)
            self.assertAlmostEqual(scorecard["macro_low_budget_quality_pct"], 62.5)
