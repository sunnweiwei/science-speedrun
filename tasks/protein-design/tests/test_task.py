"""Interface and metric tests for the protein-design task.

These use synthetic all-atom structures and a stubbed designability scorer. They
validate the contract, the harness manifest, the geometry and conditioning gates, and
the per-track pass-rate accounting. They do NOT establish a benchmark result and do not
exercise real LigandMPNN or RoseTTAFold3 weights.
"""

import json
import shutil
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

import numpy as np

import grader
from common import (CLASH_MIN, HARNESS_SCHEMA, HOTSPOT_CONTACT_MAX, LIGAND_CLASH_MIN,
                    LIGAND_CONTACT_MAX, MOTIF_RMSD_MAX, TRACKS, sha256,
                    validate_harness, validate_input, write_json)
from pipeline import ScorerUnavailable
from geometry import (backbone_aligned_motif_rmsd, minimum_distance, present_atoms,
                      write_allatom_pdb)
from residues import ATOM14, ATOM14_WIDTH

MOTIF_RESIDUES = ["SER", "HIS", "ASP"]
# A catalytic triad is scattered along the sequence, not consecutive. Using adjacent
# residues would make the motif trivially reproducible: any three consecutive residues
# of an ideal helix superimpose on any other three.
MOTIF_POSITIONS = (3, 15, 27)


def stable_seed(text, offset=0):
    """Deterministic across processes; Python's str hash is salted per interpreter."""
    return zlib.crc32(text.encode()) % 9973 + offset


def build_structure(trace, restypes=None):
    """Turn a CA trace into atom14 coords + mask with plausible sidechain placement.

    CB comes from the standard tetrahedral construction on N, CA, C, which is always
    well defined; remaining sidechain atoms sit on a small circle around CB. Projecting
    "outward from the chain centroid" instead degenerates near the chain ends, where
    that direction is nearly parallel to the chain, and folds sidechains into the
    backbone.
    """
    length = trace.shape[0]
    restypes = list(restypes) if restypes else ["GLY"] * length
    coordinates = np.zeros((length, ATOM14_WIDTH, 3))
    mask = np.zeros((length, ATOM14_WIDTH), dtype=bool)
    for index in range(length):
        alpha = trace[index]
        forward = trace[min(index + 1, length - 1)] - trace[max(index - 1, 0)]
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        helper = np.array([0.0, 0.0, 1.0])
        if abs(forward @ helper) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        side = np.cross(forward, helper)
        side /= np.linalg.norm(side) + 1e-9
        nitrogen = alpha - 1.20 * forward + 0.50 * side
        carbon = alpha + 1.20 * forward + 0.50 * side
        coordinates[index, 0] = nitrogen
        coordinates[index, 1] = alpha
        coordinates[index, 2] = carbon
        coordinates[index, 3] = carbon + 0.30 * forward + 1.05 * side
        mask[index, :4] = True

        atoms = ATOM14[restypes[index]]
        if len(atoms) <= 4:
            continue
        b = alpha - nitrogen
        c = carbon - alpha
        a = np.cross(b, c)
        beta = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + alpha
        coordinates[index, 4] = beta
        mask[index, 4] = True
        outward = beta - alpha
        outward /= np.linalg.norm(outward) + 1e-9
        perpendicular = np.cross(outward, forward)
        perpendicular /= np.linalg.norm(perpendicular) + 1e-9
        second = np.cross(outward, perpendicular)
        for slot in range(5, len(atoms)):
            angle = 2 * np.pi * (slot - 5) / max(1, len(atoms) - 5)
            coordinates[index, slot] = (beta + 0.9 * outward
                                        + 1.1 * (np.cos(angle) * perpendicular
                                                 + np.sin(angle) * second))
            mask[index, slot] = True
    return coordinates, mask, restypes


def helix(length, origin=(0.0, 0.0, 0.0), seed=0):
    """CA trace of an ideal alpha helix with exact 3.80 A spacing."""
    trace = np.zeros((length, 3))
    for index in range(length):
        angle = index * 100.0 * np.pi / 180.0
        trace[index] = [2.3 * np.cos(angle), 2.3 * np.sin(angle), index * 1.5]
    trace *= 3.80 / np.linalg.norm(trace[1] - trace[0])
    rotation, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    return trace @ rotation.T + np.asarray(origin)


def random_compact_chain(length, seed=0):
    """Self-avoiding chain at exact 3.80 A spacing; distinct seeds give distinct folds."""
    rng = np.random.default_rng(seed)
    trace = [np.zeros(3), np.array([3.80, 0.0, 0.0])]
    while len(trace) < length:
        current = trace[-1]
        inward = -current / (np.linalg.norm(current) + 1e-9)
        previous = current - trace[-2]
        previous /= np.linalg.norm(previous) + 1e-9
        for _ in range(400):
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            step = 0.45 * previous + 0.35 * inward + 0.60 * step
            step /= np.linalg.norm(step)
            candidate = current + 3.80 * step
            # 6.5 A between non-adjacent CA keeps their backbone O atoms, which sit
            # roughly 1.9 A off the CA, clear of the 2.0 A all-atom clash threshold.
            if all(np.linalg.norm(candidate - point) >= 6.5 for point in trace[:-2]):
                trace.append(candidate)
                break
        else:
            return random_compact_chain(length, seed + 1000)
    return np.asarray(trace)


def bridge_designs(length, scale=1.5, seed=3):
    """Three structures whose greedy cluster count depends on the order they are seen in.

    B and C are opposite displacements of the same chain, so each is within
    DIVERSITY_RMSD of A while being about twice that far from each other. Seeing A first
    absorbs both (1 cluster); seeing B first makes C a second centre and then absorbs A
    (2 clusters). Returned A-first so a caller cycling through them exercises both.
    """
    base = random_compact_chain(length, seed=seed)
    offset = np.random.default_rng(seed + 100).normal(size=base.shape)
    offset /= np.sqrt((offset ** 2).sum(axis=1).mean())      # unit mean-square displacement
    return [build_structure(trace) for trace in
            (base, base + scale * offset, base - scale * offset)]


def dock_near_hotspots(structure, target, target_mask, hotspots):
    """Translate a binder so it contacts hotspots without clashing into the target."""
    coordinates, mask, restypes = structure
    target_atoms = target[target_mask]
    centre_hot = target[list(hotspots)][target_mask[list(hotspots)]].mean(axis=0)
    primary = centre_hot - target_atoms.mean(axis=0)
    primary /= np.linalg.norm(primary) + 1e-9
    rng = np.random.default_rng(0)
    centre = coordinates[:, 1].mean(axis=0)
    for direction in [primary] + [primary + 0.55 * rng.normal(size=3) for _ in range(24)]:
        direction = direction / (np.linalg.norm(direction) + 1e-9)
        for distance in np.arange(5.0, 34.0, 0.25):
            moved = coordinates - centre + centre_hot + direction * distance
            atoms = moved[mask]
            if minimum_distance(atoms, target_atoms) < 2.4:
                continue
            touched = sum(minimum_distance(atoms, target[i][target_mask[i]]) <= 5.0
                          for i in hotspots)
            if touched / len(hotspots) >= 0.5:
                return (moved, mask, restypes)
    raise AssertionError("fixture could not dock a binder against the hotspots")


class StubScorer:
    """Deterministic stand-in for LigandMPNN + RoseTTAFold3.

    Passing `passed=` pins the verdict, for tests about pass-rate accounting rather than
    about the criterion.

    The default models what the PREDICTOR would say, geometrically. Every published
    criterion is a statement about a refolded complex - motif recapitulation,
    backbone-aligned ligand RMSD, ipTM, interface min PAE - and they share one
    consequence: a design that cannot fold to itself, or that does not engage the
    conditioning it was given, cannot satisfy them. `grader.score_track` deliberately
    adds no gate of its own, so these belong here, standing in for the predictor. A stub
    that returned `passed` unconditionally would report every degenerate submission as
    designable, and the anti-degeneracy tests below would prove nothing about the task.
    """

    # Non-adjacent CA atoms do not come this close in a real fold, so a design that
    # violates it would never be refolded into itself. Stands in for sc_RMSD. Chosen
    # below the ~5.0 A i/i+3 spacing of an ideal helix so the fixtures remain valid.
    MIN_CA_SEPARATION = 4.0

    def __init__(self, passed=None, sc_rmsd=1.0, iptm=0.9, interface_pae=1.0,
                 ligand_rmsd=2.0, motif_rmsd=0.8, devices=(None,)):
        # `devices` is how many designs grader.fold_designs scores at once, mirroring
        # DesignabilityScorer.devices.
        self.devices = list(devices)
        self._lock = threading.Lock()
        self.pinned = passed
        self.verdict = {"passed": True, "sc_rmsd": sc_rmsd, "iptm": iptm,
                        "interface_pae": interface_pae, "ligand_rmsd": ligand_rmsd,
                        "motif_rmsd": motif_rmsd, "sequences_tried": 1, "reason": None}
        self.calls = 0

    def evaluate(self, track, structure, context, motif_map=None):
        with self._lock:
            self.calls += 1
        verdict = dict(self.verdict)
        if self.pinned is not None:
            verdict["passed"] = self.pinned
            verdict["reason"] = None if self.pinned else "stubbed failure"
            return verdict
        reason = self._would_refold(structure) or self._engages(
            track, structure, context, motif_map, verdict)
        verdict["passed"] = reason is None
        verdict["reason"] = reason
        return verdict

    def _would_refold(self, structure):
        """sc_RMSD stand-in: self-intersecting or non-globular chains do not refold."""
        alpha = structure[0][:, 1]
        length = alpha.shape[0]
        separations = np.linalg.norm(alpha[:, None, :] - alpha[None, :, :], axis=-1)
        indices = np.arange(length)
        far = np.abs(indices[:, None] - indices[None, :]) > 2
        if far.any() and separations[far].min() < self.MIN_CA_SEPARATION:
            return f"sc_RMSD: chain self-intersects at {separations[far].min():.2f} A"
        radius = float(np.sqrt(((alpha - alpha.mean(axis=0)) ** 2).sum(axis=1).mean()))
        if radius > 3.0 * (2.2 * length ** 0.38):
            return f"sc_RMSD: radius of gyration {radius:.1f} A is not globular"
        return None

    def _engages(self, track, structure, context, motif_map, verdict):
        """The conditioning half: motif recapitulation, ligand RMSD, ipTM and PAE."""
        coordinates, mask, restypes = structure
        atoms = present_atoms(coordinates, mask)
        if track == "enzyme":
            motif_mask = context["motif_mask"].astype(bool)
            verdict["motif_rmsd"] = backbone_aligned_motif_rmsd(
                coordinates[list(motif_map)], mask[list(motif_map)],
                context["motif_coords"], motif_mask,
                [str(name) for name in context["motif_restypes"]])
            if verdict["motif_rmsd"] > MOTIF_RMSD_MAX:
                return f"motif recapitulation {verdict['motif_rmsd']:.2f} A"
        if track in ("enzyme", "sm_binder"):
            ligand = context["ligand_coords"]
            separation = minimum_distance(atoms, ligand)
            if separation < LIGAND_CLASH_MIN:
                return f"ligand RMSD: design overlaps the ligand at {separation:.2f} A"
            if separation > LIGAND_CONTACT_MAX:
                return f"ipTM: nearest ligand atom is {separation:.2f} A away"
        if track == "ppi_binder":
            target_mask = context["target_mask"].astype(bool)
            target = context["target_coords"]
            separation = minimum_distance(atoms, target[target_mask])
            if separation < CLASH_MIN:
                return f"interface PAE: binder overlaps the target at {separation:.2f} A"
            contacted = sum(
                minimum_distance(atoms, target[index][target_mask[index]])
                <= HOTSPOT_CONTACT_MAX
                for index in (int(v) for v in context["hotspot_indices"].ravel()))
            if not contacted:
                return "interface PAE: binder contacts no hotspot"
        return None


class Fixture:
    """Builds a tiny four-track evaluation view plus a matching submission."""

    LENGTH = 32
    DESIGNS = 4
    MOTIF = len(MOTIF_RESIDUES)

    def __init__(self, root):
        self.root = Path(root)
        self.input = self.root / "input"
        self.snapshot = self.root / "snapshot"
        (self.input / "contexts").mkdir(parents=True)
        (self.snapshot / "structures").mkdir(parents=True)
        self.reference = build_structure(helix(self.LENGTH, seed=1),
                                         self.residue_types())
        self._write_input()

    @classmethod
    def residue_types(cls, positions=MOTIF_POSITIONS):
        types = ["GLY"] * cls.LENGTH
        for slot, position in enumerate(positions):
            types[position] = MOTIF_RESIDUES[slot]
        return types

    def _context(self, track, example_id):
        path = self.input / "contexts" / f"{track}_{example_id}.npz"
        coordinates, mask, restypes = self.reference
        arrays = {"design_length": np.array([self.LENGTH])}
        # Seat the ligand against the reference surface: close enough to be engaged,
        # far enough not to clash. A ligand parked outside the bounding sphere would
        # fail the burial gate, which is exactly what that gate exists to catch.
        atoms = coordinates[mask]
        centre = atoms.mean(axis=0)
        middle = coordinates[self.LENGTH // 2, 1]
        outward = middle - centre
        outward = outward / (np.linalg.norm(outward) + 1e-9)
        seat = middle + outward * 4.2
        ligand = np.stack([seat, seat + outward * 0.9, seat + np.roll(outward, 1) * 0.9])
        if track == "enzyme":
            # An enzyme target is a catalytic motif plus its ligand.
            selection = list(MOTIF_POSITIONS)
            arrays["motif_coords"] = coordinates[selection]
            arrays["motif_mask"] = mask[selection]
            arrays["motif_restypes"] = np.array([restypes[i] for i in selection], dtype="<U3")
            arrays["ligand_coords"] = ligand
        if track == "sm_binder":
            # An sm_binder target is a bare ligand: no pocket template at all.
            arrays["ligand_coords"] = ligand
        if track == "ppi_binder":
            target = build_structure(helix(12, origin=(9.0, 0.0, 0.0), seed=2))
            arrays["target_coords"] = target[0]
            arrays["target_mask"] = target[1]
            arrays["hotspot_indices"] = np.array([0, 1, 2])
        np.savez(path, **arrays)
        return path

    def _write_input(self):
        files, counts = {}, {}
        for track in TRACKS:
            rows = []
            for number in range(2):
                example_id = f"{track}-{number:03d}"
                context = self._context(track, example_id)
                rows.append({
                    "example_id": example_id, "track": track,
                    "design_length": self.LENGTH, "num_designs": self.DESIGNS,
                    "context_path": str(context.relative_to(self.input)),
                    "context_sha256": sha256(context),
                    "motif_size": self.MOTIF if track == "enzyme" else 0,
                    "hotspot_count": 3 if track == "ppi_binder" else 0,
                    "ligand_atom_count": 3 if track in ("sm_binder", "enzyme") else 0,
                })
            path = self.input / f"{track}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            files[path.name] = sha256(path)
            counts[track] = len(rows)
        write_json(self.input / "manifest.json", {
            "schema": "protein-design-input/v1", "view": "verifier",
            "files": files, "counts": counts})

    def compliant_builder(self, track):
        _, requests = validate_input(self.input)
        if track == "unconditional":
            return lambda example, index: (
                build_structure(random_compact_chain(self.LENGTH,
                                                     stable_seed(example, index))), None)
        if track == "enzyme":
            return lambda example, index: (self.reference, list(MOTIF_POSITIONS))
        if track == "sm_binder":
            return lambda example, index: (self.reference, None)

        def ppi(example, index):
            context = np.load(self.input / requests[track][example]["context_path"])
            binder = build_structure(random_compact_chain(self.LENGTH,
                                                          stable_seed(example, index)))
            return dock_near_hotspots(binder, context["target_coords"],
                                      context["target_mask"].astype(bool),
                                      [int(v) for v in context["hotspot_indices"]]), None
        return ppi

    def submit(self, track, builder):
        _, requests = validate_input(self.input)
        rows = []
        for example_id in sorted(requests[track]):
            designs = []
            for index in range(self.DESIGNS):
                structure, motif_map = builder(example_id, index)
                if structure is None:
                    continue
                name = f"{example_id}-{index}.pdb"
                write_allatom_pdb(self.snapshot / "structures" / name, *structure)
                record = {"design_index": index, "ranking_score": 1.0 - 0.1 * index,
                          "structure_path": name}
                if motif_map is not None:
                    record["motif_map"] = motif_map
                designs.append(record)
            rows.append({"example_id": example_id, "designs": designs})
        path = self.snapshot / f"{track}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def score(self, track, path, scorer=None):
        _, requests = validate_input(self.input)
        return grader.score_track(track, path, requests[track], self.input,
                                  scorer or StubScorer())


class ProteinDesignTaskTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.fixture = Fixture(self.directory)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    # ---- input contract -------------------------------------------------
    def test_valid_input_view_parses(self):
        manifest, requests = validate_input(self.fixture.input)
        self.assertEqual(set(requests), set(TRACKS))
        self.assertEqual(manifest["counts"]["enzyme"], 2)

    def test_tampered_context_is_rejected(self):
        context = next((self.fixture.input / "contexts").glob("*.npz"))
        context.write_bytes(context.read_bytes() + b"\0")
        with self.assertRaisesRegex(ValueError, "context hash mismatch"):
            validate_input(self.fixture.input)

    def test_tampered_index_is_rejected(self):
        path = self.fixture.input / "enzyme.jsonl"
        path.write_text(path.read_text().replace("enzyme-000", "enzyme-999"))
        with self.assertRaisesRegex(ValueError, "input hash mismatch"):
            validate_input(self.fixture.input)

    # ---- happy path -----------------------------------------------------
    def test_compliant_designs_score_high(self):
        for track in TRACKS:
            path = self.fixture.submit(track, self.fixture.compliant_builder(track))
            result = self.fixture.score(track, path)
            self.assertIsNone(result["invalid_reason"], track)
            self.assertGreater(result["pass_rate"], 0.5, track)
            self.assertEqual(result["designs_designable"], self.fixture.DESIGNS * 2, track)

    # ---- novelty against the fit corpus ---------------------------------
    def test_near_copies_of_fit_structures_do_not_pass(self):
        """Designable designs at or above NOVELTY_MAX_TM to a fit chain fail; the rest still pass."""
        class Novelty:
            def max_tm(self, designs):
                return {key: (0.95 if key[1] % 2 == 0 else 0.4) for key in designs}
        _, requests = validate_input(self.fixture.input)
        path = self.fixture.submit("unconditional", self.fixture.compliant_builder("unconditional"))
        base = self.fixture.score("unconditional", path)
        gated = grader.score_track("unconditional", path, requests["unconditional"], self.fixture.input,
                                   StubScorer(), Novelty())
        self.assertEqual(gated["designs_designable"], base["designs_designable"])
        self.assertEqual(gated["designs_novel"], base["designs_designable"] // 2)
        self.assertAlmostEqual(gated["pass_rate"], base["pass_rate"] / 2)
        self.assertTrue(any("near-copy" in key for key in gated["failures"]))
        self.assertAlmostEqual(gated["novelty_max_tm_mean"], (0.95 + 0.4) / 2)

    def test_without_a_novelty_reference_scoring_is_unchanged(self):
        path = self.fixture.submit("unconditional", self.fixture.compliant_builder("unconditional"))
        result = self.fixture.score("unconditional", path)
        self.assertNotIn("designs_novel", result)
        self.assertEqual(result["pass_rate"] * self.fixture.DESIGNS * 2, result["designs_designable"])

    # ---- failure accounting ---------------------------------------------
    def test_empty_submission_scores_zero_not_void(self):
        for track in TRACKS:
            result = self.fixture.score(
                track, self.fixture.submit(track, lambda e, i: (None, None)))
            self.assertEqual(result["pass_rate"], 0.0, track)
            self.assertEqual(result["designs_requested"], self.fixture.DESIGNS * 2, track)

    def test_missing_designs_stay_in_the_denominator(self):
        motif = list(MOTIF_POSITIONS)
        path = self.fixture.submit(
            "enzyme", lambda e, i, m=motif: (self.fixture.reference, m) if i == 0
            else (None, None))
        result = self.fixture.score("enzyme", path)
        self.assertEqual(result["designs_requested"], 8)
        self.assertEqual(result["designs_returned"], 2)
        self.assertAlmostEqual(result["pass_rate"], 0.25)

    def test_one_bad_design_fails_without_voiding_the_track(self):
        motif = list(MOTIF_POSITIONS)

        def builder(example, index, m=motif):
            if index == 1:
                radius = 6.0
                step = 2 * np.arcsin(3.8 / (2 * radius))
                trace = np.array([[radius * np.cos(i * step), radius * np.sin(i * step), 0.0]
                                  for i in range(self.fixture.LENGTH)])
                return build_structure(trace, Fixture.residue_types()), m
            return self.fixture.reference, m

        result = self.fixture.score("enzyme", self.fixture.submit("enzyme", builder))
        self.assertIsNone(result["invalid_reason"])
        self.assertAlmostEqual(result["pass_rate"], 0.75)

    def test_malformed_index_voids_the_track(self):
        path = self.fixture.submit("enzyme", self.fixture.compliant_builder("enzyme"))
        path.write_text(path.read_text().replace("enzyme-000", "unknown-000"))
        result = self.fixture.score("enzyme", path)
        self.assertEqual(result["pass_rate"], 0.0)
        self.assertIn("unknown", result["invalid_reason"])

    def test_duplicate_example_id_voids_the_track(self):
        path = self.fixture.submit("enzyme", self.fixture.compliant_builder("enzyme"))
        path.write_text(path.read_text().replace("enzyme-001", "enzyme-000"))
        self.assertIn("duplicate", self.fixture.score("enzyme", path)["invalid_reason"])

    def test_structure_path_escape_is_rejected(self):
        path = self.fixture.submit("enzyme", self.fixture.compliant_builder("enzyme"))
        path.write_text(path.read_text().replace('"enzyme-000-0.pdb"', '"../../etc/passwd"'))
        self.assertLess(self.fixture.score("enzyme", path)["pass_rate"], 1.0)

    def test_undesignable_designs_score_zero(self):
        path = self.fixture.submit("enzyme", self.fixture.compliant_builder("enzyme"))
        result = self.fixture.score("enzyme", path, StubScorer(passed=False, sc_rmsd=9.0))
        self.assertEqual(result["pass_rate"], 0.0)
        self.assertEqual(result["conditioning_satisfied"], 8)

    # ---- conditioning gates ---------------------------------------------
    def test_ignoring_the_motif_fails_the_enzyme_track(self):
        # Right residues, wrong three-dimensional arrangement.
        wrong_positions = (5, 6, 7)
        types = Fixture.residue_types(wrong_positions)
        path = self.fixture.submit(
            "enzyme", lambda e, i, t=types, w=wrong_positions: (
                build_structure(helix(self.fixture.LENGTH, seed=i), t), list(w)))
        result = self.fixture.score("enzyme", path)
        self.assertEqual(result["pass_rate"], 0.0)
        # Rejected by the criterion now, not by a gate the grader added: the motif
        # sits at the wrong coordinates, so the refold cannot recapitulate it.
        self.assertTrue(any("motif recapitulation" in k for k in result["failures"]))

    def test_wrong_motif_residue_identity_fails(self):
        """All-atom scaffolding must place the requested residues, not just geometry."""
        wrong = ["GLY"] * self.fixture.LENGTH
        for position in MOTIF_POSITIONS:
            wrong[position] = "ALA"
        structure = build_structure(helix(self.fixture.LENGTH, seed=1), wrong)
        path = self.fixture.submit(
            "enzyme", lambda e, i, s=structure: (s, list(MOTIF_POSITIONS)))
        result = self.fixture.score("enzyme", path)
        self.assertEqual(result["pass_rate"], 0.0)
        self.assertTrue(any("identities differ" in k for k in result["failures"]))

    def test_missing_or_malformed_motif_map_fails(self):
        path = self.fixture.submit("enzyme", lambda e, i: (self.fixture.reference, None))
        self.assertEqual(self.fixture.score("enzyme", path)["pass_rate"], 0.0)
        path = self.fixture.submit("enzyme", lambda e, i: (self.fixture.reference, [3, 3, 3]))
        self.assertEqual(self.fixture.score("enzyme", path)["pass_rate"], 0.0)

    def test_ligand_clash_fails_the_sm_binder_track(self):
        _, requests = validate_input(self.fixture.input)
        context = np.load(self.fixture.input / requests["sm_binder"]["sm_binder-000"]["context_path"])
        ligand = context["ligand_coords"]

        def builder(example, index):
            # Drive a backbone atom straight onto a ligand atom. Centring the design on
            # the ligand is not enough at a 1.5 A threshold: a helix axis sits about
            # 2.3 A from its own backbone, so the ligand would fit without clashing.
            coordinates, mask, restypes = self.fixture.reference
            moved = coordinates - coordinates[0, 1] + ligand[0]
            return (moved, mask, restypes), None

        result = self.fixture.score("sm_binder", self.fixture.submit("sm_binder", builder))
        self.assertEqual(result["pass_rate"], 0.0)

    def test_binder_far_from_target_fails_the_ppi_track(self):
        path = self.fixture.submit(
            "ppi_binder",
            lambda e, i: (build_structure(helix(self.fixture.LENGTH,
                                                origin=(500.0, 0, 0), seed=i)), None))
        result = self.fixture.score("ppi_binder", path)
        self.assertEqual(result["pass_rate"], 0.0)
        self.assertTrue(any("hotspot" in k for k in result["failures"]),
                        result["failures"])

    def test_verdict_order_from_the_pool_cannot_change_the_score(self):
        """`score_track` must not depend on the order `fold_designs` hands verdicts back.

        Grading folds one design per GPU, so verdicts arrive in completion order, and
        `cluster_designs` is a GREEDY walk: a design within DIVERSITY_RMSD of two others
        that are further apart than that absorbs them if seen first and is absorbed if
        seen second. `bridge_designs` builds exactly that case - 1 cluster in submission
        order, 2 in reverse - so replaying in completion order would move `diversity`.

        Asserted by scrambling the returned mapping rather than by racing threads. Timing
        was tried twice and both attempts were unsound: mutually distinct chains cluster
        the same in any order, and a random per-call delay passed against a known-bad
        grader about a quarter of the time.
        """
        designs = bridge_designs(self.fixture.LENGTH)
        path = self.fixture.submit(
            "unconditional", lambda e, i, d=designs: (d[i % len(d)], None))
        # Pinned passed=True so every design reaches the clustering step.
        expected = self.fixture.score("unconditional", path, StubScorer(passed=True))
        self.assertGreater(expected["designs_designable"], 0)

        original = grader.fold_designs

        def rotated_mapping(scorer, track, context, to_fold):
            # Rotated, not reversed. Reversing is too weak here: the submission cycles
            # A,B,C,A, and its reverse A,C,B,A still leads with the bridging design, so
            # both cluster to 1 and a broken grader would pass. A rotation leads with B.
            verdicts = original(scorer, track, context, to_fold)
            keys = list(verdicts)
            return {key: verdicts[key] for key in keys[1:] + keys[:1]}

        grader.fold_designs = rotated_mapping
        try:
            actual = self.fixture.score("unconditional", path, StubScorer(passed=True))
        finally:
            grader.fold_designs = original
        self.assertAlmostEqual(expected["diversity"], actual["diversity"])
        self.assertAlmostEqual(expected["pass_rate"], actual["pass_rate"])

    def test_parallel_grading_gives_the_same_numbers_as_serial(self):
        """The real thread pool must agree with the one-device path, design for design."""
        designs = bridge_designs(self.fixture.LENGTH)
        path = self.fixture.submit(
            "unconditional", lambda e, i, d=designs: (d[i % len(d)], None))
        serial = self.fixture.score("unconditional", path, StubScorer(passed=True))
        pooled = StubScorer(passed=True, devices=tuple(range(8)))
        parallel = self.fixture.score("unconditional", path, pooled)

        self.assertEqual(pooled.calls, serial["designs_valid"])   # the pool really ran
        self.assertEqual(serial["designs_designable"], parallel["designs_designable"])
        self.assertAlmostEqual(serial["pass_rate"], parallel["pass_rate"])
        self.assertAlmostEqual(serial["diversity"], parallel["diversity"])
        self.assertEqual(serial["failures"], parallel["failures"])

    def test_a_scorer_failure_is_raised_not_scored_as_zero(self):
        """An unusable scorer is infrastructure failure, never a pass rate of zero.

        The parallel path has to carry the exception out of its worker thread; swallowing
        it would turn a broken grading container into a plausible-looking 0.0.
        """
        class Broken(StubScorer):
            def evaluate(self, track, structure, context, motif_map=None):
                raise ScorerUnavailable("no weights")

        path = self.fixture.submit(
            "unconditional",
            lambda e, i: (build_structure(random_compact_chain(self.fixture.LENGTH,
                                                               seed=70 + i)), None))
        for devices in ((None,), tuple(range(4))):
            with self.assertRaises(ScorerUnavailable):
                self.fixture.score("unconditional", path,
                                   Broken(devices=devices))

    # ---- anti-degeneracy -------------------------------------------------
    def test_mode_collapse_is_detected_on_the_unconditional_track(self):
        identical = build_structure(random_compact_chain(self.fixture.LENGTH, seed=7))
        collapsed = self.fixture.score(
            "unconditional",
            self.fixture.submit("unconditional", lambda e, i: (identical, None)))
        varied = self.fixture.score(
            "unconditional", self.fixture.submit(
                "unconditional",
                lambda e, i: (build_structure(random_compact_chain(self.fixture.LENGTH,
                                                                   seed=50 + i)), None)))
        # Mode collapse is invisible in the pass rate - a repeated designable fold
        # passes every time - so diversity is the readout that catches it. The two are
        # reported side by side and never blended, which is why this asserts on both.
        self.assertAlmostEqual(collapsed["pass_rate"], 1.0)
        self.assertAlmostEqual(varied["pass_rate"], 1.0)
        self.assertLess(collapsed["diversity"], varied["diversity"])

    def test_a_trivial_generator_cannot_win_the_conditioned_tracks(self):
        trivial = lambda e, i: (build_structure(helix(self.fixture.LENGTH, seed=0)), None)
        for track in ("sm_binder", "ppi_binder", "enzyme"):
            rate = self.fixture.score(track, self.fixture.submit(track, trivial))["pass_rate"]
            self.assertEqual(rate, 0.0, track)

    # ---- harness manifest -------------------------------------------------
    def _harness(self, kind, learned=True, pretrained=True):
        package = Path(self.directory) / f"pkg-{kind}-{learned}-{pretrained}"
        store = Path(self.directory) / "pretrained"
        package.mkdir(parents=True, exist_ok=True)
        store.mkdir(parents=True, exist_ok=True)
        (package / "infer.sh").write_text("#!/bin/bash\n")
        (package / "README.md").write_text("architecture notes\n")
        components, assets = [], []
        if learned:
            weights = package / "weights.pt"
            weights.write_bytes(b"synthetic-weights")
            components.append({"path": "weights.pt", "sha256": sha256(weights),
                               "parameters": 12_000_000, "trained_in_session": True})
        if pretrained:
            asset = store / "boltz1.ckpt"
            asset.write_bytes(b"synthetic-pretrained")
            assets.append({"asset_id": "boltz1.ckpt", "sha256": sha256(asset),
                           "license": "MIT", "role": "structure predictor inverted at inference"})
        write_json(package / "harness.json", {
            "schema": HARNESS_SCHEMA, "kind": kind, "entry_point": "infer.sh",
            "learned_components": components, "pretrained_assets": assets})
        return package, store

    def test_harness_accepts_each_paradigm(self):
        for kind, learned, pretrained in (("trained_generator", True, False),
                                          ("predictor_inversion", False, True),
                                          ("hybrid", True, True)):
            package, store = self._harness(kind, learned, pretrained)
            manifest = validate_harness(package, store)
            self.assertEqual(manifest["kind"], kind)

    def test_harness_rejects_paradigm_mismatch(self):
        package, store = self._harness("trained_generator", learned=False, pretrained=True)
        with self.assertRaisesRegex(ValueError, "trained in session"):
            validate_harness(package, store)
        package, store = self._harness("predictor_inversion", learned=True, pretrained=False)
        with self.assertRaisesRegex(ValueError, "pretrained asset"):
            validate_harness(package, store)

    def test_harness_rejects_checksum_mismatch(self):
        package, store = self._harness("trained_generator", True, False)
        (package / "weights.pt").write_bytes(b"tampered-weights")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_harness(package, store)

    def test_harness_requires_a_license_for_pretrained_assets(self):
        package, store = self._harness("predictor_inversion", False, True)
        manifest = json.loads((package / "harness.json").read_text())
        manifest["pretrained_assets"][0]["license"] = ""
        write_json(package / "harness.json", manifest)
        with self.assertRaisesRegex(ValueError, "license"):
            validate_harness(package, store)


class VendoredConstantsTest(unittest.TestCase):
    """residues.py exists twice on purpose and must never drift.

    The grading copy lives in tests/; the starter ships its own so a submission stays
    self-contained after relocation, as the contract requires.
    """

    def test_starter_and_grader_share_one_atom14_table(self):
        # Compares two files that only coexist in the repository. The verifier image copies
        # tests/ alone, so there is no baseline copy to compare against inside it; the check
        # is a repository invariant and skips rather than failing when run from the image.
        root = Path(__file__).resolve().parent.parent
        baseline = root / "environment" / "baseline" / "rfd3-small" / "residues.py"
        if not baseline.is_file():
            self.skipTest("repository layout absent (running from the verifier image)")
        grading = (root / "tests" / "residues.py").read_bytes()
        shipped = baseline.read_bytes()
        self.assertEqual(
            sha256_bytes(grading), sha256_bytes(shipped),
            "tests/residues.py and environment/baseline/rfd3-small/residues.py have diverged; "
            "the atom14 table must be identical in the grader and in a submission")


def sha256_bytes(payload):
    import hashlib
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    unittest.main(verbosity=2)
