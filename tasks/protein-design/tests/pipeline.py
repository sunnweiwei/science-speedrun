"""The designability pipeline: LigandMPNN inverse folding plus RoseTTAFold3 refolding.

Runs inside the trusted grader only. It never imports, executes, or links against the
delivered submission; it consumes sealed coordinates.

Criteria follow the benchmark's per-track `success.txt` (vendored at
environment/data/targets/*/success.txt). FOUR sequences per backbone on every track -
the task's own uniform budget, against 8 for enzyme and sm_binder upstream. The
criterion is "pass if ANY sequence passes", so fewer sequences can only lower a pass
rate; these are conservative against the published numbers, never flattering.

    enzyme      4 LigandMPNN sequences, 5 predictor models. Passes if ANY model for ANY
                sequence has, after aligning on the motif backbone (N, CA, C), a motif
                all-atom RMSD <= 1.5 A.
    sm_binder   4 LigandMPNN sequences. backbone RMSD <= 1.5 A AND backbone-aligned
                ligand RMSD <= 5 A AND interface min PAE <= 1.5 AND ipTM >= 0.8.
    ppi_binder  4 ProteinMPNN sequences. min inter-chain PAE <= 1.5, plus binder
                backbone RMSD <= 2 A.
    unconditional
                4 ProteinMPNN sequences. backbone RMSD <= 1.5 A.

LigandMPNN is selected exactly where the context carries a ligand; ProteinMPNN
otherwise. The enzyme ligand-clash term - the one published criterion measured on the raw
design rather than on the refold - is applied by grader.py before folding, purely to skip
a fold that cannot pass. The grader adds no criterion of its own; see
environment/evaluation/RESULTS.md.

ipTM and PAE require a complex-capable predictor, which is why RoseTTAFold3 is used
rather than ESMFold: ESMFold is single-chain and emits neither, and cannot model a
protein-ligand complex at all.

Pinned assets
-------------
LigandMPNN   github.com/dauparas/LigandMPNN @ 26ec57ac976ade5379920dbd43c7f97a91cf82de
             MIT. Weights fetched by its own get_model_params.sh, mirrored in
             `environment/fetch_designability_weights.sh` with pinned sha256.

             The ORIGINAL repository, deliberately - not the re-implementation in
             foundry's `models/mpnn`. Upstream itself says to "use the old repositories
             ... for model benchmarking/comparison until the API and public weights
             stabilize", and that warning is justified: run against the foundry
             re-implementation, `ligand_mpnn` returned byte-identical sequences AND
             confidences with and without a ligand that had 52 of 53 atoms within 5 A
             of the backbone, reporting `ligand_interface_sequence_recovery=nan`. Its
             ligand conditioning is inert, and ligand conditioning is the whole point
             of two of the four tracks.

RoseTTAFold3 github.com/RosettaCommons/foundry @ b02eed6a6bdf8f44d14a80cc36e3da13c9f2291c
             BSD-3, IPD/UW. Source vendored at `environment/evaluation/rf3`;
             parameters are a public download (files.ipd.uw.edu), pinned by sha256 in
             `environment/fetch_designability_weights.sh`.

             RF3 replaces AlphaFold 3, whose parameters are not redistributable and so
             had to be mounted by the operator - which kept designability off the
             agent's own scoring path entirely. RF3's parameters ship, so the criterion
             an agent optimises is the criterion it is graded on.

             NOTE the consequence for comparability: the per-track thresholds below come
             from RFdiffusion3's published `success.txt`, which was calibrated with
             AlphaFold. Scoring with a different predictor does not preserve those
             numbers' published meaning. This is a deliberate, recorded trade.
"""

import contextlib
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

from common import (IPTM_MIN, INTERFACE_PAE_MAX, LIGAND_RMSD_MAX, MODELS_PER_SEQUENCE,
                    MOTIF_RMSD_MAX, PPI_RMSD_MAX, SAMPLING_TEMPERATURE, SC_RMSD_MAX,
                    SEQUENCES_PER_BACKBONE, sha256)
from geometry import kabsch_rotation, superposed_rmsd, write_allatom_pdb
from residues import THREE_TO_ONE

LIGANDMPNN_COMMIT = "26ec57ac976ade5379920dbd43c7f97a91cf82de"
RF3_REVISION = "b02eed6a6bdf8f44d14a80cc36e3da13c9f2291c"
RF3_CHECKPOINT = "rf3_foundry_01_24_latest_remapped.ckpt"
# model_type -> the checkpoint run.py expects for it. Both are fetched by
# environment/fetch_designability_weights.sh.
LIGANDMPNN_WEIGHTS = {"ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
                      "protein_mpnn": "proteinmpnn_v_48_020.pt"}
BINDER_CHAIN = "A"
TARGET_CHAIN = "B"


class ScorerUnavailable(RuntimeError):
    """Raised when the pipeline cannot run. Never degrades into a zero score."""


def sequence_of(restypes):
    return "".join(THREE_TO_ONE.get(name, "X") for name in restypes)


def _het_atom_name(element, index):
    """PDB columns 13-16 for a ligand atom. Always exactly four characters.

    A one-character element sits in column 14 and a two-character element starts in 13,
    so ` C12` is never confused with `CL1`. The index wraps rather than overflowing the
    field: a ligand with more than 999 atoms would otherwise shift every later column,
    and het atom names are never read back - LigandMPNN embeds these atoms by element
    (`getElements()`), not by name.
    """
    body = f"{index % 100:<2d}"          # two digits is all the field has room for
    name = f"{element[:2]}{body}" if len(element) >= 2 else f" {element}{body}"
    assert len(name) == 4, name
    return name


def verify_assets(assets_root, ligandmpnn_root):
    """Fail loudly unless every pinned scoring asset is present.

    Every message names the one command that fixes it, because this runs on whatever
    cluster the operator has and a missing checkpoint must not read as a broken task.
    """
    assets_root, ligandmpnn_root = Path(assets_root), Path(ligandmpnn_root)
    fetch = ("bash environment/fetch_designability_weights.sh "
             f"{assets_root}")

    runner = ligandmpnn_root / "run.py"
    if not runner.is_file():
        raise ScorerUnavailable(
            f"LigandMPNN {LIGANDMPNN_COMMIT} is not installed at {ligandmpnn_root}. "
            "Both images COPY it there from environment/evaluation/ligandmpnn, which vendors that "
            "revision; outside an image, point LIGANDMPNN_ROOT at that directory.")
    weights = assets_root / "ligandmpnn_params"
    missing = [name for name in (LIGANDMPNN_WEIGHTS.values())
               if not (weights / name).is_file()]
    if missing:
        raise ScorerUnavailable(
            f"LigandMPNN parameters missing from {weights}: {', '.join(missing)}. "
            f"Run: {fetch}")

    try:
        import rf3  # noqa: F401
    except ImportError as error:
        raise ScorerUnavailable(
            f"RoseTTAFold3 {RF3_REVISION} is not importable ({error}). Its source is "
            "vendored at environment/evaluation/rf3; put that plus "
            "starter/rfd3/src on PYTHONPATH.")
    checkpoint = assets_root / "rf3" / RF3_CHECKPOINT
    if not checkpoint.is_file():
        raise ScorerUnavailable(
            f"The RoseTTAFold3 checkpoint is not at {checkpoint}. Run: {fetch}")

    return {"ligandmpnn": str(ligandmpnn_root), "ligandmpnn_params": str(weights),
            "rf3_ckpt": str(checkpoint)}


def resolve_devices(devices=None):
    """Which GPUs grading may use - one worker per device, in physical ordinals.

    Every fold and every inverse-folding call is already an isolated subprocess, so the
    only thing needed to use N GPUs is to pin each subprocess to one of them. Ids are
    taken from CUDA_VISIBLE_DEVICES when the environment restricts us, which keeps them
    physical: the CUDA runtime reads that variable afresh in each child, so passing one
    of the parent's ids through selects the same card the parent meant.

    Returns [None] when no GPU is visible, meaning "inherit the environment" - one
    worker, unpinned, which is the single-GPU behaviour this used to have.
    """
    if devices:
        return [str(device) for device in devices]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [part.strip() for part in visible.split(",") if part.strip()] or [None]
    try:
        import torch
        count = torch.cuda.device_count()
    except Exception:                                    # noqa: BLE001 - no torch, no GPU
        count = 0
    return [str(index) for index in range(count)] or [None]


class DesignabilityScorer:
    """Structure -> LigandMPNN sequences -> RoseTTAFold3 complexes -> per-track verdict.

    Thread-safe across `evaluate` calls: each one leases a GPU for its duration and pins
    its subprocesses to it, so `grader.score_track` can fold `len(self.devices)` designs
    at once. Nothing else is shared - the weights live in the child processes, and every
    call works in its own temporary directory.
    """

    def __init__(self, assets_root="/assets", ligandmpnn_root="/opt/ligandmpnn",
                 timeout=3600, devices=None):
        self.paths = verify_assets(assets_root, ligandmpnn_root)
        self.timeout = timeout
        self.devices = resolve_devices(devices)
        self._free = queue.Queue()
        for device in self.devices:
            self._free.put(device)
        self._leased = threading.local()

    @contextlib.contextmanager
    def _lease(self):
        """Hold one GPU for the length of one design's scoring."""
        device = self._free.get()
        self._leased.device = device
        try:
            yield device
        finally:
            self._leased.device = None
            self._free.put(device)

    def _env(self):
        """Child environment, pinned to this thread's leased GPU if it has one."""
        env = dict(os.environ)
        device = getattr(self._leased, "device", None)
        if device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(device)
        return env

    # ---- inverse folding ------------------------------------------------
    def design_sequences(self, track, structure, context, motif_map):
        """LigandMPNN sequences for one backbone, with the ligand in context."""
        coordinates, mask, restypes = structure
        count = SEQUENCES_PER_BACKBONE[track]
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            target = scratch / "design.pdb"
            write_allatom_pdb(target, coordinates, mask, restypes)
            # LigandMPNN reads het atoms from the same file, so append the ligand.
            ligand = context.get("ligand_coords")
            has_ligand = ligand is not None and np.size(ligand) > 0
            if has_ligand:
                names = context.get("ligand_ccd")
                label = str(names[0]) if names is not None and len(names) else "LIG"
                # LigandMPNN embeds each het atom by ELEMENT
                # (`data_utils.parse_PDB` -> `getElements()` -> `element_dict`), so
                # labelling the whole ligand carbon would hand the model the wrong
                # chemistry. The context carries the real elements; use them.
                elements = context.get("ligand_elements")
                lines = target.read_text().splitlines()
                lines = [line for line in lines if line not in ("TER", "END")]
                for index, position in enumerate(np.asarray(ligand), start=1):
                    x, y, z = position
                    element = (str(elements[index - 1]).strip().upper()
                               if elements is not None and len(elements) >= index
                               else "C")
                    lines.append(
                        f"HETATM{9000 + index:>5d} {_het_atom_name(element, index)}"
                        f" {label:<3s} "
                        f"L{1:>4d}    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00"
                        + " " * 10 + f"{element:>2s}")
                target.write_text("\n".join(lines + ["TER", "END"]) + "\n")

            # run.py reads the checkpoint belonging to --model_type, and every
            # --checkpoint_* default is RELATIVE ("./model_params/..."). Passing only
            # --checkpoint_ligand_mpnn while selecting protein_mpnn therefore resolves
            # against the current directory and only works if it happens to be the
            # LigandMPNN root. Pass the flag that matches the model.
            model_type = "ligand_mpnn" if has_ligand else "protein_mpnn"
            weights = {"ligand_mpnn": ("--checkpoint_ligand_mpnn",
                                       "ligandmpnn_v_32_010_25.pt"),
                       "protein_mpnn": ("--checkpoint_protein_mpnn",
                                        "proteinmpnn_v_48_020.pt")}[model_type]
            command = [
                sys.executable, str(Path(self.paths["ligandmpnn"]) / "run.py"),
                "--model_type", model_type,
                weights[0],
                str(Path(self.paths["ligandmpnn_params"]) / weights[1]),
                "--pdb_path", str(target),
                "--out_folder", str(scratch),
                "--batch_size", "1",
                "--number_of_batches", str(count),
                "--temperature", str(SAMPLING_TEMPERATURE),
                "--seed", "37",
            ]
            if motif_map:
                # Motif residues keep their declared identities; only the scaffold is
                # redesigned, which is what the enzyme criterion measures.
                fixed = " ".join(f"{BINDER_CHAIN}{index + 1}" for index in sorted(motif_map))
                command += ["--fixed_residues", fixed]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=self.timeout, env=self._env())
            if result.returncode:
                raise ScorerUnavailable(f"LigandMPNN failed: {result.stderr[-500:]}")
            fasta = list((scratch / "seqs").glob("*.fa"))
            if not fasta:
                raise ScorerUnavailable("LigandMPNN produced no sequences")
            sequences = []
            for line in fasta[0].read_text().splitlines():
                if line and not line.startswith(">"):
                    sequences.append(line.strip().split("/")[0])
            # The first record is the input sequence echoed back.
            return (sequences[1:] or sequences)[:count]

    # ---- complex prediction ---------------------------------------------
    def _components(self, sequence, context, scratch):
        """RF3 `components` for one designed complex, plus the chains to template."""
        components = [{"seq": sequence, "chain_id": BINDER_CHAIN}]
        templates = []

        ccd = context.get("ligand_ccd")
        if ccd is not None and len(ccd):
            components += [{"ccd_code": str(code)} for code in ccd]

        target = context.get("target_coords")
        if target is not None and np.size(target):
            # TEMPLATE the target rather than handing RF3 its bare sequence. A natural
            # target chain does not fold single-sequence - measured predicted-vs-native
            # CA RMSD of 5.3-17.7 A across the five benchmark targets - and an interface
            # scored against an unfolded target is meaningless. RF3 is trained on
            # explicit atomic templates and its docs give templating as the way to fold
            # a chain without an MSA. Measured with the template: target CA RMSD 0.097 A.
            target_pdb = scratch / "target.pdb"
            if not target_pdb.is_file():
                write_allatom_pdb(
                    target_pdb, np.asarray(target),
                    np.asarray(context["target_mask"], dtype=bool),
                    [str(res) for res in context["target_restypes"]],
                    chain=TARGET_CHAIN)
            components.append({"path": str(target_pdb)})
            templates.append(TARGET_CHAIN)
        return components, templates

    def predict_all(self, name, sequences, context, seeds):
        """Fold every designed sequence in ONE RF3 invocation.

        Batching matters: RF3 loads a ~2.8 GB checkpoint per process, so folding eight
        sequences as eight subprocesses spends most of its wall clock loading the same
        weights. `inputs` takes a LIST of jobs, so one call covers a whole design.

        Returns {job name -> [model record, ...]}, in the order given.
        """
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            jobs, names = [], []
            for index, sequence in enumerate(sequences):
                components, templates = self._components(sequence, context, scratch)
                job_name = f"{name}-{index}"
                job = {"name": job_name, "components": components}
                if templates:
                    job["template_selection"] = templates
                jobs.append(job)
                names.append(job_name)
            (scratch / "input.json").write_text(json.dumps(jobs))

            # One seed; RF3 draws `diffusion_batch_size` diffusion samples from it, so
            # that is what carries MODELS_PER_SEQUENCE. `seeds` is the caller's
            # statement of how many models the criterion asks for.
            command = [
                sys.executable, "-c", "from rf3.cli import app; app()", "fold",
                f"inputs={scratch / 'input.json'}",
                f"out_dir={scratch / 'out'}",
                f"ckpt_path={self.paths['rf3_ckpt']}",
                f"diffusion_batch_size={max(1, len(list(seeds)))}",
                "seed=37",
            ]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=self.timeout, env=self._env())
            if result.returncode:
                raise ScorerUnavailable(f"RoseTTAFold3 failed: {result.stderr[-500:]}")

            out = {}
            for job_name in names:
                # RF3 writes `<name>/seed-N_sample-M/<name>_seed-N_sample-M_model.cif`
                # plus one top-ranked `<name>_model.cif`. Read the SAMPLES: the enzyme
                # criterion is "any of 5 predictor models", and matching only the ranked
                # file silently reduces that to one. `*model.cif` also matches
                # AlphaFold's bare `model.cif`, so either layout works.
                directory = scratch / "out" / job_name
                samples = sorted(directory.rglob("seed-*_sample-*/*model.cif"))
                if not samples:
                    samples = sorted(directory.rglob("*_model.cif"))
                out[job_name] = [self._read_model(path) for path in samples]
            return [out[job_name] for job_name in names]

    def predict(self, name, sequence, context, seeds):
        """Fold one designed complex. Returns one record per model."""
        return self.predict_all(name, [sequence], context, seeds)[0]

    def _read_model(self, model_path):
        """Extract designed-chain coordinates, ligand atoms, ipTM and interface PAE."""
        try:
            import gemmi
        except ImportError as error:
            raise ScorerUnavailable(f"gemmi is required to read predictor output: {error}")
        structure = gemmi.read_structure(str(model_path))
        structure.remove_alternative_conformations()
        structure.remove_hydrogens()
        tabulated = gemmi.find_tabulated_residue

        backbone, ligand = [], []
        for chain in structure[0]:
            for residue in chain:
                info = tabulated(residue.name)
                if info is not None and info.is_amino_acid():
                    if chain.name != BINDER_CHAIN:
                        continue
                    atoms = {a.name: [a.pos.x, a.pos.y, a.pos.z] for a in residue}
                    if {"N", "CA", "C"} <= set(atoms):
                        backbone.append({"name": residue.name, "atoms": atoms})
                elif not residue.is_water():
                    ligand.extend([a.pos.x, a.pos.y, a.pos.z] for a in residue)

        # Two layouts: `<name>_model.cif` beside `<name>_summary_confidences.json`,
        # and `seed-N_sample-M/model.cif` beside a bare `summary_confidences.json`.
        stem = model_path.name.replace("_model.cif", "")
        summary = model_path.parent / f"{stem}_summary_confidences.json"
        if not summary.is_file():
            summary = model_path.parent / "summary_confidences.json"
        iptm = interface_pae = None
        if summary.is_file():
            payload = json.loads(summary.read_text())
            iptm = payload.get("iptm")
            pairs = payload.get("chain_pair_pae_min")
            if isinstance(pairs, list) and pairs:
                values = [pairs[i][j] for i in range(len(pairs))
                          for j in range(len(pairs[i])) if i != j
                          and isinstance(pairs[i][j], (int, float))]
                interface_pae = min(values) if values else None
        return {"backbone": backbone, "ligand": np.array(ligand) if ligand else None,
                "iptm": iptm, "interface_pae": interface_pae}

    # ---- per-track criteria ---------------------------------------------
    @staticmethod
    def _motif_arrays(residues, indices):
        """Atom14-shaped coordinates and mask for selected refolded residues."""
        from residues import ATOM14, ATOM14_WIDTH, atom_slot
        coordinates = np.zeros((len(indices), ATOM14_WIDTH, 3))
        mask = np.zeros((len(indices), ATOM14_WIDTH), dtype=bool)
        names = []
        for row, index in enumerate(indices):
            if index >= len(residues):
                return None, None, None
            residue = residues[index]
            names.append(residue["name"])
            if residue["name"] not in ATOM14:
                continue
            for atom, position in residue["atoms"].items():
                slot = atom_slot(residue["name"], atom)
                if slot is not None:
                    coordinates[row, slot] = position
                    mask[row, slot] = True
        return coordinates, mask, names

    def _motif_recapitulation(self, model, motif_map, context):
        """Align refolded motif on N/CA/C, then score all shared motif atoms."""
        reference = context["motif_coords"]
        reference_mask = np.asarray(context["motif_mask"], dtype=bool)
        coordinates, mask, _ = self._motif_arrays(model["backbone"], motif_map)
        if coordinates is None:
            return None
        shared = mask & reference_mask
        frame = shared.copy()
        frame[:, 3:] = False                      # align on N, CA, C only
        if frame.sum() < 3:
            # These contexts mostly carry fixed catalytic TIP atoms, not backbone: 25 of
            # the 41 enzyme evaluation contexts have fewer than 3 backbone atoms in
            # `motif_mask` (median 1, min 0). Returning None here made the criterion
            # unsatisfiable by construction for 61% of the track. `geometry.py::
            # backbone_aligned_motif_rmsd` has always had this fallback; keep the two
            # in step.
            frame = shared
        if shared.sum() < 3:
            return None
        mobile, target = coordinates[frame], reference[frame]
        offset_mobile, offset_target = mobile.mean(axis=0), target.mean(axis=0)
        rotation = kabsch_rotation(mobile - offset_mobile, target - offset_target)
        moved = (coordinates[shared] - offset_mobile) @ rotation.T + offset_target
        return float(np.sqrt(np.mean(np.sum((moved - reference[shared]) ** 2, axis=1))))

    @staticmethod
    def _backbone_rmsd_and_transform(model, design_ca):
        """CA RMSD of a refolded chain against the design, plus the fitted transform."""
        refolded = np.array([r["atoms"]["CA"] for r in model["backbone"]])
        if refolded.shape[0] != design_ca.shape[0] or refolded.shape[0] < 3:
            return None, None
        offset_mobile, offset_target = refolded.mean(axis=0), design_ca.mean(axis=0)
        rotation = kabsch_rotation(refolded - offset_mobile, design_ca - offset_target)
        moved = (refolded - offset_mobile) @ rotation.T + offset_target
        rmsd = float(np.sqrt(np.mean(np.sum((moved - design_ca) ** 2, axis=1))))
        return rmsd, (offset_mobile, rotation, offset_target)

    def evaluate(self, track, structure, context, motif_map=None):
        """Run the track's published criterion. Returns a verdict dict."""
        with self._lease():
            return self._evaluate(track, structure, context, motif_map)

    def _evaluate(self, track, structure, context, motif_map):
        coordinates, mask, restypes = structure
        design_ca = coordinates[:, 1]
        sequences = self.design_sequences(track, structure, context, motif_map)
        seeds = list(range(MODELS_PER_SEQUENCE[track]))
        best = {"passed": False, "sc_rmsd": float("inf"), "iptm": None,
                "interface_pae": None, "ligand_rmsd": None, "motif_rmsd": None,
                "sequences_tried": 0, "reason": "no sequence satisfied the criterion"}

        # One RF3 call for the whole design, not one per sequence - see predict_all.
        # This does mean every sequence is folded rather than stopping at the first
        # success; the criterion is "any sequence passes", so the verdict is identical.
        folded = self.predict_all("design", sequences, context, seeds)

        for number, models in enumerate(folded):
            best["sequences_tried"] = number + 1
            for model in models:
                rmsd, transform = self._backbone_rmsd_and_transform(model, design_ca)
                if rmsd is None:
                    continue
                record = {"sc_rmsd": rmsd, "iptm": model["iptm"],
                          "interface_pae": model["interface_pae"],
                          "ligand_rmsd": None, "motif_rmsd": None}

                if track == "enzyme":
                    # Passes on motif recapitulation alone, by any model of any
                    # sequence. The ligand-clash half of the criterion is a geometry
                    # check on the design and is applied by the grader.
                    record["motif_rmsd"] = self._motif_recapitulation(
                        model, motif_map or [], context)
                    ok = (record["motif_rmsd"] is not None
                          and record["motif_rmsd"] <= MOTIF_RMSD_MAX)
                elif track == "sm_binder":
                    ligand = context.get("ligand_coords")
                    if model["ligand"] is not None and ligand is not None and \
                            model["ligand"].shape == np.asarray(ligand).shape:
                        origin, rotation, destination = transform
                        moved = (model["ligand"] - origin) @ rotation.T + destination
                        record["ligand_rmsd"] = float(np.sqrt(np.mean(
                            np.sum((moved - np.asarray(ligand)) ** 2, axis=1))))
                    ok = (rmsd <= SC_RMSD_MAX
                          and record["ligand_rmsd"] is not None
                          and record["ligand_rmsd"] <= LIGAND_RMSD_MAX
                          and record["interface_pae"] is not None
                          and record["interface_pae"] <= INTERFACE_PAE_MAX
                          and record["iptm"] is not None
                          and record["iptm"] >= IPTM_MIN)
                elif track == "ppi_binder":
                    # success.txt gives no PAE threshold for this track; the
                    # sm_binder value is reused and flagged as inferred.
                    ok = (rmsd <= PPI_RMSD_MAX
                          and record["interface_pae"] is not None
                          and record["interface_pae"] <= INTERFACE_PAE_MAX)
                else:
                    ok = rmsd <= SC_RMSD_MAX

                if rmsd < best["sc_rmsd"]:
                    best.update(record)
                if ok:
                    best.update(record)
                    best["passed"] = True
                    best["reason"] = None
                    return best
        return best
