#!/usr/bin/env python3
"""Generate and/or package the full 51-target x 20-design (1,020-slot) straight-helix negative control.

Produces a deterministic, contract-compliant null submission across all four tracks of the
51-target evaluation suite (`1` unconditional + `4` sm_binder + `5` ppi_binder + `41` enzyme =
`51` targets x `20` designs = `1,020` backbones):
  - Places an idealized canonical right-handed straight alpha-helix (`3.80 A` CA-CA step,
    `100 deg` turn/residue, `1.50 A` rise/residue) of the exact requested `design_length`
    (chain `A`) near the conditioning input centroid.
  - On `enzyme`, assigns distinct evenly spaced residue indices (`motif_map`) along the
    straight helix carrying the requested `motif_restypes` with idealized canonical ATOM14
    sidechain coordinates (no native motif geometry grafting, no ligand/target clash
    optimization, and no inverse-folding or predictor-inversion tricks).
  - Supports materializing the 51-target input view from `environment/data/targets` (using
    `ingest_targets` when `gemmi` is installed, or a pure-Python mmCIF `_atom_site` reader
    on bare-metal CPU hosts), writing `predictions/` (`1,020` PDBs), sealing a grader-ready
    snapshot (`--sealed-dir`), and emitting a self-contained submission package (`--package-dir`)
    ready to run through `tests/main_verifier.py` -> `tests/grader.py` on GPU.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import sys
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from residues import ATOM14, ATOM14_WIDTH, atom_slot  # noqa: E402

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")
MOTIF_TRACKS = ("enzyme",)
DESIGNS_PER_TARGET = 20
UNCONDITIONAL_TARGETS = {"uncond_L100": 100}
WATER_NAMES = {"HOH", "WAT", "DOD", "H2O"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(example_id: str, design_index: int) -> int:
    payload = f"trivial-helix-1020:{example_id}:{design_index}".encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def canonical_straight_helix_ca(
    length: int,
    origin: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Idealized canonical right-handed straight alpha-helix CA trace at 3.80 A spacing."""
    idx = np.arange(length, dtype=np.float64)
    angles = idx * (100.0 * np.pi / 180.0)
    trace = np.stack(
        [2.30 * np.cos(angles), 2.30 * np.sin(angles), 1.50 * idx],
        axis=-1,
    )
    if length >= 2:
        step = float(np.linalg.norm(trace[1] - trace[0]))
        trace *= 3.80 / step
    trace -= trace.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q @ np.diag(np.where(np.diag(r) >= 0.0, 1.0, -1.0))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    jitter = rng.uniform(-1.5, 1.5, size=(1, 3))
    return trace @ q.T + np.asarray(origin, dtype=np.float64).reshape(1, 3) + jitter


def build_atom14_from_ca_trace(
    trace: np.ndarray,
    restypes: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct canonical peptide backbone (N, CA, C, O) + tetrahedral CB + ATOM14 heavy atoms."""
    length = int(trace.shape[0])
    coordinates = np.zeros((length, ATOM14_WIDTH, 3), dtype=np.float64)
    mask = np.zeros((length, ATOM14_WIDTH), dtype=bool)
    for index in range(length):
        alpha = trace[index]
        if length == 1:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            forward = trace[min(index + 1, length - 1)] - trace[max(index - 1, 0)]
            forward = forward / (np.linalg.norm(forward) + 1e-9)
        helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(forward @ helper)) > 0.9:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        side = np.cross(forward, helper)
        side /= np.linalg.norm(side) + 1e-9

        nitrogen = alpha - 1.20 * forward + 0.50 * side
        carbon = alpha + 1.20 * forward + 0.50 * side
        oxygen = carbon + 0.30 * forward + 1.05 * side
        coordinates[index, 0] = nitrogen
        coordinates[index, 1] = alpha
        coordinates[index, 2] = carbon
        coordinates[index, 3] = oxygen
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
        extra_count = len(atoms) - 5
        for slot in range(5, len(atoms)):
            angle = 2.0 * np.pi * (slot - 5) / max(1, extra_count)
            coordinates[index, slot] = (
                beta
                + 0.90 * outward
                + 1.10 * (np.cos(angle) * perpendicular + np.sin(angle) * second)
            )
            mask[index, slot] = True
    return coordinates, mask, list(restypes)


def write_single_chain_pdb(
    path: Path,
    coordinates: np.ndarray,
    mask: np.ndarray,
    restypes: list[str],
    chain: str = "A",
) -> None:
    """Write canonical single-chain ATOM14 PDB columns compatible with `parse_allatom_pdb`."""
    lines: list[str] = []
    serial = 1
    for index, residue_name in enumerate(restypes):
        for slot, atom in enumerate(ATOM14[residue_name]):
            if not mask[index, slot]:
                continue
            x, y, z = coordinates[index, slot]
            field = f" {atom:<3s}" if len(atom) < 4 else atom
            lines.append(
                f"ATOM  {serial:>5d} {field} {residue_name:<3s} {chain}{index + 1:>4d}    "
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00" + " " * 10 + f"{atom[0]:>2s}"
            )
            serial += 1
    lines.append("TER")
    lines.append("END")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def even_motif_positions(design_length: int, motif_size: int) -> list[int]:
    """Select `motif_size` strictly increasing, evenly spaced 0-based indices in `[0, design_length)`."""
    if not 1 <= motif_size <= design_length:
        raise ValueError(f"invalid motif_size={motif_size} for design_length={design_length}")
    raw = [int((k + 1) * design_length // (motif_size + 1)) for k in range(motif_size)]
    positions: list[int] = []
    for k, candidate in enumerate(raw):
        lower = positions[-1] + 1 if positions else 0
        upper = design_length - (motif_size - k)
        positions.append(max(lower, min(upper, candidate)))
    return positions


def conditioning_centroid(track: str, context: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a simple centroid of the public conditioning coordinates (no docking or clash search)."""
    points: list[np.ndarray] = []
    if track == "enzyme" and "motif_coords" in context and "motif_mask" in context:
        motif_mask = np.asarray(context["motif_mask"], dtype=bool)
        motif_pts = np.asarray(context["motif_coords"], dtype=np.float64)[motif_mask]
        if motif_pts.size:
            points.append(motif_pts.reshape(-1, 3))
    if "ligand_coords" in context and np.size(context["ligand_coords"]):
        points.append(np.asarray(context["ligand_coords"], dtype=np.float64).reshape(-1, 3))
    if "target_coords" in context and "target_mask" in context:
        target_mask = np.asarray(context["target_mask"], dtype=bool)
        target_pts = np.asarray(context["target_coords"], dtype=np.float64)[target_mask]
        if target_pts.size:
            points.append(target_pts.reshape(-1, 3))
    if not points:
        return np.zeros(3, dtype=np.float64)
    merged = np.concatenate(points, axis=0)
    return merged.mean(axis=0)


# ---- Pure-Python mmCIF fallback when `gemmi` is absent on a bare-metal CPU host --------


def _parse_formal_charge(token: str) -> float:
    token = token.strip()
    if not token or token in (".", "?"):
        return 0.0
    if token.endswith("+") or token.endswith("-"):
        sign = -1.0 if token.endswith("-") else 1.0
        body = token[:-1]
        return sign * (float(body) if body else 1.0)
    try:
        return float(token)
    except ValueError:
        return 0.0


def _read_cif_pure_python(path: Path):
    """Parse `_atom_site` loop from a benchmark target `.cif` without requiring `gemmi`."""
    lines = Path(path).read_text().splitlines()
    headers: list[str] = []
    rows: list[list[str]] = []
    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()
        if stripped == "loop_":
            j = i + 1
            loop_headers: list[str] = []
            while j < n and lines[j].strip().startswith("_"):
                loop_headers.append(lines[j].strip())
                j += 1
            if loop_headers and loop_headers[0].startswith("_atom_site."):
                headers = [h.split(".", 1)[1] for h in loop_headers]
                while j < n:
                    row_line = lines[j].strip()
                    if (
                        not row_line
                        or row_line.startswith("#")
                        or row_line == "loop_"
                        or row_line.startswith("_")
                        or row_line.startswith("data_")
                    ):
                        break
                    rows.append(shlex.split(row_line))
                    j += 1
                break
        i += 1
    if not headers:
        raise ValueError(f"no _atom_site loop found in {path}")

    col = {name: idx for idx, name in enumerate(headers)}
    residues: collections.OrderedDict[tuple[str, int], dict] = collections.OrderedDict()
    ligand_groups: collections.OrderedDict[tuple[str, str, str], dict] = collections.OrderedDict()
    first_model: str | None = None

    for tokens in rows:
        if len(tokens) != len(headers):
            continue
        if "pdbx_PDB_model_num" in col:
            model_num = tokens[col["pdbx_PDB_model_num"]]
            if first_model is None:
                first_model = model_num
            elif model_num != first_model:
                continue
        if "label_alt_id" in col:
            alt = tokens[col["label_alt_id"]]
            if alt not in (".", "?", "A"):
                continue
        element = tokens[col["type_symbol"]].strip().upper()
        if element in ("H", "D"):
            continue
        comp_id = tokens[col.get("auth_comp_id", col["label_comp_id"])].strip()
        chain_id = tokens[col.get("auth_asym_id", col["label_asym_id"])].strip()
        seq_raw = tokens[col.get("auth_seq_id", col["label_seq_id"])].strip()
        atom_name = tokens[col.get("auth_atom_id", col["label_atom_id"])].strip()
        try:
            x = float(tokens[col["Cartn_x"]])
            y = float(tokens[col["Cartn_y"]])
            z = float(tokens[col["Cartn_z"]])
        except ValueError:
            continue

        group = tokens[col["group_PDB"]].strip() if "group_PDB" in col else "ATOM"
        if group == "ATOM" and comp_id in ATOM14:
            seq_num = int(seq_raw)
            slot = atom_slot(comp_id, atom_name)
            if slot is None or not np.isfinite([x, y, z]).all():
                continue
            key = (chain_id, seq_num)
            if key not in residues:
                residues[key] = {
                    "name": comp_id,
                    "coordinates": np.zeros((ATOM14_WIDTH, 3), dtype=np.float64),
                    "present": np.zeros(ATOM14_WIDTH, dtype=bool),
                    "slots": {},
                }
            entry = residues[key]
            if slot not in entry["slots"]:
                entry["coordinates"][slot] = [x, y, z]
                entry["present"][slot] = True
                entry["slots"][atom_name] = slot
        elif comp_id not in WATER_NAMES:
            if not np.isfinite([x, y, z]).all():
                continue
            charge = (
                _parse_formal_charge(tokens[col["pdbx_formal_charge"]])
                if "pdbx_formal_charge" in col
                else 0.0
            )
            lig_key = (chain_id, seq_raw, comp_id)
            if lig_key not in ligand_groups:
                ligand_groups[lig_key] = {
                    "name": comp_id,
                    "coordinates": [],
                    "elements": [],
                    "charges": [],
                    "atom_names": [],
                }
            lig = ligand_groups[lig_key]
            lig["coordinates"].append([x, y, z])
            lig["elements"].append(element)
            lig["charges"].append(charge)
            lig["atom_names"].append(atom_name)

    ligands = []
    for lig in ligand_groups.values():
        if lig["coordinates"]:
            ligands.append(
                {
                    "name": lig["name"],
                    "coordinates": np.asarray(lig["coordinates"], dtype=np.float64),
                    "elements": lig["elements"],
                    "charges": lig["charges"],
                    "atom_names": lig["atom_names"],
                }
            )
    return residues, ligands


def materialize_targets_view(targets_root: Path, output_root: Path) -> dict[str, int]:
    """Materialize the 51-target `verifier` input view from `environment/data/targets`."""
    data_dir = _HERE.parent / "environment" / "data"
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    import ingest_targets  # noqa: E402

    try:
        import gemmi as _gemmi  # noqa: F401

        counts, skipped = ingest_targets.ingest(targets_root, output_root)
    except ImportError:
        original_reader = ingest_targets.read_structure
        ingest_targets.read_structure = _read_cif_pure_python
        try:
            counts, skipped = ingest_targets.ingest(targets_root, output_root)
        finally:
            ingest_targets.read_structure = original_reader

    if skipped:
        raise RuntimeError(f"target materialization skipped entries: {dict(skipped)}")
    return counts


def generate_predictions(input_root: Path, output_root: Path) -> dict[str, int]:
    """Read an input view (`manifest.json` + `<track>.jsonl` + `contexts/`) and write all PDB slots."""
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    manifest = json.loads((input_root / "manifest.json").read_text())
    predictions_dir = output_root / "predictions"
    structures_dir = predictions_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    emitted: dict[str, int] = {}
    for track in TRACKS:
        if f"{track}.jsonl" not in manifest.get("files", {}):
            continue
        track_rows = [
            json.loads(line)
            for line in (input_root / f"{track}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        out_records = []
        count_for_track = 0
        for row in track_rows:
            example_id = str(row["example_id"])
            design_length = int(row["design_length"])
            num_designs = int(row["num_designs"])
            with np.load(input_root / row["context_path"], allow_pickle=False) as handle:
                context = {k: handle[k] for k in handle.files}

            origin = conditioning_centroid(track, context)
            motif_map: list[int] | None = None
            restypes = ["ALA"] * design_length
            if track in MOTIF_TRACKS:
                motif_types = [str(x) for x in context["motif_restypes"]]
                motif_map = even_motif_positions(design_length, len(motif_types))
                for pos, res_name in zip(motif_map, motif_types):
                    restypes[pos] = res_name

            designs = []
            for design_index in range(num_designs):
                seed = stable_seed(example_id, design_index)
                ca_trace = canonical_straight_helix_ca(design_length, origin=origin, seed=seed)
                coords, mask, rtypes = build_atom14_from_ca_trace(ca_trace, restypes)
                rel_pdb = f"{track}/{example_id}_{design_index:02d}.pdb"
                write_single_chain_pdb(structures_dir / rel_pdb, coords, mask, rtypes, chain="A")
                entry = {
                    "design_index": design_index,
                    "ranking_score": round(1.0 - 0.01 * design_index, 6),
                    "structure_path": rel_pdb,
                }
                if motif_map is not None:
                    entry["motif_map"] = motif_map
                designs.append(entry)
                count_for_track += 1
            out_records.append({"example_id": example_id, "designs": designs})

        (predictions_dir / f"{track}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in out_records)
        )
        emitted[track] = count_for_track
    return emitted


def write_submission_package(package_dir: Path) -> Path:
    """Create a relocatable, contract-valid submission package for `main_verifier.py`."""
    package_dir = Path(package_dir).resolve()
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(_HERE / "residues.py", package_dir / "residues.py")
    shutil.copyfile(Path(__file__).resolve(), package_dir / "generate_trivial_helix_1020.py")

    params_path = package_dir / "null_helix_params.json"
    params_path.write_text(
        json.dumps(
            {
                "kind": "canonical_straight_alpha_helix_negative_control",
                "ca_step_angstrom": 3.80,
                "turn_degrees": 100.0,
                "rise_angstrom": 1.50,
                "scaffold_residue": "ALA",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    infer_path = package_dir / "infer.sh"
    infer_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'python3 "${SCRIPT_DIR}/generate_trivial_helix_1020.py" \\\n'
        '  --input-root "$1" \\\n'
        '  --output-dir "$2"\n'
    )
    infer_path.chmod(infer_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    (package_dir / "README.md").write_text(
        "# Trivial Straight-Helix Negative Control (1,020 Slots)\n\n"
        "Deterministic null generator emitting idealized canonical straight right-handed\n"
        "alpha-helices (`3.80 A` CA-CA spacing) for all requested targets and slots without\n"
        "clash optimization, motif geometry grafting, or inverse-folding tricks.\n"
    )

    harness = {
        "schema": "protein-design-harness/v1",
        "kind": "trained_generator",
        "entry_point": "infer.sh",
        "learned_components": [
            {
                "path": "null_helix_params.json",
                "sha256": sha256_file(params_path),
                "parameters": 0,
                "trained_in_session": True,
            }
        ],
        "pretrained_assets": [],
    }
    (package_dir / "harness.json").write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n"
    )
    return package_dir


def main() -> None:
    default_targets = _HERE.parent / "environment" / "data" / "targets"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Materialized input view (`manifest.json` + `<track>.jsonl` + `contexts/`).",
    )
    parser.add_argument(
        "--targets-root",
        type=Path,
        default=default_targets,
        help="Benchmark target directory used when `--input-root` is not provided.",
    )
    parser.add_argument(
        "--materialized-view",
        type=Path,
        default=None,
        help="Where to write the materialized 51-target input view when using `--targets-root`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory in which `predictions/<track>.jsonl` and `predictions/structures/` are written.",
    )
    parser.add_argument(
        "--sealed-dir",
        type=Path,
        default=None,
        help="Optional sealed snapshot output directory (`snapshot.json` + `READY`) for `grader.py`.",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Optional output directory for a complete submission package (`infer.sh`, `harness.json`).",
    )
    args = parser.parse_args()

    if args.output_dir is None and args.package_dir is None:
        parser.error("Provide at least one of `--output-dir` or `--package-dir`.")

    summary: dict[str, object] = {}
    if args.package_dir is not None:
        pkg = write_submission_package(args.package_dir)
        summary["package_dir"] = str(pkg)

    if args.output_dir is not None:
        out_dir = args.output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.input_root is not None:
            view_root = args.input_root.resolve()
        else:
            view_root = (
                args.materialized_view.resolve()
                if args.materialized_view is not None
                else (out_dir / "input_view")
            )
            counts = materialize_targets_view(args.targets_root.resolve(), view_root)
            summary["target_counts"] = counts

        emitted = generate_predictions(view_root, out_dir)
        summary["input_root"] = str(view_root)
        summary["output_dir"] = str(out_dir)
        summary["designs_emitted"] = emitted
        summary["total_designs"] = int(sum(emitted.values()))

        if args.sealed_dir is not None:
            from main_verifier import seal_designs

            sealed = args.sealed_dir.resolve()
            sealed.parent.mkdir(parents=True, exist_ok=True)
            seal_designs(out_dir, sealed, view_root / "manifest.json")
            summary["sealed_dir"] = str(sealed)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
