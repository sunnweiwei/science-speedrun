"""Shared constants, safe I/O, and input validation for the protein-design task."""

import hashlib
import json
from pathlib import Path

TRACKS = ("unconditional", "sm_binder", "ppi_binder", "enzyme")
CONDITIONED_TRACKS = ("sm_binder", "ppi_binder", "enzyme")
# Only the enzyme track scaffolds a motif. An sm_binder target is a bare ligand:
# RFdiffusion3 designs a protein around it with no pocket template.
MOTIF_TRACKS = ("enzyme",)

INPUT_SCHEMA = "protein-design-input/v1"
HARNESS_SCHEMA = "protein-design-harness/v1"
SNAPSHOT_SCHEMA = "protein-design-design-snapshot/v1"
REPORT_SCHEMA = "protein-design-report/v1"

# Structures use the atom14 layout: [L, 14, 3] with an [L, 14] occupancy mask and
# backbone N/CA/C/O pinned to slots 0-3. See residues.py.
ATOM14_WIDTH = 14
BACKBONE_ATOMS = ("N", "CA", "C", "O")

# What a delivered design system may be. BoltzDesign-style inversion trains nothing,
# so "learned parameters trained in session" cannot be a universal requirement.
HARNESS_KINDS = ("trained_generator", "predictor_inversion", "hybrid")

# Acceptance thresholds. These define the metric and must not drift silently.
# Per-track evaluation budget. See environment/data/targets/*/success.txt.
# FOUR sequences per backbone on every track, which is the task's own budget: the
# published criteria use 8 on enzyme and sm_binder and 4 on ppi_binder, and 4
# everywhere trades a little statistical power for a uniform, affordable protocol.
# The criterion is "pass if ANY sequence passes", so fewer sequences can only LOWER
# a pass rate - these numbers are conservative against the published ones, never
# flattering, and they are not comparable with RFdiffusion3's own tables.
# enzyme additionally folds 5 predictor models per sequence; the other tracks fold 1.
SEQUENCES_PER_BACKBONE = {"unconditional": 4, "sm_binder": 4,
                          "ppi_binder": 4, "enzyme": 4}
MODELS_PER_SEQUENCE = {"unconditional": 1, "sm_binder": 1,
                       "ppi_binder": 1, "enzyme": 5}
SAMPLING_TEMPERATURE = 0.1

# RFdiffusion3's published success criteria (results/*/success.txt).
SC_RMSD_MAX = 1.5          # angstrom, backbone RMSD of the refolded design
PLDDT_MIN = 70.0           # refolded-model confidence, when no complex predictor is used
MOTIF_RMSD_MAX = 1.5       # angstrom, motif all-atom RMSD after motif-backbone alignment
NOVELTY_MAX_TM = 0.8       # a passing design's max TM-score to any fit-corpus chain must stay below this (novelty.py)
LIGAND_RMSD_MAX = 5.0      # angstrom, backbone-aligned ligand RMSD (sm_binder)
INTERFACE_PAE_MAX = 1.5    # predicted aligned error at the interface
IPTM_MIN = 0.8             # interface predicted TM-score
PPI_RMSD_MAX = 2.0         # angstrom, binder backbone RMSD (ppi_binder)
HOTSPOT_CONTACT_MAX = 5.0  # angstrom, binder-to-hotspot heavy-atom contact
LIGAND_CLASH_MIN = 1.5     # angstrom, non-motif backbone atom to nearest ligand atom
LIGAND_CONTACT_MAX = 5.0   # angstrom, design atom to ligand atom counted as contact
DIVERSITY_RMSD = 2.0       # angstrom, within-target structural clustering radius
CLASH_MIN = 2.0            # angstrom, inter-atom distance below which a fold is impossible

# Of the thresholds above, only LIGAND_CLASH_MIN and DIVERSITY_RMSD are applied to a
# graded submission: the first is the published enzyme ligand-clash term, the second
# reports diversity beside the pass rate. The rest are the predictor's own criteria, or -
# CLASH_MIN, LIGAND_CONTACT_MAX, HOTSPOT_CONTACT_MAX - the test suite's stand-in for what
# a predictor would refuse to refold.
#
# CA_CA_DISTANCE, CA_CA_TOLERANCE, LIGAND_BURIAL_MIN and HOTSPOT_FRACTION_MIN were the
# task's own pre-fold gate, deleted 2026-09-26 along with it: designability is the only
# metric, and a threshold table should not advertise limits nothing enforces. See
# environment/evaluation/RESULTS.md for what that gate had been costing.
MIN_DESIGN_LENGTH = 20
MAX_DESIGN_LENGTH = 600
# RFdiffusion3's benchmark protocol is 100 backbones per target, and this task reports
# its designability numbers on exactly that. `validate_input` rejects anything larger,
# so raising the protocol means raising this too - they were 32 and 4 respectively, and
# the mismatch made 100 unrepresentable on every path that validates an input view.
MAX_DESIGNS_PER_TARGET = 100

INPUT_FIELDS = {
    "example_id", "track", "design_length", "num_designs",
    "context_path", "context_sha256", "motif_size", "hotspot_count",
    "ligand_atom_count",
}
DESIGN_FIELDS = {"design_index", "ranking_score", "structure_path", "motif_map"}
HARNESS_FIELDS = {"schema", "kind", "entry_point", "learned_components", "pretrained_assets"}
LEARNED_FIELDS = {"path", "sha256", "parameters", "trained_in_session"}
ASSET_FIELDS = {"asset_id", "sha256", "license", "role"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def contained(root, name):
    """Resolve `name` under `root`, rejecting absolute paths, escapes, and symlinks."""
    root = Path(root).resolve()
    if Path(name).is_absolute():
        raise ValueError(f"unsafe absolute path: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError(f"unsafe path: {name}")
    return path


def jsonl(path, max_bytes=512 * 1024 * 1024, max_line=1024 * 1024):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise ValueError(f"unsafe JSONL: {path.name}")
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if len(line.encode()) > max_line:
                raise ValueError(f"oversized line: {path.name}:{number}")
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON: {path.name}:{number}") from error


def validate_input(root, views=("verifier",)):
    """Validate an input view and return (manifest, requests-by-track).

    The verifier accepts only its own `verifier` view. Development scoring passes
    `views=("verifier", "development")`, because materialize.py labels fit and tune
    `development` and score.py must be able to read them - the documented
    `score.py --input /data/tune` is the whole self-scoring workflow.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    names = {f"{track}.jsonl" for track in TRACKS}
    if (set(manifest) != {"schema", "view", "files", "counts"}
            or manifest["schema"] != INPUT_SCHEMA
            or manifest["view"] not in views
            or set(manifest["files"]) != names
            or set(manifest["counts"]) != set(TRACKS)):
        raise ValueError("invalid input manifest")
    requests = {}
    seen = set()
    for track in TRACKS:
        path = contained(root, f"{track}.jsonl")
        if sha256(path) != manifest["files"][path.name]:
            raise ValueError(f"input hash mismatch: {path.name}")
        rows = {}
        for row in jsonl(path):
            if set(row) != INPUT_FIELDS or row["track"] != track:
                raise ValueError(f"invalid request fields: {track}")
            example_id = row["example_id"]
            if not isinstance(example_id, str) or not example_id or example_id in seen:
                raise ValueError(f"invalid or duplicate example_id: {track}")
            length, count = row["design_length"], row["num_designs"]
            if (type(length) is not int or not MIN_DESIGN_LENGTH <= length <= MAX_DESIGN_LENGTH
                    or type(count) is not int or not 1 <= count <= MAX_DESIGNS_PER_TARGET):
                raise ValueError(f"invalid design length/count: {example_id}")
            for field in ("motif_size", "hotspot_count", "ligand_atom_count"):
                if type(row[field]) is not int or row[field] < 0:
                    raise ValueError(f"invalid {field}: {example_id}")
            if track in MOTIF_TRACKS and row["motif_size"] < 1:
                raise ValueError(f"{track} requires a motif: {example_id}")
            if track == "ppi_binder" and row["hotspot_count"] < 1:
                raise ValueError(f"ppi_binder requires hotspots: {example_id}")
            if track == "unconditional" and (row["motif_size"] or row["hotspot_count"]
                                             or row["ligand_atom_count"]):
                raise ValueError(f"unconditional requests carry no conditioning: {example_id}")
            if row["motif_size"] > length:
                raise ValueError(f"motif larger than design: {example_id}")
            context = contained(root, row["context_path"])
            if sha256(context) != row["context_sha256"]:
                raise ValueError(f"context hash mismatch: {example_id}")
            seen.add(example_id)
            rows[example_id] = row
        if len(rows) != manifest["counts"][track]:
            raise ValueError(f"input count mismatch: {track}")
        requests[track] = rows
    return manifest, requests


def validate_harness(root, pretrained_root):
    """Validate `harness.json` and every component it declares.

    The graded artifact is a design SYSTEM, which may be a trained generator, a
    training-free inversion of a pretrained predictor, or a hybrid. What is enforced is
    that the manifest matches reality: declared learned components exist inside the
    package with the stated digest, and declared pretrained assets exist in the
    operator-mounted store with the stated digest and a license.
    """
    root = Path(root).resolve()
    manifest_path = root / "harness.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise FileNotFoundError("submission/harness.json is required")
    manifest = json.loads(manifest_path.read_text())
    if (set(manifest) != HARNESS_FIELDS
            or manifest["schema"] != HARNESS_SCHEMA
            or manifest["kind"] not in HARNESS_KINDS
            or not isinstance(manifest["entry_point"], str)
            or not isinstance(manifest["learned_components"], list)
            or not isinstance(manifest["pretrained_assets"], list)):
        raise ValueError("invalid harness manifest fields, schema, or kind")

    total_parameters = 0
    for component in manifest["learned_components"]:
        if (set(component) != LEARNED_FIELDS
                or type(component["parameters"]) is not int or component["parameters"] < 0
                or not isinstance(component["trained_in_session"], bool)):
            raise ValueError("invalid learned component declaration")
        path = contained(root, component["path"])
        if sha256(path) != component["sha256"]:
            raise ValueError(f"learned component checksum mismatch: {component['path']}")
        total_parameters += component["parameters"]

    for asset in manifest["pretrained_assets"]:
        if set(asset) != ASSET_FIELDS or not asset["license"] or not asset["role"]:
            raise ValueError("pretrained assets must declare a license and a role")
        path = contained(pretrained_root, asset["asset_id"])
        if sha256(path) != asset["sha256"]:
            raise ValueError(f"pretrained asset checksum mismatch: {asset['asset_id']}")

    kind = manifest["kind"]
    trained = [c for c in manifest["learned_components"] if c["trained_in_session"]]
    if kind in ("trained_generator", "hybrid") and not trained:
        raise ValueError(f"kind '{kind}' requires at least one component trained in session")
    if kind in ("predictor_inversion", "hybrid") and not manifest["pretrained_assets"]:
        raise ValueError(f"kind '{kind}' requires at least one declared pretrained asset")
    manifest["declared_parameters"] = total_parameters
    return manifest
