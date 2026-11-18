"""Shared unconditional CrystalGen evaluation orchestration."""

import json
import math
from pathlib import Path
import re
import tempfile
import time

from .contract import load_output, load_requests
from .experimental_v2 import formation_energies, negative_formation_low_energy_score
from .upstream import normalize_result, run_lemat
from .relaxation import RECIPE, relax_records
from .displacement import sample_slots
from .mace_assets import CHECKPOINTS

_FORMULA_LINE_RE = re.compile(
    r"^\s*_chemical_formula_(?:sum|structural|moiety)\s+['\"]?([^'\"\r\n]+)['\"]?\s*$",
    re.MULTILINE,
)
_ELEMENT_TOKEN_RE = re.compile(r"([A-Z][a-z]?)")
_VALID_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def harbor_reward(values):
    """Harbor parses reward.json as a flat object of finite numbers; strings, None and nested details
    belong in report.json / status.json."""
    import math as _math
    return {k: v for k, v in values.items() if isinstance(v, (int, float)) and _math.isfinite(v)}


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def _cif_arity(path: Path, pymatgen_structure_cls) -> int | None:
    if pymatgen_structure_cls is not None:
        try:
            structure = pymatgen_structure_cls.from_file(path)
            comp = getattr(structure, "composition", None)
            elems = getattr(comp, "elements", None)
            if elems is not None:
                return len(elems)
        except Exception:
            pass
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return None
    match = _FORMULA_LINE_RE.search(text)
    if not match:
        return None
    symbols = {tok for tok in _ELEMENT_TOKEN_RE.findall(match.group(1)) if tok in _VALID_ELEMENTS}
    return len(symbols) if symbols else None


def element_arity_distribution(records: dict, selected: set[str], upstream_doc: dict | None = None) -> dict:
    """Diagnostic-only element-arity (n_elements) distribution; never used as a reward gate."""
    if isinstance(upstream_doc, dict) and isinstance(upstream_doc.get("n_elements_distribution"), dict):
        counts = {str(k): int(v) for k, v in upstream_doc["n_elements_distribution"].items()}
        total = sum(counts.values())
        return {
            "counts_by_arity": counts,
            "fractions_by_arity": {k: (v / total if total else 0.0) for k, v in counts.items()},
            "parsed_structures": total,
        }
    try:
        from pymatgen.core import Structure as _PymatgenStructure
    except ImportError:
        _PymatgenStructure = None
    counts: dict[int, int] = {}
    for (req_id, _), cif_path in sorted(records.items()):
        if req_id not in selected or not isinstance(cif_path, Path) or not cif_path.is_file():
            continue
        arity = _cif_arity(cif_path, _PymatgenStructure)
        if arity is not None and arity > 0:
            counts[arity] = counts.get(arity, 0) + 1
    ordered = {str(k): counts[k] for k in sorted(counts)}
    total = sum(ordered.values())
    return {
        "counts_by_arity": ordered,
        "fractions_by_arity": {k: (v / total if total else 0.0) for k, v in ordered.items()},
        "parsed_structures": total,
    }


def _negative_formation_fraction(upstream_doc: dict | None, valid_count: int) -> float | None:
    if not valid_count:
        return 0.0
    if not isinstance(upstream_doc, dict):
        return None
    try:
        energies = formation_energies(upstream_doc, valid_count)
    except Exception:
        return None
    neg = sum(1 for e in energies if isinstance(e, (int, float)) and math.isfinite(e) and e < 0.0)
    return neg / valid_count


def evaluate(
    input_root: Path,
    generated_root: Path,
    scorer_root: Path,
    scorer_python: Path,
    fixture_results: Path | None = None,
    artifact_root: Path | None = None,
    skip_relaxation: bool = False,
) -> tuple[dict, dict]:
    requests = load_requests(input_root)
    raw_records, output_errors = load_output(generated_root, requests)
    total_expected = sum(request["n_samples"] for request in requests.values())
    raw_present_count = len(raw_records)
    records = raw_records
    fixture = json.loads(fixture_results.read_text()) if fixture_results else None
    rewards = {}
    version = "v3-submitted-geometry" if skip_relaxation else "v4-uma200-hull-aligned"
    report = {"schema": f"crystalgen-report/{version}", "output_errors": output_errors, "tracks": {},
              "protocol": f"crystalgen-evaluation/{version}",
              "primary_geometry": "submitted" if skip_relaxation else "uma200_relaxed",
              "pre_scoring_relaxation": not skip_relaxation,
              "mace_energy_model": "medium-0b3", "mace_embedding_model": "medium-mpa-0",
              "mace_checkpoint_sha256": CHECKPOINTS,
              "hull_revision": "70d505bb294c16658a9551e2b81ace69a28d9790",
              "matcher_version": "mattergen-v1.0.3-or-ssi-dilute-0.25",
              "matcher_property_policy": "finite-element-properties/v1",
              "matcher": "MatterGen v1.0.3 + SSI dilute-substitution guard",
              "hull_quantity": "e_above_hull_mean across ORB-v3, MACE-MP and UMA on the scored geometry",
              "reward_formula": "(SUN + 0.5 * MetaSUN) / expected"}
    with tempfile.TemporaryDirectory(prefix="crystalgen-score-") as directory:
        work = Path(directory)
        if fixture is not None:
            report["relaxation"] = {"status": "fixture_not_run"}
        elif skip_relaxation:
            report["relaxation"] = {"status": "as_submitted_sensitivity"}
        else:
            records, report["relaxation"] = relax_records(
                records, total_expected,
                (artifact_root or work) / "relaxation", scorer_python,
            )
            if report["relaxation"].get("recipe") != RECIPE:
                raise RuntimeError("Relaxation did not follow the frozen task recipe")
        scoring_started = time.monotonic()
        for mode in ("unconditional",):
            selected = {key for key, request in requests.items() if request["mode"] == mode}
            expected = sum(requests[key]["n_samples"] for key in selected)
            if not selected:
                continue
            if fixture is not None:
                rates, details = normalize_result(fixture[mode], expected)
            else:
                rates, details = run_lemat(
                    records, selected, expected, work / mode, scorer_root, scorer_python,
                    rmsd_slots=sample_slots({key: requests[key] for key in selected}) if skip_relaxation else []
                )
            for name, value in rates.items():
                rewards[f"{mode}_{name}"] = value
            valid_count = int(details.get("counts", {}).get("valid", 0) or 0)
            if valid_count:
                details["negative_formation"] = negative_formation_low_energy_score(details["upstream"], expected)
                rewards[f"{mode}_negative_formation_unique_novel"] = details["negative_formation"]["reward"]
            else:
                rewards[f"{mode}_negative_formation_unique_novel"] = 0.0
            neg_form_frac = _negative_formation_fraction(details.get("upstream"), valid_count)
            arity_dist = element_arity_distribution(records, selected, details.get("upstream"))
            details["n_elements"] = arity_dist
            details["negative_formation_energy_fraction"] = neg_form_frac
            diagnostics = details.get("displacement", {})
            if diagnostics.get("mean_A") is not None:
                rewards[f"{mode}_relaxation_rmsd_A"] = diagnostics["mean_A"]
            report["tracks"][mode] = details
        # Primary SSI V4 / V3 score: (a + b) / 2 = (SUN + 0.5 * MetaSUN) / expected, where
        # a = SUN / expected (delta E_hull <= 0) and b = (SUN + MetaSUN) / expected (delta E_hull <= 0.1 eV/atom).
        # E_form < 0, relaxation RMSD, as-submitted counts, and element-arity (n_elements) are diagnostics ONLY.
        if "unconditional_ssi_score" in rewards:
            primary_score = rewards["unconditional_ssi_score"]
        elif "unconditional_sun" in rewards and "unconditional_sun_metasun" in rewards:
            primary_score = 0.5 * (rewards["unconditional_sun"] + rewards["unconditional_sun_metasun"])
        elif "unconditional_sun" in rewards and "unconditional_metasun" in rewards:
            primary_score = rewards["unconditional_sun"] + 0.5 * rewards["unconditional_metasun"]
        else:
            primary_score = rewards.get("unconditional_sun_metasun", 0.0)
        rewards["score"] = primary_score
        rewards["reward"] = primary_score
        uncond_details = report["tracks"].get("unconditional", {})
        report["diagnostics"] = {
            "gating_policy": "diagnostics_only_no_adhoc_gates",
            "strict_sun_yield": rewards.get("unconditional_sun", 0.0),
            "sun_plus_metasun_yield": rewards.get("unconditional_sun_metasun", 0.0),
            "ssi_v4_weighted_yield": primary_score,
            "negative_formation_energy_fraction": uncond_details.get("negative_formation_energy_fraction"),
            "negative_formation_unique_novel_yield": rewards.get("unconditional_negative_formation_unique_novel", 0.0),
            "relaxation_rmsd": uncond_details.get("displacement", {}),
            "n_elements": uncond_details.get("n_elements", {}),
            "as_submitted": (
                {"evaluated_directly": True, "counts": uncond_details.get("counts", {})}
                if skip_relaxation
                else {
                    "evaluated_directly": False,
                    "pre_relaxation_present_cifs": raw_present_count,
                    "pre_relaxation_missing_or_invalid_manifest": total_expected - raw_present_count,
                    "fallback_unrelaxed_retained": (
                        report["relaxation"].get("fallback_original")
                        if isinstance(report.get("relaxation"), dict)
                        else None
                    ),
                }
            ),
        }
        report["scoring_seconds"] = time.monotonic() - scoring_started
    return clean_json(rewards), clean_json(report)


def write_results(output_root: Path, rewards: dict, report: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    # reward.json is the completion signal observed by the other container.
    for name, payload in (("report.json", report), ("reward.json", harbor_reward(rewards))):
        temp = output_root / (name + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temp.replace(output_root / name)
