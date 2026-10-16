"""Audit paper-table resource candidates; never select or validate benchmark cases."""

from collections import Counter
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parameters(group):
    if "conditions" in group:
        return [{"Re": re, "angle": angle} for re, angle in group["conditions"]]
    if "friction_reynolds" in group:
        return [{"Retau": re} for re in group["friction_reynolds"]]
    if "reynolds_range" in group:
        return [{"Re_range": group["reynolds_range"]}]
    if "pressure_ratio" in group:
        return [{"pressure_ratio": group["pressure_ratio"]}]
    return [dict(zip(("Re", "angle", "Mach"), values))
            for values in itertools.product(group["reynolds"],
                                             group.get("angles", [None]),
                                             group.get("mach", [None]))]


def candidates(group, values, inventory):
    """Keep related-but-different parameters separate from actual candidates."""
    family, dim = group["family"], group["dimension"]
    related = []
    if family == "dra2303":
        suffix = {0.2: "Ma02", 0.7: "Ma07"}[values["Mach"]]
        names = [n for n in group["resource_candidates"] if n.endswith(suffix)]
    elif family == "turbulent-channel":
        names = [f"TCFmini_3D_Re{values['Retau']}"]
        related = [n for n in group["resource_candidates"] if n not in names]
    else:
        patterns = group.get("resource_patterns", [group.get("resource_pattern")])
        names = [pattern.format(**values) for pattern in patterns if pattern]
        # Firedrake mesh fidelities are candidates, never separate case counts.
        if "firedrake" in group["solvers"]:
            names += [n for base in list(names) for n in inventory
                      if n.startswith(base + "_") and n.endswith("_FD")]
        if family == "cavity" and dim == 2 and values.get("Re") == 4200:
            related = ["Cavity_2D_Re4140", "Cavity3Jet_2D_Re4140"]
    return ([n for n in dict.fromkeys(names) if n in inventory],
            [n for n in related if n in inventory])


def audit(coverage, catalog):
    if coverage["resource_revision"] != catalog["revision"]:
        raise ValueError("Coverage and resource inventory revisions differ")
    inventory = {r["name"]: r for r in catalog["directories"]}
    if len(inventory) != len(catalog["directories"]):
        raise ValueError("Duplicate resource directory")
    rows = []
    for group in coverage["groups"]:
        for values in parameters(group):
            values = {k: v for k, v in values.items() if v is not None}
            names, related = candidates(group, values, inventory)
            payload = [n for n in names if inventory[n]["bytes"] > 0]
            markers = [n for n in names if inventory[n]["bytes"] == 0]
            if payload:
                status = "partial_variant_payload" if markers else "candidate_payload_unvalidated"
            elif related:
                status = "parameter_mismatch"
            else:
                status = "missing_payload"
            rows.append({"paper_row": len(rows) + 1,
                         "family": group["family"], "dimension": group["dimension"],
                         "parameters": values, "resource_status": status,
                         "candidate_payloads": payload, "marker_only": markers,
                         "related_not_equivalent": related,
                         "solver_candidates": group["solvers"],
                         "evaluation_status": "not_full_suite_validated",
                         "notes": group.get("notes", "Payload presence does not establish paper parity.")})
    counts = Counter(row["resource_status"] for row in rows)
    return {"status": "resource-audit-not-benchmark-release",
            "source_revision": catalog["revision"],
            "denominator": "Paper table parameter rows, not expanded benchmark cases or the advertised 61 environments",
            "paper_rows": len(rows), "resource_counts": dict(sorted(counts.items())),
            "missing_payload_fraction": counts["missing_payload"] / len(rows),
            "scope_exclusions": [],
            "warning": "Missing fraction is a lower-bound inventory diagnostic; candidate payloads may still be incomplete or incompatible. No cases are excluded by this audit.",
            "rows": rows}


if __name__ == "__main__":
    coverage = json.loads((ROOT / "paper-coverage.json").read_text())
    catalog = json.loads((ROOT / "catalog.json").read_text())
    print(json.dumps(audit(coverage, catalog), indent=2, allow_nan=False))
