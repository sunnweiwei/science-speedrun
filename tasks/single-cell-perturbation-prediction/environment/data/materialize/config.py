"""Release configuration: sources, roles, caps and seeds. Everything here is recorded in
DATA_ROOT/manifest.json so a release is reproducible from the pinned sources."""

RELEASE = "virtual-cell-zeroshot/v1"
SEED = 20260921

# Sub-sampling caps are development-side compression only (see data/README.md). Pseudobulk
# tables always cover every cell of the source. Test contexts are never subsampled.
SOURCES = [
    # name, context, role, loader, relative cache path(s), per-target cap, control cap, chemistry
    dict(name="k562_gwps", context="k562", role="train", loader="replogle",
         path="replogle2022/K562_gwps_raw_singlecell_01.h5ad", keep_per_target=64, keep_controls=20000,
         chemistry="10x_3p_v3", source="replogle2022"),
    dict(name="k562_essential", context="k562", role="train", loader="replogle",
         path="replogle2022/K562_essential_raw_singlecell_01.h5ad", keep_per_target=200, keep_controls=20000,
         chemistry="10x_3p_v3", source="replogle2022"),
    dict(name="jurkat", context="jurkat", role="train", loader="replogle",
         path="nadig2025/jurkat_raw_singlecell_01.h5ad", keep_per_target=200, keep_controls=20000,
         chemistry="10x_3p_v3", source="nadig2025"),
    dict(name="hct116", context="hct116", role="train", loader="orion", line="HCT116",
         path="orion/X-Atlas-Orion", keep_per_target=40, keep_controls=20000,
         chemistry="10x_gemx_5p", source="xatlas_orion"),
    dict(name="hek293t", context="hek293t", role="train", loader="orion", line="HEK293T",
         path="orion/X-Atlas-Orion", keep_per_target=40, keep_controls=20000,
         chemistry="10x_gemx_5p", source="xatlas_orion"),
    dict(name="hepg2", context="ctx_near", role="test", loader="replogle",
         path="nadig2025/hepg2_raw_singlecell_01.h5ad", keep_per_target=None, keep_controls=None,
         chemistry="10x_3p_v3", source="nadig2025", identity="HepG2", tier="near"),
    dict(name="rpe1", context="ctx_mid", role="test", loader="replogle",
         path="replogle2022/rpe1_raw_singlecell_01.h5ad", keep_per_target=None, keep_controls=None,
         chemistry="10x_3p_v3", source="replogle2022", identity="RPE1", tier="mid"),
    dict(name="h1", context="ctx_far", role="test", loader="vcc",
         path=["vcc2025/adata_Training.h5ad.generation-1765904883947296.h5ad",
               "vcc2025/adata_Validation.h5ad.generation-1765905029341737.h5ad",
               "vcc2025/adata_Test.h5ad.generation-1765905295402700.h5ad"],
         keep_per_target=None, keep_controls=None,
         chemistry="10x_flex", source="vcc2025", identity="H1 hESC", tier="far"),
]

TEST = dict(
    n_requests=300,            # per test context
    n_withheld=80,             # globally withheld targets (shared across test contexts)
    n_leaderboard=100,         # requests whose scores are returned to the agent
    min_cells_test_target=40,  # eligibility of a seen target as a near/mid request
    min_cells_withheld=20,     # a withheld target must have >= this many cells in every test context
    max_response_cells=1000,   # private responses per target
    max_basal_cells=10000,
    n_cells_request={"10x_3p_v3": 200, "10x_flex": 400, "10x_gemx_5p": 200},
)

DEV = dict(n_requests=300, min_cells=40, max_basal_cells=10000, n_cells_request=200)

# Q1 effect-pattern strata: per training context, a target is "strong" if its number of
# |z|>z_threshold genes is in the top tertile of that context's targets (with >= min_cells_measured
# cells) and "weak" if in the bottom tertile; conserved-strong / weak = same call in every
# measured context (one dissent allowed when >= 3 are measured), context-variable = otherwise
STRATA = dict(strong_quantile=2 / 3, weak_quantile=1 / 3, z_threshold=3.0, min_cells_measured=20)

ORION_QC = dict(max_mito=20.0, min_counts=2000.0)

# Test contexts are anonymised in everything under dev/: which public screen each one comes from
# (source names, per-source stats, download entries) stays in the operator manifest.
_TEST = [s for s in SOURCES if s["role"] == "test"]
TEST_ALIAS = {s["name"]: s["context"] for s in _TEST}
TEST_PATHS = {p for s in _TEST for p in (s["path"] if isinstance(s["path"], list) else [s["path"]])}


def agent_manifest(manifest: dict) -> dict:
    """The copy of DATA_ROOT/manifest.json shipped as dev/manifest.json."""
    out = {k: v for k, v in manifest.items() if k not in ("sources", "source_stats")}
    src = manifest["sources"]
    out["sources"] = {**src, "files": {k: v for k, v in src["files"].items() if v.get("path") not in TEST_PATHS}}
    out["source_stats"] = {k: v for k, v in manifest["source_stats"].items() if k not in TEST_ALIAS}
    out["test_context_provenance"] = "withheld from the development view; recorded in the operator manifest"
    return out
