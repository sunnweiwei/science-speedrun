#!/usr/bin/env python3
"""CPU-runnable V4 unit tests for #18 target-based-drug-screening.

Covers:
1. Unidirectional target-pocket decontamination (`mmseqs_cov_mode=1`, `tcov >= 0.80`)
   across `protocol.json`, `mmseqs.py`, `profsa.py`, and `biolip.py`.
2. Swapped-pocket counterfactual replay diagnostic (`swapped_pocket_counterfactual`
   in `report.json`) distinguishing pocket-specific vs ligand-only predictors while
   keeping the primary score untouched.
3. `< 1e-12` score parity with `harness/ssi_index_v3.py:drug_screening()`, including
   both `"score"` and `"reward"` keys in `reward.json`, `report.json`, and `reaggregate.py`.
4. Strict visibility boundary (`tests/` never copied/mounted into dev `/workspace`,
   and `tests/README.md` contains the 3-part Literature Justification).
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

TASK_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file())


def _install_host_cpu_fallbacks_if_missing() -> None:
    """Provide exact in-memory LMDB and Truchon-Bayly RDKit scoring shims ONLY when
    running on a bare host CPU Python that lacks container-installed C extensions."""
    try:
        import lmdb  # noqa: F401
    except ModuleNotFoundError:
        stores: dict[str, dict[bytes, bytes]] = {}

        class _Cursor:
            def __init__(self, data: dict[bytes, bytes]):
                self._items = list(data.items())

            def __iter__(self):
                return iter(self._items)

        class _Txn:
            def __init__(self, data: dict[bytes, bytes], write: bool):
                self._data = data
                self._write = write

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def stat(self) -> dict[str, int]:
                return {"entries": len(self._data)}

            def get(self, key: bytes):
                return self._data.get(bytes(key))

            def put(self, key: bytes, value: bytes) -> bool:
                if not self._write:
                    raise PermissionError("read-only transaction")
                self._data[bytes(key)] = bytes(value)
                return True

            def cursor(self):
                return _Cursor(self._data)

            def commit(self) -> None:
                pass

        class _Env:
            def __init__(self, path: str, readonly: bool = False):
                self._path = str(Path(path).resolve())
                self._readonly = readonly
                p = Path(self._path)
                if readonly:
                    if self._path not in stores and p.is_file():
                        stores[self._path] = pickle.loads(p.read_bytes())
                else:
                    if p.is_file() and self._path not in stores:
                        stores[self._path] = pickle.loads(p.read_bytes())
                    else:
                        stores.setdefault(self._path, {})

            def begin(self, write: bool = False, buffers: bool = False):
                if write and self._readonly:
                    raise PermissionError("read-only environment")
                return _Txn(stores[self._path], write=write)

            def sync(self) -> None:
                if not self._readonly:
                    Path(self._path).write_bytes(pickle.dumps(stores[self._path], protocol=4))

            def close(self) -> None:
                if not self._readonly and self._path in stores:
                    p = Path(self._path)
                    if p.parent.is_dir():
                        try:
                            p.write_bytes(pickle.dumps(stores[self._path], protocol=4))
                        except PermissionError:
                            pass

        lmdb_mod = types.ModuleType("lmdb")
        lmdb_mod.open = lambda path, subdir=False, readonly=False, **kwargs: _Env(path, readonly=readonly)
        sys.modules["lmdb"] = lmdb_mod

    try:
        from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC, CalcEnrichment  # noqa: F401
    except ModuleNotFoundError:
        def _calc_auc(scores, col: int) -> float:
            arr = np.asarray([[float(r[0]), float(r[col])] for r in scores], dtype=np.float64)
            labels = arr[:, 1]
            n = len(labels)
            n_act = int(labels.sum())
            ranks = np.arange(1, n + 1, dtype=np.float64)
            rank_sum = float(np.sum(ranks[labels == 1]))
            return float(1.0 - (rank_sum - n_act * (n_act + 1) / 2.0) / (n_act * (n - n_act)))

        def _calc_bedroc(scores, col: int, alpha: float) -> float:
            # Exact Truchon & Bayly (JCIM 2007) Eq. 9 & 36 matching RDKit CalcBEDROC
            arr = np.asarray([[float(r[0]), float(r[col])] for r in scores], dtype=np.float64)
            labels = arr[:, 1]
            n = len(labels)
            n_act = int(labels.sum())
            ra = n_act / n
            ranks = np.flatnonzero(labels == 1) + 1
            rie_num = np.sum(np.exp(-alpha * ranks / n))
            rie_den = ra * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n) - 1.0)
            rie = rie_num / rie_den
            rie_min = (1.0 - np.exp(alpha * ra)) / (ra * (1.0 - np.exp(alpha)))
            rie_max = (1.0 - np.exp(-alpha * ra)) / (ra * (1.0 - np.exp(-alpha)))
            return float(np.clip((rie - rie_min) / (rie_max - rie_min), 0.0, 1.0))

        def _calc_enrichment(scores, col: int, fractions) -> list[float]:
            arr = np.asarray([[float(r[0]), float(r[col])] for r in scores], dtype=np.float64)
            labels = arr[:, 1]
            n = len(labels)
            n_act = int(labels.sum())
            out = []
            for f in fractions:
                k = int(np.ceil(n * f))
                out.append(float(labels[:k].sum() * n / (k * n_act)))
            return out

        rdkit_mod = types.ModuleType("rdkit")
        chem_mod = types.ModuleType("rdkit.Chem")
        allchem_mod = types.ModuleType("rdkit.Chem.AllChem")
        ds_mod = types.ModuleType("rdkit.Chem.DataStructs")
        ml_mod = types.ModuleType("rdkit.ML")
        scoring_pkg = types.ModuleType("rdkit.ML.Scoring")
        scoring_mod = types.ModuleType("rdkit.ML.Scoring.Scoring")
        scoring_mod.CalcAUC = _calc_auc
        scoring_mod.CalcBEDROC = _calc_bedroc
        scoring_mod.CalcEnrichment = _calc_enrichment
        rdkit_mod.Chem = chem_mod
        rdkit_mod.RDLogger = types.SimpleNamespace()
        rdkit_mod.rdBase = types.SimpleNamespace()
        chem_mod.AllChem = allchem_mod
        chem_mod.DataStructs = ds_mod
        sys.modules.setdefault("rdkit", rdkit_mod)
        sys.modules.setdefault("rdkit.Chem", chem_mod)
        sys.modules.setdefault("rdkit.Chem.AllChem", allchem_mod)
        sys.modules.setdefault("rdkit.Chem.DataStructs", ds_mod)
        sys.modules.setdefault("rdkit.ML", ml_mod)
        sys.modules.setdefault("rdkit.ML.Scoring", scoring_pkg)
        sys.modules.setdefault("rdkit.ML.Scoring.Scoring", scoring_mod)

    try:
        import tqdm  # noqa: F401
    except ModuleNotFoundError:
        tqdm_mod = types.ModuleType("tqdm")
        tqdm_mod.tqdm = lambda iterable=None, **kwargs: iterable
        sys.modules["tqdm"] = tqdm_mod

    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["requests"] = types.ModuleType("requests")

    try:
        from Bio.PDB.Polypeptide import protein_letters_3to1  # noqa: F401
    except ModuleNotFoundError:
        bio_mod = types.ModuleType("Bio")
        pdb_mod = types.ModuleType("Bio.PDB")
        poly_mod = types.ModuleType("Bio.PDB.Polypeptide")
        pdb_mod.PDBParser = object
        poly_mod.protein_letters_3to1 = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }
        sys.modules.setdefault("Bio", bio_mod)
        sys.modules.setdefault("Bio.PDB", pdb_mod)
        sys.modules.setdefault("Bio.PDB.Polypeptide", poly_mod)


_install_host_cpu_fallbacks_if_missing()

sys.path.insert(0, str(TASK_DIR / "environment" / "data"))
sys.path.insert(0, str(TASK_DIR / "environment"))
sys.path.insert(0, str(TASK_DIR / "environment" / "evaluation"))
sys.path.insert(0, str(REPO_ROOT / "harness"))

import lmdb  # noqa: E402
import ssi_index_v3  # noqa: E402
from aggregation import summarize_targets  # noqa: E402
from drugclip_data.biolip import apply_biolip_protein_decontamination  # noqa: E402
from drugclip_data.config import load_protocol  # noqa: E402
from drugclip_data.mmseqs import (  # noqa: E402
    alignment_coverage,
    is_contaminated_hit,
    parse_mmseqs_tsv,
    select_best_hits,
)
from drugclip_data.profsa import apply_profsa_protein_decontamination  # noqa: E402
from input_view import BENCHMARK_TARGETS, build_pocket_derangement, stage_inputs  # noqa: E402

_grade_spec = importlib.util.spec_from_file_location("v4_screening_grade", TASK_DIR / "tests" / "grade.py")
grader = importlib.util.module_from_spec(_grade_spec)
_grade_spec.loader.exec_module(grader)

_reagg_spec = importlib.util.spec_from_file_location("v4_screening_reagg", TASK_DIR / "tests" / "reaggregate.py")
reaggregator = importlib.util.module_from_spec(_reagg_spec)
_reagg_spec.loader.exec_module(reaggregator)


def _write_lmdb(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = lmdb.open(str(path), subdir=False, map_size=4 << 20)
    with db.begin(write=True) as txn:
        for idx, rec in enumerate(records):
            txn.put(str(idx).encode("ascii"), pickle.dumps(rec, protocol=4))
    db.sync()
    db.close()


class V4DrugScreeningTests(unittest.TestCase):
    def test_protocol_and_multidomain_pocket_decontamination(self) -> None:
        protocol = load_protocol()
        self.assertEqual(protocol["mmseqs_cov_mode"], 1)
        self.assertAlmostEqual(protocol["protein_coverage_threshold"], 0.8)
        self.assertAlmostEqual(protocol["protein_identity_threshold_percent"], 30.0)
        self.assertAlmostEqual(protocol["ligand_tanimoto_threshold"], 0.60)

        # Synthetic 900-aa multi-domain training protein containing a 250-aa test kinase pocket:
        # qcov = 0.278 (< 0.80), tcov = 0.96 (>= 0.80), fident = 0.42 (pident = 42.0 > 30.0).
        multidomain_hit = {
            "query": "1abc_A",
            "target": "dude|akt1|chainA",
            "fident": 0.42,
            "qcov": 0.278,
            "tcov": 0.96,
            "alnlen": 250,
            "qlen": 900,
            "tlen": 250,
            "evalue": 1e-45,
            "bits": 180.0,
        }
        # Legacy cov_mode=0 used min(qcov, tcov)=0.278 < 0.80 and let it leak:
        self.assertAlmostEqual(alignment_coverage(multidomain_hit, cov_mode=0), 0.278)
        self.assertEqual(select_best_hits([multidomain_hit], coverage=0.80, cov_mode=0), {})
        self.assertFalse(is_contaminated_hit(multidomain_hit, 30.0, 0.80, cov_mode=0))

        # V4 cov_mode=1 uses unidirectional target coverage tcov=0.96 >= 0.80 and excludes it:
        self.assertAlmostEqual(alignment_coverage(multidomain_hit, cov_mode=1), 0.96)
        selected = select_best_hits([multidomain_hit], coverage=0.80, cov_mode=1)
        self.assertIn("1abc_A", selected)
        self.assertAlmostEqual(selected["1abc_A"]["pident"], 42.0)
        self.assertTrue(is_contaminated_hit(selected["1abc_A"], 30.0, 0.80, cov_mode=1))

        # TSV parser check + boundary 30.0% retention check
        with tempfile.TemporaryDirectory() as tmp:
            tsv = Path(tmp) / "hits.tsv"
            tsv.write_text(
                "1abc_A\tdude|akt1|chainA\t42.0\t0.278\t0.96\t250\t900\t250\t1e-45\t180.0\n"
                "2def_B\tlpcba|ESR1|chainA\t30.0\t0.310\t0.92\t230\t740\t250\t1e-20\t110.0\n"
                "3ghi_C\tdude|cdk2|chainA\t65.0\t0.150\t0.55\t138\t920\t250\t1e-18\t95.0\n"
            )
            hits_v4 = parse_mmseqs_tsv(tsv, coverage=0.80, cov_mode=1)
            self.assertEqual(set(hits_v4.keys()), {"1abc_A", "2def_B"})

            target_table = pd.DataFrame(
                {"benchmark": ["dude", "lit_pcba", "dude"]},
                index=pd.Index(["akt1", "ESR1", "cdk2"], name="target_id"),
            )
            profsa_df = pd.DataFrame([
                {
                    "sample_id": "s1", "source_pdb_id": "1abc", "source_chain": "A",
                    "exclusion_reason": "", "max_test_sequence_identity": 0.0,
                    "test_alignment_coverage": 0.0, "nearest_test_target": "",
                    "nearest_test_benchmark": "",
                },
                {
                    "sample_id": "s2", "source_pdb_id": "2def", "source_chain": "B",
                    "exclusion_reason": "", "max_test_sequence_identity": 0.0,
                    "test_alignment_coverage": 0.0, "nearest_test_target": "",
                    "nearest_test_benchmark": "",
                },
            ])
            apply_profsa_protein_decontamination(profsa_df, hits_v4, target_table, protocol)
            self.assertEqual(profsa_df.loc[0, "exclusion_reason"], "test_protein_similarity")
            self.assertAlmostEqual(profsa_df.loc[0, "test_alignment_coverage"], 0.96)
            self.assertEqual(profsa_df.loc[1, "exclusion_reason"], "")  # exact 30.0% retained
            self.assertAlmostEqual(profsa_df.loc[1, "test_alignment_coverage"], 0.92)

            biolip_df = pd.DataFrame([
                {
                    "complex_id": "c1",
                    "protein_sequences_json": json.dumps({"A": "SEQ_MULTIDOMAIN_KINASE"}),
                    "exclusion_reason": "",
                    "max_test_sequence_identity": 0.0,
                    "test_alignment_coverage": 0.0,
                    "nearest_test_target": "",
                    "nearest_test_benchmark": "",
                },
                {
                    "complex_id": "c2",
                    "protein_sequences_json": json.dumps({"A": "SEQ_BOUNDARY_30PCT"}),
                    "exclusion_reason": "",
                    "max_test_sequence_identity": 0.0,
                    "test_alignment_coverage": 0.0,
                    "nearest_test_target": "",
                    "nearest_test_benchmark": "",
                },
            ])
            seq_to_query = {
                "SEQ_MULTIDOMAIN_KINASE": "1abc_A",
                "SEQ_BOUNDARY_30PCT": "2def_B",
            }
            apply_biolip_protein_decontamination(
                biolip_df, seq_to_query, hits_v4, target_table, protocol
            )
            self.assertEqual(biolip_df.loc[0, "exclusion_reason"], "test_protein_similarity")
            self.assertAlmostEqual(biolip_df.loc[0, "test_alignment_coverage"], 0.96)
            self.assertEqual(biolip_df.loc[1, "exclusion_reason"], "")

    def test_pocket_derangement_separates_adjacent_paralogs(self) -> None:
        fake_targets = {
            ("dude", name): (Path("m"), Path("p"))
            for name in ("akt1", "akt2", "cdk2", "esr1", "esr2", "fgfr1", "jak2", "mmp13", "nos1", "src")
        }
        fake_targets.update({
            ("lit_pcba", name): (Path("m"), Path("p"))
            for name in ("ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1", "KAT2A")
        })
        derangement = build_pocket_derangement(fake_targets)
        self.assertEqual(set(derangement.keys()), set(fake_targets.keys()))
        self.assertEqual(set(derangement.values()), set(fake_targets.keys()))
        for recipient, donor in derangement.items():
            self.assertNotEqual(recipient, donor)
            self.assertEqual(recipient[0], donor[0])
            # Alphabetically adjacent paralogs (akt1/akt2, esr1/esr2, ESR1_ago/ESR1_ant) do not swap onto each other
            self.assertNotEqual(recipient[1][:3].lower(), donor[1][:3].lower())

    def test_v3_score_parity_and_swapped_pocket_counterfactual_end_to_end(self) -> None:
        # 1. Exact V3 formula parity check across normal & boundary inputs
        for dude_val, lit_val in [
            (0.2945875917123425, 0.03818222131848757),
            (0.0, 0.0),
            (1.0, 1.0),
            (0.812345678901, 0.198765432109),
        ]:
            rows = [
                {
                    "benchmark": bench,
                    "target_id": f"t-{i:03d}",
                    "roc_auc": 0.7,
                    "bedroc_80.5": val,
                    "ef_0.005": 5.0,
                    "ef_0.01": 4.0,
                    "ef_0.02": 3.0,
                    "ef_0.05": 2.0,
                }
                for bench, count, val in (("dude", 102, dude_val), ("lit_pcba", 15, lit_val))
                for i in range(count)
            ]
            summary = summarize_targets(rows, require_full_suite=True)
            v3 = ssi_index_v3.drug_screening(summary["benchmarks"])
            self.assertLess(abs(summary["dataset_balanced_bedroc"] - v3["score"]), 1e-12)
            migrated = reaggregator.reaggregate(
                {"valid": True, "reward": 0.5, "per_target": rows}, "src-sha"
            )
            self.assertLess(abs(migrated["score"] - v3["score"]), 1e-12)
            self.assertLess(abs(migrated["reward"] - v3["score"]), 1e-12)

        # 2. End-to-end grade() test comparing pocket-specific vs ligand-only predictor
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            private_root = tmp_path / "private"
            private_root.mkdir(mode=0o700)
            target_idx = 0
            for bench, count in BENCHMARK_TARGETS.items():
                for i in range(count):
                    target_idx += 1
                    tdir = private_root / "test" / bench / f"target-{i:03d}"
                    # Encode unique pocket coordinate signature = float(target_idx)
                    # Active ligands (first 5 of 50) carry coordinate x = float(target_idx) AND smi = "ACTIVE_CHEMOTYPE"
                    # Decoy ligands carry coordinate x = -1.0 AND smi = "DECOY_CHEMOTYPE"
                    mols = []
                    for m in range(50):
                        is_act = m < 5
                        mols.append({
                            "atoms": ["C", "O"],
                            "coordinates": [
                                np.array(
                                    [[float(target_idx) if is_act else -1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                    dtype=np.float32,
                                )
                            ],
                            "smi": "ACTIVE_CHEMOTYPE" if is_act else "DECOY_CHEMOTYPE",
                            "label": int(is_act),
                        })
                    _write_lmdb(tdir / "mols.lmdb", mols)
                    pocket_file = "pocket.lmdb" if bench == "dude" else "pockets.lmdb"
                    _write_lmdb(tdir / pocket_file, [{
                        "pocket_atoms": ["C", "N"],
                        "pocket_coordinates": np.array([[float(target_idx), 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                    }])

            def _make_pkg(pkg_dir: Path, mode: str) -> None:
                pkg_dir.mkdir(parents=True, exist_ok=True)
                weights = b"v4-unit-test-weights"
                (pkg_dir / "weights.bin").write_bytes(weights)
                (pkg_dir / "config.json").write_text("{}")
                (pkg_dir / "model_manifest.json").write_text(json.dumps({
                    "method": f"unit-{mode}",
                    "parameter_count": 1,
                    "model_config": "config.json",
                    "checkpoint": {"path": "weights.bin", "sha256": hashlib.sha256(weights).hexdigest()},
                }))
                (pkg_dir / "README.md").write_text("Unit test model.")
                # Use the exact same Python interpreter with fallback shims if needed
                helper_py = pkg_dir / "run_infer.py"
                helper_py.write_text(
                    f"""import csv, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, {str(TASK_DIR / "tests")!r})
import test_v4_drug_screening as _shim  # installs lmdb fallback if needed
import lmdb

in_root, out_root = Path(sys.argv[1]), Path(sys.argv[2])
mode = {mode!r}
with (out_root / "scores.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(("benchmark", "target_id", "ligand_index", "score"))
    for bench, p_name in (("dude", "pocket.lmdb"), ("lit_pcba", "pockets.lmdb")):
        bdir = in_root / "test" / bench
        for tdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            pdb = lmdb.open(str(tdir / p_name), subdir=False, readonly=True, lock=False)
            with pdb.begin() as txn:
                prec = pickle.loads(txn.get(b"0"))
            pdb.close()
            pocket_sig = float(prec["pocket_coordinates"][0, 0])
            mdb = lmdb.open(str(tdir / "mols.lmdb"), subdir=False, readonly=True, lock=False)
            with mdb.begin() as txn:
                n = txn.stat()["entries"]
                for idx in range(n):
                    mrec = pickle.loads(txn.get(str(idx).encode("ascii")))
                    if mode == "pocket_specific":
                        mol_sig = float(mrec["coordinates"][0][0, 0])
                        sc = 1.0 if abs(mol_sig - pocket_sig) < 1e-3 else 0.0
                    else:
                        sc = 1.0 if mrec["smi"] == "ACTIVE_CHEMOTYPE" else 0.0
                    w.writerow((bench, tdir.name, idx, sc))
            mdb.close()
"""
                )
                (pkg_dir / "infer.sh").write_text(
                    f'#!/usr/bin/env bash\nset -euo pipefail\n"{sys.executable}" run_infer.py "$1" "$2"\n'
                )

            with (
                mock.patch.object(grader, "require_private_boundary", lambda p: None),
                mock.patch.object(grader, "demote", lambda: None),
                mock.patch.object(grader.os, "chown", lambda *a, **kw: None),
            ):
                # Model A: pocket-specific predictor
                pkg_a = tmp_path / "pkg_specific"
                out_a = tmp_path / "out_specific"
                _make_pkg(pkg_a, "pocket_specific")
                res_a = grader.grade(pkg_a, private_root, out_a, timeout=60.0)
                self.assertTrue(res_a["valid"])
                self.assertAlmostEqual(res_a["score"], 1.0, places=12)
                self.assertAlmostEqual(res_a["reward"], 1.0, places=12)
                cf_a = res_a["swapped_pocket_counterfactual"]
                self.assertEqual(cf_a["status"], "ok")
                # On swapped pockets, pocket_sig never matches mol_sig, so all scores tie at 0 -> random BEDROC
                self.assertLess(cf_a["dataset_balanced_bedroc"], 0.2)
                self.assertGreater(cf_a["pocket_specificity_delta"], 0.8)
                v3_a = ssi_index_v3.drug_screening(res_a["benchmarks"])
                self.assertLess(abs(res_a["score"] - v3_a["score"]), 1e-12)
                rew_json_a = json.loads((out_a / "reward.json").read_text())
                self.assertLess(abs(rew_json_a["score"] - v3_a["score"]), 1e-12)
                self.assertLess(abs(rew_json_a["reward"] - v3_a["score"]), 1e-12)

                # Model B: ligand-only shortcut predictor
                pkg_b = tmp_path / "pkg_ligand_only"
                out_b = tmp_path / "out_ligand_only"
                _make_pkg(pkg_b, "ligand_only")
                res_b = grader.grade(pkg_b, private_root, out_b, timeout=60.0)
                self.assertTrue(res_b["valid"])
                cf_b = res_b["swapped_pocket_counterfactual"]
                self.assertEqual(cf_b["status"], "ok")
                self.assertLess(abs(cf_b["pocket_specificity_delta"]), 1e-12)
                # Primary score is STILL strictly 0.5 * DUD-E + 0.5 * LIT-PCBA (no ad-hoc gate!)
                v3_b = ssi_index_v3.drug_screening(res_b["benchmarks"])
                self.assertLess(abs(res_b["score"] - v3_b["score"]), 1e-12)

                # Invalid package -> score=0.0, reward=0.0
                pkg_bad = tmp_path / "pkg_bad"
                pkg_bad.mkdir()
                out_bad = tmp_path / "out_bad"
                res_bad = grader.grade(pkg_bad, private_root, out_bad, timeout=10.0)
                self.assertFalse(res_bad["valid"])
                self.assertEqual(res_bad["score"], 0.0)
                self.assertEqual(res_bad["reward"], 0.0)
                self.assertEqual(
                    json.loads((out_bad / "reward.json").read_text()),
                    {"score": 0.0, "reward": 0.0},
                )

    def test_visibility_boundary_and_readme_justification(self) -> None:
        dev_dockerfile = (TASK_DIR / "environment" / "Dockerfile").read_text()
        dev_compose = (TASK_DIR / "environment" / "docker-compose.yaml").read_text()
        self.assertNotIn("COPY tests", dev_dockerfile)
        self.assertNotIn("/tests", dev_dockerfile)
        self.assertNotIn("tests/", dev_compose)

        readme = (TASK_DIR / "tests" / "README.md").read_text()
        for heading in (
            "## Scientific & Literature Justification of Verifier Design",
            "### 1. Primary Metric & Upstream Paper Provenance",
            "### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted",
            "### 3. Diagnostic Signals Retained in report.json",
        ):
            self.assertIn(heading, readme)


if __name__ == "__main__":
    unittest.main()
