"""CPU tests for the ZeroForce + D3(BJ) MOF bulk-modulus anchor measurement pipeline."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import aggregate
import measure_mof_zero_force_d3_anchor as mof_anchor


class _DummyAtoms:
    def __init__(self, n_atoms: int, v0_ang3: float, b0_d3_gpa: float) -> None:
        self._n_atoms = n_atoms
        self.v0 = v0_ang3
        self.b0_d3_gpa = b0_d3_gpa

    def __len__(self) -> int:
        return self._n_atoms

    def copy(self) -> "_DummyAtoms":  # ase's Calculator.calculate stores atoms.copy()
        return _DummyAtoms(self._n_atoms, self.v0, self.b0_d3_gpa)


class _SyntheticSumZeroPlusD3:
    """Pairs ZeroForceCalculator (exact 0) with an analytical Birch-Murnaghan D3(BJ) dispersion curve."""

    EV_PER_ANG3_PER_GPA = 1.0 / 160.21766208

    def __init__(self, zero_calc: mof_anchor.ZeroForceCalculator) -> None:
        self.zero_calc = zero_calc
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.zero_calc.reset()

    def energy_at_volume(self, atoms: _DummyAtoms, volume: float) -> float:
        self.zero_calc.calculate(atoms)
        e_zero = float(self.zero_calc.results["energy"])
        assert e_zero == 0.0
        assert np.array_equal(self.zero_calc.results["forces"], np.zeros((len(atoms), 3)))
        assert np.array_equal(self.zero_calc.results["stress"], np.zeros(6))
        # Exact 2nd-order Birch-Murnaghan dispersion curve E_D3(V) with bulk modulus b0_d3_gpa
        eta = (atoms.v0 / volume) ** (2.0 / 3.0) - 1.0
        b0_ev_ang3 = atoms.b0_d3_gpa * self.EV_PER_ANG3_PER_GPA
        e_d3 = -12.5 + (9.0 / 8.0) * atoms.v0 * b0_ev_ang3 * (eta ** 2)
        return e_zero + e_d3


def _fit_birch_murnaghan_b0_gpa(volumes: list[float], energies: list[float]) -> float:
    """2nd-order Birch-Murnaghan linear least-squares fit in f = ((V0/V)**(2/3) - 1)/2."""
    vols = np.asarray(volumes, dtype=np.float64)
    ens = np.asarray(energies, dtype=np.float64)
    v_ref = float(vols[np.argmin(ens)])
    x = (v_ref / vols) ** (2.0 / 3.0) - 1.0
    coeffs = np.polyfit(x, ens, deg=3)
    # Near x = 0, E(x) = c3*x^3 + c2*x^2 + c1*x + c0; at extremum d2E/dV2 gives B0 = (8/9)*c2/V0
    c2 = float(coeffs[1])
    b0_ev_ang3 = (8.0 / 9.0) * c2 / v_ref
    return float(b0_ev_ang3 * 160.21766208)


def _synthetic_mof_runner(structure_path: Path, calc: _SyntheticSumZeroPlusD3) -> dict:
    spec = json.loads(structure_path.read_text())
    if spec.get("collapse"):
        raise RuntimeError("Pure attractive D3 dispersion collapsed cell without short-range repulsion")
    atoms = _DummyAtoms(spec["n_atoms"], spec["v0_ang3"], spec["d3_b0_gpa"])
    vols, ens = [], []
    for scale in np.linspace(0.96, 1.04, 11):
        v = atoms.v0 * float(scale)
        vols.append(v)
        ens.append(calc.energy_at_volume(atoms, v))
    b_gpa = _fit_birch_murnaghan_b0_gpa(vols, ens)
    return {"B_GPa": b_gpa, "v0": atoms.v0, "atoms": len(atoms), "volumes": vols, "energies": ens}


def test_zero_force_calculator_outputs_exact_zeros():
    calc = mof_anchor.ZeroForceCalculator()
    dummy = _DummyAtoms(n_atoms=12, v0_ang3=1000.0, b0_d3_gpa=4.0)
    calc.calculate(dummy)
    assert calc.results["energy"] == 0.0
    assert calc.results["free_energy"] == 0.0
    assert calc.results["forces"].shape == (12, 3)
    assert np.all(calc.results["forces"] == 0.0)
    assert calc.results["stress"].shape == (6,)
    assert np.all(calc.results["stress"] == 0.0)


def test_measure_and_atomic_update_on_synthetic_two_mof_fixture(tmp_path):
    inputs = tmp_path / "mof" / "inputs"
    refs = tmp_path / "mof" / "references"
    (inputs / "structures").mkdir(parents=True)
    refs.mkdir(parents=True)

    # Structure 1: ZeroForce + D3(BJ) gives B0 ≈ 4.2 GPa vs PBE-D3 reference 15.0 GPa (~72% APE != 100%)
    # Structure 2: ZeroForce + D3(BJ) gives B0 ≈ 26.5 GPa vs PBE-D3 reference 12.0 GPa (~120.83% APE != 100%)
    (inputs / "structures" / "mof_1.json").write_text(
        json.dumps({"n_atoms": 24, "v0_ang3": 1400.0, "d3_b0_gpa": 4.2})
    )
    (inputs / "structures" / "mof_2.json").write_text(
        json.dumps({"n_atoms": 36, "v0_ang3": 2100.0, "d3_b0_gpa": 26.5})
    )
    (inputs / "manifest.json").write_text(
        json.dumps({"suite": "mof", "files": {"mof_1": "mof_1.json", "mof_2": "mof_2.json"}})
    )
    (refs / "references.json").write_text(
        json.dumps([{"id": "mof_1", "B0_GPa": 15.0}, {"id": "mof_2", "B0_GPa": 12.0}])
    )

    receipt = mof_anchor.measure_mof_zero_force_anchor(
        inputs,
        refs,
        device="cpu",
        d3_calculator_factory=lambda device, zc: _SyntheticSumZeroPlusD3(zc),
        mof_runner=_synthetic_mof_runner,
    )
    mape = receipt["mof_bulk_modulus_MAPE_pct"]
    expected_ape_1 = abs(4.2 - 15.0) / 15.0
    expected_ape_2 = abs(26.5 - 12.0) / 12.0
    assert mape == pytest.approx(100.0 * 0.5 * (expected_ape_1 + expected_ape_2), rel=1e-6)
    assert abs(mape - 100.0) > 1.0  # proves ZeroForce + D3(BJ) is NOT 100.0%
    assert receipt["items"] == 2 and receipt["failed"] == 0
    assert receipt["verification"]["abs_mof_component_lt_1e_15"] is True
    assert abs(receipt["verification"]["mof_component_at_measured_anchor"]) < 1e-15

    # Verify atomic update of both container anchor JSON and a copy of harness ssi_index_v3_inputs.json
    anchors_copy = tmp_path / "atom_zero_force_anchors.json"
    anchors_copy.write_text((aggregate.ANCHORS_JSON_PATH).read_text())

    harness_dir = next(p for p in Path(__file__).resolve().parents if (p / "harness" / "ssi_index_v3.py").is_file()) / "harness"
    harness_inputs_copy = tmp_path / "ssi_index_v3_inputs.json"
    harness_inputs_copy.write_text((harness_dir / "ssi_index_v3_inputs.json").read_text())

    mof_anchor.update_anchor_files(
        receipt,
        anchors_json_path=anchors_copy,
        harness_inputs_path=harness_inputs_copy,
    )

    loaded_anchors = aggregate.load_atom_zero_force_anchors(anchors_path=anchors_copy)
    assert loaded_anchors["mof_bulk_modulus_MAPE_pct"] == mape

    zero_force_metrics = {
        "discovery_F1": aggregate.DISCOVERY_F1_BASE,
        **loaded_anchors,
        "mof_bulk_modulus_MAPE_pct": mape,
    }
    v4_scored = aggregate.ssi_v4_score(zero_force_metrics, anchors_path=anchors_copy)
    assert abs(v4_scored["components"]["Metal\u2013organic frameworks"]) < 1e-15
    assert abs(v4_scored["signed_score"]) < 1e-15
    assert abs(v4_scored["score"]) < 1e-15

    # Load harness/ssi_index_v3.py pointed at the updated ssi_index_v3_inputs.json copy and check < 1e-15 parity
    harness_code = (harness_dir / "ssi_index_v3.py").read_text()
    spec = importlib.util.spec_from_loader("ssi_index_v3_isolated", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(tmp_path / "ssi_index_v3.py")
    exec(compile(harness_code, mod.__file__, "exec"), mod.__dict__)
    v3_scored = mod.atom(zero_force_metrics)
    assert abs(v3_scored["components"]["Metal\u2013organic frameworks"]) < 1e-15
    assert abs(v3_scored["score"] - v4_scored["score"]) < 1e-15


def test_collapsed_mof_scores_cap_and_cli_export_submission(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    refs = tmp_path / "refs"
    (inputs / "structures").mkdir(parents=True)
    refs.mkdir(parents=True)
    (inputs / "structures" / "ok.json").write_text(
        json.dumps({"n_atoms": 10, "v0_ang3": 800.0, "d3_b0_gpa": 5.0})
    )
    (inputs / "structures" / "bad.json").write_text(json.dumps({"collapse": True}))
    (inputs / "manifest.json").write_text(
        json.dumps({"suite": "mof", "files": {"ok": "ok.json", "bad": "bad.json"}})
    )
    (refs / "references.json").write_text(
        json.dumps([{"id": "ok", "B0_GPa": 10.0}, {"id": "bad", "B0_GPa": 10.0}])
    )

    monkeypatch.setattr(
        mof_anchor,
        "build_zero_force_d3_calculator",
        lambda device="cpu", **kw: _SyntheticSumZeroPlusD3(mof_anchor.ZeroForceCalculator()),
    )
    monkeypatch.setattr(
        mof_anchor,
        "measure_mof_zero_force_anchor",
        lambda inp, ref, **kw: mof_anchor.measure_mof_zero_force_anchor.__wrapped__(inp, ref, **kw)
        if hasattr(mof_anchor.measure_mof_zero_force_anchor, "__wrapped__")
        else None,
    )
    # Restore and patch extra.run_mof via a lightweight fake module
    monkeypatch.undo()
    import sys
    import types

    fake_extra = types.ModuleType("extra")
    fake_extra.run_mof = _synthetic_mof_runner
    monkeypatch.setitem(sys.modules, "extra", fake_extra)
    monkeypatch.setattr(
        mof_anchor,
        "build_zero_force_d3_calculator",
        lambda device="cpu", **kw: _SyntheticSumZeroPlusD3(mof_anchor.ZeroForceCalculator()),
    )

    receipt_out = tmp_path / "mof_zero_force_d3_receipt.json"
    export_dir = tmp_path / "zero_force_submission"
    rc = mof_anchor.main(
        [
            "--inputs",
            str(inputs),
            "--references",
            str(refs),
            "--device",
            "cpu",
            "--output",
            str(receipt_out),
            "--export-submission",
            str(export_dir),
        ]
    )
    assert rc == 0
    saved = json.loads(receipt_out.read_text())
    assert saved["items"] == 2 and saved["failed"] == 1
    assert saved["mof_bulk_modulus_MAPE_pct"] == pytest.approx(100.0 * 0.5 * (0.5 + 2.0), rel=1e-6)
    assert (export_dir / "calculator.py").is_file()
    assert (export_dir / "README.md").is_file()
