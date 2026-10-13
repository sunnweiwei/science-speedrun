"""8-bit train-target storage round-trips within metric tolerance."""
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "environment" / "baseline" / "alphagenome-reference"))
from agbench.quantize import decode, encode  # noqa: E402


def test_dense_round_trip_keeps_metric_space_pearson():
    rng = np.random.default_rng(0)
    counts = rng.gamma(0.3, 40.0, size=(1536, 64)) * (rng.random((1536, 64)) < 0.7)
    codes, lo, hi = encode(counts, asinh=True)
    assert codes.dtype == np.uint8 and lo.shape == hi.shape == (64,)
    back = decode(codes, lo, hi, asinh=True)
    y, yq = np.arcsinh(counts), np.arcsinh(back)
    worst = min(np.corrcoef(y[:, t], yq[:, t])[0, 1] for t in range(64))
    assert worst > 0.999
    assert np.all(back[counts == 0] == 0)  # zero coverage stays exactly zero (lo == 0)


def test_constant_track_and_contact_maps_round_trip():
    flat = np.full((100, 3), 2.5)
    codes, lo, hi = encode(flat, asinh=False)
    assert np.allclose(decode(codes, lo, hi, asinh=False), flat)
    contact = np.random.default_rng(1).normal(0, 1, size=(5000, 28))
    codes, lo, hi = encode(contact, asinh=False)
    step = (hi - lo) / 255
    assert np.all(np.abs(decode(codes, lo, hi, asinh=False) - contact) <= step / 2 + 1e-6)


def test_training_reader_decodes_like_the_encoder():
    import pytest
    pytest.importorskip("torch")
    from reference.alphagenome_50m.dataset import AlphaGenomeDataset
    counts = np.random.default_rng(2).gamma(0.5, 10.0, size=(1536, 8))
    codes, lo, hi = encode(counts, asinh=True)
    group = zarr.group()
    group.create_dataset("atac/values", data=codes)
    group.create_dataset("atac/values_lo", data=lo)
    group.create_dataset("atac/values_hi", data=hi)
    decoded = AlphaGenomeDataset._decode(group, "atac/values", asinh=True)
    assert np.allclose(decoded, decode(codes, lo, hi, asinh=True), rtol=1e-5, atol=1e-6)
    group.create_dataset("cage/values", data=counts.astype(np.float32))
    assert np.array_equal(AlphaGenomeDataset._decode(group, "cage/values", asinh=True),
                          counts.astype(np.float32))
