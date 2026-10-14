"""Analytic metric oracles and adversarial protocol checks, without weather data."""
import json

import numpy as np
import pytest
import xarray as xr

from evaluate import cell_weights, evaluate, panel, summarize, validate_contract, verify_metric_source


def fixture():
    protocol = dict(status="test", variables={"temperature": {"units": "K", "pressure_levels": True}},
                    evaluation_initializations=["2022-01-05T00:00:00"], lead_hours=[6, 12],
                    members=2, grid=[3, 4], pressure_levels_hpa=[500], pool_cells=[1, 2], truth_source="HRES-fc0")
    coords = dict(case=np.array(protocol["evaluation_initializations"], dtype="datetime64[ns]"),
                  lead=np.array([6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]"),
                  lat=[-90., 0., 90.], lon=[0., 90., 180., 270.], level=[500])
    dims = ("case", "lead", "level", "lat", "lon")
    t = xr.Dataset({"temperature": xr.DataArray(np.ones((1, 2, 1, 3, 4)), dims=dims, coords=coords, attrs={"units": "K"})},
                   attrs={"truth_source": "HRES-fc0"})
    t = t.assign_coords(valid_time=(("case", "lead"), coords["case"][:, None] + coords["lead"][None, :]))
    p = xr.concat([t * 0, t * 2], dim=xr.IndexVariable("member", [0, 1]))
    p.temperature.attrs["units"] = "K"
    return p, t, protocol


def test_upstream_revision():
    verify_metric_source()


def test_empirical_crps_is_not_fair_crps():
    p, t, _ = fixture()
    actual = panel(p.temperature.isel(case=0, lead=0, level=0), t.temperature.isel(case=0, lead=0, level=0))
    # For ensemble {0,2}, truth 1: empirical CRPS=1 - .5*1=.5;
    # fair CRPS would be zero. This guards the scientific estimator choice.
    assert actual == pytest.approx(dict(crps=0.5, mse=0., ensemble_variance=2., valid_area_fraction=1.))


def test_metric_matches_pairwise_oracle_and_area_integral():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(8, 5, 8))
    truth = rng.normal(size=(5, 8))
    p = xr.DataArray(values, dims=("member", "lat", "lon"), coords={"lat": np.linspace(-90, 90, 5), "lon": np.arange(8)*45})
    t = p.isel(member=0, drop=True).copy(data=truth)
    w = np.asarray(cell_weights(t.lat))[:, None] * np.ones((1, 8))
    oracle = np.abs(values-truth).mean(0) - .5*np.abs(values[:,None]-values[None,:]).mean((0,1))
    assert panel(p, t)["crps"] == pytest.approx(np.sum(oracle*w)/w.sum())
    # Constant fields retain their scores after partial polar-block pooling.
    assert panel(p*0, t*0+3, 2)["crps"] == pytest.approx(3.)


def test_global_rmse_reduces_mse_before_square_root():
    rows = [dict(variable="t", level_hpa=500, lead_hours=6, pool_cells=1,
                 crps=1, mse=mse, ensemble_variance=2) for mse in [0, 8]]
    assert summarize(rows)[0]["rmse"] == 2


@pytest.mark.parametrize("fault", ["member", "lead", "case", "lat", "units", "variable", "valid_time", "truth_source", "dimensions"])
def test_contract_rejects_bad_outputs(fault):
    p, t, protocol = fixture()
    if fault in ("member", "lead", "case", "lat"):
        p = p.isel({fault: slice(0, 0)})
    elif fault == "units": p.temperature.attrs["units"] = "degC"
    elif fault == "variable": p = p.rename(temperature="wrong")
    elif fault == "valid_time": p = p.assign_coords(valid_time=p.valid_time + np.timedelta64(6, "h"))
    elif fault == "truth_source": t.attrs["truth_source"] = "ERA5"
    elif fault == "dimensions": p["temperature"] = p.temperature.expand_dims(extra=[1])
    with pytest.raises(ValueError): validate_contract(p, t, protocol)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_bad_prediction_is_not_skipped(bad):
    p, t, protocol = fixture()
    p.temperature.values.flat[0] = bad
    with pytest.raises(ValueError, match="Nonfinite"): list(evaluate(p, t, protocol))


def test_complete_evaluation_and_streamed_serialization(tmp_path):
    p, t, protocol = fixture()
    p.to_netcdf(tmp_path / "prediction.nc")
    t.to_netcdf(tmp_path / "truth.nc")
    with xr.open_dataset(tmp_path / "prediction.nc") as pp, xr.open_dataset(tmp_path / "truth.nc") as tt:
        rows = list(evaluate(pp, tt, protocol))
    assert len(rows) == 4
    assert all(r["crps"] == pytest.approx(.5) for r in rows)
    assert all(r["spread_skill"] is None for r in summarize(rows))
    json.dumps(summarize(rows), allow_nan=False)


def test_fixed_truth_mask_and_pooled_mask_are_shared():
    p,t,_=fixture()
    p=p.temperature.isel(case=0,lead=0,level=0)
    t=t.temperature.isel(case=0,lead=0,level=0).copy(deep=True)
    t.values[0,0]=np.nan
    with pytest.raises(ValueError,match='truth area'):panel(p,t)
    for factor in (1,2):
        score=panel(p,t,factor,minimum_valid_area_fraction=.9)
        assert score['crps']==pytest.approx(.5)
        assert .9<score['valid_area_fraction']<1
    p=p.copy(deep=True);p.values[0,0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):panel(p,t,minimum_valid_area_fraction=.9)
