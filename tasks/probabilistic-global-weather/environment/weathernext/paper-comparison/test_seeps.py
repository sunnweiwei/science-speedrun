import numpy as np
from seeps import score


def test_all_nine_seeps_categories():
    rain=np.array([0.,.001,.004]);expected=np.array([[0.,1.,4.],[1.,0.,3.],[1.6,.6,0.]])
    for i,p in enumerate(rain):
        for j,t in enumerate(rain):
            np.testing.assert_allclose(score(p,t,.003,.5,1.,.25)['seeps'],expected[i,j])


def test_seeps_mask_and_empty_support():
    result=score([0.,0.,0.],[.001,.004,.004],[.003,.003,np.nan],[.5,.99,1.],[1.,10.,10.],.25)
    assert result['seeps']==1. and result['valid_grid_points']==1
    np.testing.assert_allclose(result['valid_area_fraction'],1/21)
    assert score(0.,0.,np.nan,1.,1.,.25)['seeps'] is None


def test_seeps_matches_pinned_weatherbenchx():
    import xarray as xr
    from weatherbenchX.metrics.categorical import SEEPS
    rng=np.random.default_rng(91);shape=(3,4)
    p=rng.uniform(0,.006,shape);t=rng.uniform(0,.006,shape)
    p[0,0]=.000251;t[0,1]=.003  # Matrix oracle away from the differing dry boundary.
    q=rng.uniform(.05,.95,shape);wet=np.full(shape,.003)
    coords=dict(lat=[-60.,0.,60.],lon=[0.,90.,180.,270.],init_time=np.datetime64('2022-01-01'),lead_time=np.timedelta64(12,'h'))
    pred=xr.DataArray(p,dims=['lat','lon'],coords=coords);truth=xr.DataArray(t,dims=['lat','lon'],coords=coords)
    clim=xr.Dataset(dict(rain_seeps_threshold=(('dayofyear','hour','lat','lon'),wet[None,None]),
                         rain_seeps_dry_fraction=(('dayofyear','hour','lat','lon'),q[None,None])),
                    coords=dict(dayofyear=[1],hour=[12],lat=coords['lat'],lon=coords['lon']))
    weights=np.broadcast_to(np.cos(np.deg2rad(coords['lat']))[:,None],shape)
    for dry in [.1,.25]:
        native=SEEPS(['rain'],clim,dry_threshold_mm=dry).compute({'rain':pred},{'rain':truth})['rain'].values
        valid=np.isfinite(native)
        expected=np.sum(np.where(valid,native,0.)*weights)/np.sum(np.where(valid,weights,0.))
        actual=score(p,t,wet,q,weights,dry)
        np.testing.assert_allclose(actual['seeps'],expected,rtol=1e-13)


def test_paper_strict_dry_boundary():
    assert score(.00025,0.,.003,.5,1.,.25)['seeps']==1.
    assert score(.000249,0.,.003,.5,1.,.25)['seeps']==0.


def test_accumulation_context_and_monthly_thresholds():
    import warnings
    from prepare_seeps import accumulate,thresholds
    raw=np.arange(10.)[:,None]
    np.testing.assert_array_equal(accumulate(raw,12)[:,0],np.arange(5,18,2))
    np.testing.assert_array_equal(accumulate(raw,24)[:,0],np.arange(6,31,4))
    rng=np.random.default_rng(40)
    samples=rng.uniform(0,.004,(127,3,4));samples[:,0,0]=0
    for dry in [.1,.25]:
        p1,wet=thresholds(samples,dry)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='All-NaN slice encountered')
            expected=np.nanquantile(np.where(samples<dry/1000,np.nan,samples),2/3,axis=0,method='linear')
        np.testing.assert_allclose(wet,expected,rtol=1e-13,equal_nan=True)
        np.testing.assert_array_equal(p1,(samples<dry/1000).mean(axis=0))


def test_seeps_paired_aggregation_checks_support():
    import copy
    import pytest
    from compare import compare_seeps
    def report(date,value):
        return dict(seeps_per_case=[dict(case=date,accumulation_hours=hours,lead_hours=lead,
            dry_threshold_mm=dry,seeps=value,valid_area_fraction=.8)
            for hours in (12,24) for lead in range(hours,361,12) for dry in (.1,.25)])
    baseline=[report('d0',.5),report('d1',.9)]
    candidate=[report('d0',.4),report('d1',.8)]
    result=compare_seeps(baseline,candidate,['d0','d1'])
    assert len(result)==118
    np.testing.assert_allclose(result[0]['baseline_seeps'],.7)
    np.testing.assert_allclose(result[0]['candidate_one_minus_seeps'],.4)
    candidate=copy.deepcopy(candidate);candidate[0]['seeps_per_case'][0]['valid_area_fraction']=.7
    with pytest.raises(ValueError,match='support'):compare_seeps(baseline,candidate,['d0','d1'])
