import copy
import numpy as np
import pytest
import xarray as xr
from simple_baselines import forecast
from spherical_pool import empirical_scores


def fixture():
    p=dict(variables={'2m_temperature':dict(pressure_levels=False,units='K')},lead_hours=[12,24],members=56)
    rain=dict(variables={'total_precipitation_12hr':dict(pressure_levels=False,units='m')})
    dates=np.array(['2022-01-04T18','2022-01-05T00'],dtype='datetime64[ns]')
    ds=xr.Dataset({'2m_temperature':(('time','lat','lon'),np.array([np.ones((3,4))*280,np.ones((3,4))*282]))},
        coords=dict(time=np.array([-6,0],dtype='timedelta64[h]'),lat=[-90,0,90],lon=[0,90,180,270],datetime=('time',dates)))
    return ds,p,rain


def test_initial_state_only_and_dry_accumulation():
    from precipitation import accumulate_24h
    ds,p,r=fixture();result=forecast(ds,p,r,'2022-01-05T00')
    assert result['2m_temperature'].chunks is not None
    np.testing.assert_array_equal(result['2m_temperature'].values,282)
    np.testing.assert_array_equal(accumulate_24h(result.total_precipitation_12hr).values,0)
    assert result.sizes['member']==56
    np.testing.assert_array_equal(result.valid_time.values, np.array([['2022-01-05T12','2022-01-06T00']],dtype='datetime64[ns]'))


def test_future_context_and_wrong_initialization_rejected():
    ds,p,r=fixture()
    with pytest.raises(ValueError,match='future'):
        forecast(ds.assign_coords(time=np.array([0,6],dtype='timedelta64[h]')),p,r,'2022-01-05T00')
    with pytest.raises(ValueError,match='initialization'):
        forecast(ds,p,r,'2022-01-06T00')


def test_point_mass_compression_preserves_scores_and_zero_spread():
    field=np.array([1.,4.,-2.]);truth=np.array([0.,2.,3.]);weights=np.array([1.,2.,1.])
    a=empirical_scores(np.tile(field,(2,1)),truth,weights)
    b=empirical_scores(np.tile(field,(56,1)),truth,weights)
    assert a==b
    assert a['crps']==2.5 and a['mse']==8.5 and a['ensemble_variance_ddof0']==0


def test_archived_model_comparison_still_rejects_changed_truth():
    from compare import compare,MATCH
    row=dict(case='2022-01-05T00:00:00',variable='2m_temperature',level_hpa=None,lead_hours=12,crps=2.,mse=4.)
    b=dict(status='passed',lineage={},per_case=[row],pooled_per_case=[],**{k:'same' for k in MATCH})
    c=copy.deepcopy(b);b.update(baseline='persistence-dry',baseline_reference_scorer_sha256='same',evaluator_sha256='adapter')
    c['per_case'][0].update(crps=1.,mse=1.)
    assert compare([b],[c],[row['case']])[0]['rmse_skill_vs_baseline']==.5
    c['truth_values_sha256']='changed'
    with pytest.raises(ValueError,match='truth_values'):compare([b],[c],[row['case']])
