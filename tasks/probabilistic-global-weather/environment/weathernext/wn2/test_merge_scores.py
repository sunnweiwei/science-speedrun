import numpy as np
import pytest
import xarray as xr
from merge_scores import merge,reward


def fixture():
    protocol=dict(status='test',evaluation_initializations=['2000-01-01'],variables={'t':dict(pressure_levels=False)},
                  pressure_levels_hpa=[],lead_hours=[6],pool_cells=[1])
    row=dict(case='2000-01-01',variable='t',level_hpa=None,lead_hours=6,pool_cells=1,crps=2.,mse=9.,ensemble_variance=4.)
    report=dict(status='passed',case_index=0,protocol_sha256='p',evaluator_sha256='e',shard_scorer_sha256='s',
                weatherbenchx_revision='w',estimator='crps',scored_input_values_sha256='i',truth_values_sha256='t',per_case=[row])
    return protocol,report


def test_merge_rejects_missing_and_duplicate_case_or_panel():
    p,r=fixture()
    for reports in ([],[r,r],[r|dict(per_case=[])],[r|dict(per_case=r['per_case']*2)]):
        with pytest.raises(ValueError):merge(reports,p,'p')
    assert merge([r],p,'p')['aggregate'][0]['rmse']==3


def test_reward_quality_and_units():
    p,r=fixture();report=merge([r],p,'p')
    result=reward(report,xr.Dataset(dict(t=4.)))
    assert result['reward']==pytest.approx(2/3)
    report['aggregate'][0]['crps']*=1000
    assert reward(report,xr.Dataset(dict(t=4000.)))['reward']==result['reward']
    assert reward(report,xr.Dataset(dict(t=4.)))['reward']<result['reward']
    with pytest.raises(ValueError):reward(report,xr.Dataset(dict(t=np.nan)))
