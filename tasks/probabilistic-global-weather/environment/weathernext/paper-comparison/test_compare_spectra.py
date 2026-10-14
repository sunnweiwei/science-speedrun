import copy
import numpy as np
import pytest
from compare_spectra import compare,MATCH


def test_spectra_average_powers_and_reject_changed_truth():
    base=dict(**{k:'same' for k in MATCH},case='d0',status='passed',lineage={'seed':0},
        truth_receipt_sha256='truth0',truth_values_sha256='values0',rows=[dict(lead_hours=12,variable='t',level_hpa=None,
        member_mean_power=[1.,4.],ensemble_mean_field_power=[0.,1.],truth_power=[2.,3.])])
    base['total_wavenumber']=[0,1];base['transform']={'normalization':'test'}
    other=copy.deepcopy(base);other.update(case='d1',truth_receipt_sha256='truth1',truth_values_sha256='values1')
    other['rows'][0]['member_mean_power']=[9.,16.]
    candidates=copy.deepcopy([base,other]);panels={(12,'t',None)}
    result=compare([base,other],candidates,['d0','d1'],panels)
    np.testing.assert_array_equal(result['rows'][0]['baseline_member_mean_power'],[5.,10.])
    np.testing.assert_array_equal(result['rows'][0]['baseline_ensemble_mean_field_power'],[0.,1.])
    candidates[0]['truth_values_sha256']='changed'
    with pytest.raises(ValueError,match='Incompatible'):compare([base,other],candidates,['d0','d1'],panels)


def test_missing_spectral_truth_excludes_same_case_from_both_models_only():
    report=dict(**{k:'same' for k in MATCH},case='d0',status='passed',lineage={'seed':0},
        truth_receipt_sha256='truth',truth_values_sha256='values',rows=[dict(lead_hours=12,variable='t',level_hpa=None,
        member_mean_power=[100.,400.],ensemble_mean_field_power=[0.,1.],truth_power=None,
        truth_status='missing_truth',missing_truth_grid_points=4)])
    report['total_wavenumber']=[0,1];report['transform']={'normalization':'test'}
    other=copy.deepcopy(report);other['case']='d1';other['rows'][0].update(truth_power=[2.,3.],truth_status='passed',missing_truth_grid_points=0,member_mean_power=[1.,4.])
    result=compare([report,other],copy.deepcopy([report,other]),['d0','d1'],{(12,'t',None)})
    row=result['rows'][0]
    assert row['cases']==1 and row['requested_cases']==2
    assert row['excluded_truth_cases']==[dict(case='d0',missing_grid_points=4)]
    np.testing.assert_array_equal(row['baseline_member_mean_power'],[1.,4.])
