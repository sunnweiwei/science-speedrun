import numpy as np
import pytest
from scipy.sparse import csr_matrix
from spherical_pool import empirical_scores, apply_pool, grid_geometry


def test_crps_matches_pairwise_definition():
    rng=np.random.default_rng(15)
    p=rng.normal(size=(56,13)); t=rng.normal(size=13); w=rng.uniform(size=13)
    direct=np.abs(p-t).mean(axis=0)-0.5*np.abs(p[:,None,:]-p[None,:,:]).mean(axis=(0,1))
    assert np.isclose(empirical_scores(p,t,w)['crps'], np.average(direct,weights=w))


def test_pool_masks_truth_identically_and_max_is_not_average():
    p=np.array([[1.,3.,100.,8.],[2.,4.,200.,9.]])
    t=np.array([2.,4.,np.nan,7.])
    matrix=csr_matrix([[1.,2.,3.,0.],[0.,2.,3.,4.]])
    result=apply_pool(p,t,matrix)
    np.testing.assert_allclose(result['average'][0], [[7/3,38/6],[10/3,44/6]])
    np.testing.assert_allclose(result['maximum'][0],[[3,8],[4,9]])
    np.testing.assert_allclose(result['maximum'][1],[4,7])


def test_spherical_grid_area_and_longitude_wrap():
    points,weights=grid_geometry(np.linspace(-90,90,721),np.arange(1440)/4)
    np.testing.assert_allclose(weights.sum(),4*np.pi)
    np.testing.assert_allclose(np.linalg.norm(points,axis=1),1,atol=1e-12)
    # Quarter-degree neighbors straddle the zero/360 meridian normally.
    assert np.linalg.norm(points[360*1440]-points[361*1440-1]) < 0.005


def test_paired_comparison_rejects_changed_truth():
    import copy
    import pytest
    from compare import compare, MATCH
    row=dict(case='2022-01-05T00:00:00',variable='2m_temperature',level_hpa=None,lead_hours=12,crps=2.,mse=4.)
    baseline=dict(status='passed',lineage={'runner_sha256':'runner','context_sha256':'case1'},per_case=[row],pooled_per_case=[],**{k:'same' for k in MATCH})
    candidate=copy.deepcopy(baseline)
    candidate['per_case'][0]['crps']=1.
    result=compare([baseline],[candidate],[row['case']])
    assert result[0]['crps_skill_vs_baseline']==0.5
    candidate['truth_values_sha256']='different'
    with pytest.raises(ValueError,match='truth_values_sha256'):
        compare([baseline],[candidate],[row['case']])


def test_comparison_rejects_model_changes_across_dates():
    import copy
    import pytest
    from compare import compare, MATCH
    dates=['2022-01-05T00:00:00','2022-03-01T06:00:00']
    reports=[]
    for date in dates:
        row=dict(case=date,variable='2m_temperature',level_hpa=None,lead_hours=12,crps=2.,mse=4.)
        reports.append(dict(status='passed',lineage={'package_sha256':'model','context_sha256':date},
            per_case=[row],pooled_per_case=[],**{k:'same' for k in MATCH}))
    candidate=copy.deepcopy(reports)
    assert compare(reports,candidate,dates)[0]['cases']==2
    candidate[1]['lineage']['package_sha256']='another-model'
    with pytest.raises(ValueError,match='lineage changed'):
        compare(reports,candidate,dates)


def test_precipitation_preserves_members_and_has_no_initial_24h_value():
    import xarray as xr
    import pytest
    from precipitation import accumulate_24h
    rain=xr.DataArray([[1.,4.,2.],[3.,0.,5.]],dims=['member','lead'],
        coords={'member':[7,13],'lead':np.array([12,24,36],dtype='timedelta64[h]')})
    accumulated=accumulate_24h(rain)
    np.testing.assert_array_equal(accumulated.member,[7,13])
    np.testing.assert_array_equal(accumulated.lead,np.array([24,36],dtype='timedelta64[h]'))
    np.testing.assert_allclose(accumulated,[[5.,6.],[3.,5.]])
    with pytest.raises(ValueError,match='consecutive'):
        accumulate_24h(rain.isel(lead=[0,2]))


def test_potential_rev_perfect_and_constant_forecasts():
    from rev import confusion,potential_value
    truth=np.array([0.,2.,0.,2.]);threshold=np.ones(4);weights=np.array([1.,2.,3.,4.])
    perfect=np.tile(truth,(56,1))
    matrix=confusion(perfect,truth,threshold,weights)
    assert matrix.shape==(58,4)
    np.testing.assert_allclose(potential_value(matrix,[.01,.1,.5]),1.)
    constant=np.zeros_like(perfect)
    np.testing.assert_allclose(potential_value(confusion(constant,truth,threshold,weights),[.01,.1,.5]),0.,atol=1e-14)
    assert np.isnan(potential_value(confusion(constant,np.zeros(4),threshold,weights),[.1])).all()


@pytest.mark.parametrize('runner',['predict_gencast.py','predict_wn2.py'])
def test_member_queue_two_processes_claim_each_member_once(tmp_path,runner):
    import ast
    import fcntl
    import multiprocessing
    from pathlib import Path
    import time
    # Exercise the actual queue function without importing GPU runtime modules.
    tree=ast.parse((Path(__file__).parent/runner).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='claimed_members')
    namespace={'fcntl':fcntl}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'queue-under-test','exec'),namespace)
    context=multiprocessing.get_context('fork')
    barrier=context.Barrier(2)
    def run(worker):
        barrier.wait()
        for member in namespace['claimed_members'](tmp_path,0,56):
            time.sleep(.002)
            # Exclusive creation exposes duplicate claims immediately.
            with (tmp_path/f'case-000-member-{member:03d}.complete.json').open('x') as f:
                f.write(str(worker))
    processes=[context.Process(target=run,args=(i,)) for i in range(2)]
    for process in processes:process.start()
    for process in processes:
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode==0
    assert len(list(tmp_path.glob('*.complete.json')))==56
    assert list(namespace['claimed_members'](tmp_path,0,56))==[]


def test_pooling_only_comparison_preserves_paired_dates():
    import copy
    from compare import compare, MATCH
    date='2022-01-05T00:00:00'
    row=dict(case=date,variable='2m_temperature',level_hpa=None,lead_hours=12,
             pooling='maximum',mesh_refinement=2,crps=2.,mse=4.)
    baseline=dict(status='passed',lineage={'runner_sha256':'runner'},per_case=[],
                  pooled_per_case=[row],**{k:'same' for k in MATCH})
    candidate=copy.deepcopy(baseline)
    candidate['pooled_per_case'][0]['crps']=1.
    assert compare([baseline],[candidate],[date])[0]['crps_skill_vs_baseline']==.5
    candidate['pooled_per_case'][0]['case']='2022-03-01T06:00:00'
    with pytest.raises(ValueError,match='Mismatched paired panels'):
        compare([baseline],[candidate],[date])
