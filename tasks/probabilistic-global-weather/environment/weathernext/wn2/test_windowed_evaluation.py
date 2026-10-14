"""Continuation, budget, and all-endpoint window lifecycle regression tests."""
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import xarray as xr
from disk_budget import plan, DiskBudget
from inference_state import load_state, save_state, state_identity, resume_lead
from window_scores import merge_windows
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'paper-comparison'))
from precipitation import accumulate_24h


def protocol():
    p=json.loads(Path(__file__).with_name('protocol.json').read_text())
    p.update(grid=[2,2],members=2,evaluation_initializations=p['evaluation_initializations'][:1],pressure_levels_hpa=[300,500])
    return p


@pytest.mark.parametrize('backend',['numpy','jax'])
def test_checkpoint_is_lossless_and_bound_to_member_seed_and_package(tmp_path,backend):
    state=xr.Dataset({'weather':(('time','lat','lon'),np.arange(8,dtype='float32').reshape(2,2,2)),
                      'forcing':('time',np.array([0.123456789,0.234567891],dtype='float64'))},
                     coords={'time':np.array([-6,0],dtype='timedelta64[h]'),'lat':[-90,90],'lon':[0,180]})
    if backend=='jax':
        import jax.numpy as jnp
        state['weather'].data=jnp.asarray(state.weather.values)
    identity=state_identity('2022-01-01',123,7,24,'package')
    save_state(tmp_path/'state',state,identity)
    actual=load_state(tmp_path/'state',identity)
    xr.testing.assert_identical(actual,state)
    for name,value in [('seed',124),('member',8),('lead_hours',36),('package_sha256','other')]:
        with pytest.raises(ValueError,match='identity'):
            load_state(tmp_path/'state',dict(identity,**{name:value}))
    chunk=tmp_path/'state/context.zarr/weather/0.0.0'
    chunk.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):
        load_state(tmp_path/'state',identity)


@pytest.mark.parametrize('start,leads', [(12,[12,24]),(24,[48]),(0,[24]),(6,[12]),(12,[24,48])])
def test_continuation_cannot_skip_or_recompute_steps(start,leads):
    with pytest.raises(ValueError):
        resume_lead(dict(schema='wn2-inference/v3',cases=[{}],lead_hours=leads,resume=dict(lead_hours=start)))


@pytest.mark.parametrize('lazy',[False,True])
def test_precipitation_boundary_exact_without_forecast_recomputation(lazy):
    rain=xr.DataArray(np.arange(60,dtype='float32').reshape(2,30),dims=['member','lead'],
                      coords={'member':[0,1],'lead':np.arange(12,361,12).astype('timedelta64[h]')})
    if lazy:
        pytest.importorskip('dask.array')
        rain=rain.chunk({'lead':1})
    expected=accumulate_24h(rain)
    for size in (1,2,7,15):
        pieces=[]
        for start in range(0,30,size):
            current=rain.isel(lead=slice(max(0,start-1),min(start+size,30)))
            result=accumulate_24h(current,allow_partial=True)
            owned=rain.lead.values[start:start+size]
            pieces.append(result.sel(lead=[x for x in owned if x in result.lead]))
        xr.testing.assert_identical(xr.concat(pieces,dim='lead'),expected)


def test_planner_uses_allowance_without_compression_assumption():
    p=json.loads(Path(__file__).with_name('protocol.json').read_text())
    small=plan(p,250,58838549659,780970016)
    large=plan(p,500,58838549659,780970016)
    assert 1<=small['lead_window']<large['lead_window']<30
    for result in (small,large):
        assert [h for w in result['windows'] for h in w]==p['lead_hours']
        assert result['fixed_bytes']+result['bytes_per_lead']*result['lead_window']<=result['disk_allowance_bytes']
        assert result['forecast_recomputation'] is False
    with pytest.raises(ValueError,match='too small'):plan(p,100,58838549659,780970016)
    for value in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):plan(p,value,0,0)


def test_disk_budget_counts_fixed_inputs_and_stops_on_excess(tmp_path):
    budget=DiskBudget(tmp_path,dict(resource_bytes=10,package_bytes=10,reserve_bytes=10,disk_allowance_bytes=50))
    (tmp_path/'data').write_bytes(b'x'*15)
    assert budget.check()==45
    (tmp_path/'data').write_bytes(b'x'*25)
    with pytest.raises(RuntimeError,match='Disk allowance exceeded'):budget.check()
    assert budget.peak==55


def test_window_merge_rejects_missing_overlap_and_identity_changes(tmp_path):
    reports=[]
    for lead in (12,24):
        path=tmp_path/f'{lead}.json'
        path.write_text(json.dumps(dict(status='passed',lead_hours=[lead],protocol_sha256='p',members=56,
            lineage={'seed':1},per_case=[dict(lead_hours=lead,crps=2)])))
        reports.append(path)
    merged=merge_windows(reports,[12,24])
    assert [r['lead_hours'] for r in merged['per_case']]==[12,24]
    for paths in (reports[:1],reports[::-1],[reports[0],reports[0]]):
        with pytest.raises(ValueError,match='windows'):merge_windows(paths,[12,24])
    r=json.loads(reports[1].read_text());r['lineage']['seed']=2;reports[1].write_text(json.dumps(r))
    with pytest.raises(ValueError,match='identity'):merge_windows(reports,[12,24])


def threaded(jobs):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in [pool.submit(job) for job in jobs]:future.result()


@pytest.mark.parametrize('window_width,score_chunk',[(1,None),(7,None),(30,None),(7,2),(30,1)])
def test_full_window_lifecycle_all_30_leads_all_endpoints(tmp_path,window_width,score_chunk):
    """Real artifact capture/export/merge; external GPU/scorer binaries substituted."""
    from windowed_evaluation import evaluate_case
    from training_data import sha256
    p=protocol();protocol_path=tmp_path/'protocol.json';protocol_path.write_text(json.dumps(p));digest=sha256(protocol_path)
    package=tmp_path/'package';package.mkdir();(package/'package.json').write_text('{}')
    root=tmp_path/'benchmark';history=root/'tracker/truth';history.mkdir(parents=True)
    date=np.datetime64(p['evaluation_initializations'][0],'ns')
    coords=dict(time=date+np.arange(-240,361,12).astype('timedelta64[h]'),lat=[-90,90],lon=[0,180])
    truth=xr.Dataset({n:(('time','lat','lon'),np.zeros((51,2,2),dtype='float32')) for n in ('msl','thickness','wind')},coords=coords)
    truth['zs']=(('lat','lon'),np.zeros((2,2),dtype='float32'));truth.to_netcdf(history/'case-000.nc')
    (history/'case-000.json').write_text(json.dumps(dict(status='passed',protocol_sha256=digest,
        sha256=sha256(history/'case-000.nc'),initialization=str(date))))
    output=tmp_path/'out';(output/'tracker-truth').mkdir(parents=True)
    scales=xr.Dataset({n:('level',[1.,1.]) if spec['pressure_levels'] else 1. for n,spec in p['variables'].items()},coords={'level':[300,500]})
    storage=plan(p,.1,0,2,0);storage['windows']=[p['lead_hours'][i:i+window_width] for i in range(0,30,window_width)]
    budget=DiskBudget(output,storage);seen=[];scored=[]
    def execute(command,label,env):
        def arg(name):return command[command.index(name)+1]
        script=Path(command[1]).name
        if script=='verify_case.py':
            lead_idx=command.index('--lead-hours')+1
            leads=[]
            for x in command[lead_idx:]:
                if x.startswith('--'):break
                leads.append(int(x))
            if '--state' in command:
                state=Path(arg('--state'));assert state.is_dir()
                import shutil
            seen.extend(leads)
            replay=Path(arg('--output'));(replay/'public').mkdir(parents=True)
            if '--state' in command:shutil.move(str(state),str(replay/'public/resume'))
            (replay/'relocated-model').mkdir();(replay/'untrusted-output').mkdir()
            (replay/'replay.json').write_text('{}')
            pred=Path(arg('--predictions'));pred.mkdir()
            for member in range(p['members']):
                variables={}
                for n,spec in dict(p['variables'],total_precipitation_12hr={'pressure_levels':False}).items():
                    dims=['case','lead','member']+(['level'] if spec['pressure_levels'] else [])+['lat','lon']
                    shape=[1,len(leads),1]+([2] if spec['pressure_levels'] else [])+[2,2]
                    variables[n]=(dims,np.ones(shape,dtype='float32')*(member+1))
                ds=xr.Dataset(variables,coords=dict(case=[date],lead=np.array(leads,dtype='timedelta64[h]'),member=[member],level=[300,500],lat=[-90,90],lon=[0,180]))
                stem=f'case-000-member-{member:03d}'
                ds.to_zarr(pred/f'{stem}.zarr',consolidated=True)
                record=dict(status='passed',protocol_sha256=digest,package_sha256='p',runner_sha256='r',seed=1,member=member,initialization=str(date),context_sha256='c',files={},lead_hours=leads)
                (pred/f'{stem}.complete.json').write_text(json.dumps(record))
                if leads[-1]<360:(replay/f'untrusted-output/state/member-{member:03d}').mkdir(parents=True)
        elif script in ('score_marginals.py','score_spectra.py'):
            leads=[]
            flag='--score-leads' if '--score-leads' in command else '--lead-hours'
            for x in command[command.index(flag)+1:]:
                if x.startswith('--'):break
                leads.append(int(x))
            # The boundary state belongs to the window, not to a lead chunk inside it.
            if int(arg('--lead-hours'))>12 and script=='score_marginals.py':assert Path(arg('--previous-rain')).is_dir()
            report=dict(status='passed',protocol_sha256=digest,lead_hours=leads,members=p['members'])
            if script=='score_marginals.py':
                report['per_case']=[dict(case=p['evaluation_initializations'][0],variable=n,level_hpa=level,lead_hours=lead,crps=1.)
                    for lead in leads for n,spec in p['variables'].items() for level in ([300,500] if spec['pressure_levels'] else [None])]
            else:report['rows']=[dict(lead_hours=lead,power=1.) for lead in leads]
            Path(arg('--output')).write_text(json.dumps(report));scored.append((script,leads))
        elif script=='run_tracker.py':
            dest=Path(arg('--output'));dest.mkdir(parents=True)
            with xr.open_dataset(arg('--input')) as ds:assert ds.sizes['time']==51
            (dest/'complete.json').write_text(json.dumps(dict(status='passed',input_sha256=sha256(Path(arg('--input'))))))
        elif script=='score_cyclones.py':
            for member in range(p['members']):
                dest=Path(arg('--tracks'))/f'case-000-member-{member:03d}'
                assert (dest/'input.json').exists() and not (dest/'input.nc').exists()
            Path(arg('--output')).write_text(json.dumps(dict(status='passed',protocol_sha256=digest)))
        else:raise AssertionError(script)
    options={} if score_chunk is None else dict(run_jobs=threaded,score_chunk=score_chunk)
    evaluate_case(0,'0',package,root,output,protocol_path,p,storage,execute,'python',Path('/tracker'),scales,budget,**options)
    assert seen==p['lead_hours'] # No forecast lead was requested twice.
    chunks=sum(-(-len(w)//(score_chunk or len(w))) for w in storage['windows'])
    assert len(scored)==3*chunks
    assert sorted(h for script,leads in scored if script=='score_spectra.py' for h in leads)==p['lead_hours']
    scores=json.loads((output/'scores/case-000.json').read_text())
    assert [r['lead_hours'] for r in scores['per_case']]==sorted(r['lead_hours'] for r in scores['per_case'])
    assert not list(output.rglob('*.zarr'))
    for endpoint in ('scores','pooling-scores','spectra','cyclone-scores'):
        assert json.loads((output/endpoint/'case-000.json').read_text())['status']=='passed'
    assert json.loads((output/'replay-000/replay.json').read_text())['forecast_recomputation'] is False
    assert budget.peak<=storage['disk_allowance_bytes']


def main_fixture(tmp_path,monkeypatch):
    import verify_track
    p=json.loads(Path(verify_track.__file__).with_name('protocol.json').read_text())
    package=tmp_path/'package';package.mkdir()
    (package/'package.json').write_text(json.dumps({'inference_schemas':['wn2-inference/v3']}))
    root=tmp_path/'benchmark';(root/'tracker/truth').mkdir(parents=True)
    (root/'tracker/named-truth.json').write_text('{}')
    (root/'resources.json').write_text(json.dumps({'files':{}}))
    scales=xr.Dataset({n:('level',[1.]*len(p['pressure_levels_hpa'])) if s['pressure_levels'] else 1.
                       for n,s in p['variables'].items()},coords={'level':p['pressure_levels_hpa']})
    scales.to_netcdf(root/'training-scales.nc')
    output=tmp_path/'out'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    monkeypatch.setattr(sys,'argv',['verify_track','--package',str(package),'--benchmark',str(root),
        '--output',str(output),'--disk-allowance-gb','250'])
    return p,output


def test_default_entry_routes_all_eight_cases_through_disk_plan(tmp_path,monkeypatch):
    import verify_track,windowed_evaluation
    from training_data import sha256
    p,output=main_fixture(tmp_path,monkeypatch);seen=[]
    def case(index,device,package,root,output,protocol_path,p,storage,execute,model_python,tracker_bin,scales,budget,**options):
        seen.append(index)
        assert storage['forecast_recomputation'] is False
        for endpoint in verify_track.ENDPOINTS:
            path=output/endpoint;path.mkdir(exist_ok=True)
            r=dict(status='passed',protocol_sha256=sha256(protocol_path),members=p['members'])
            if endpoint=='scores':
                r['per_case']=[dict(case=p['evaluation_initializations'][index],variable=n,level_hpa=level,lead_hours=h,crps=1.)
                    for n,s in p['variables'].items() for level in (p['pressure_levels_hpa'] if s['pressure_levels'] else [None]) for h in p['lead_hours']]
            (path/f'case-{index:03d}.json').write_text(json.dumps(r))
        replay=output/f'replay-{index:03d}';replay.mkdir();(replay/'replay.json').write_text('{}')
        budget.check()
    monkeypatch.setattr(windowed_evaluation,'evaluate_case',case)
    verify_track.main()
    assert seen==list(range(8))
    assert json.loads((output/'reward.json').read_text())['reward']==.5
    status=json.loads((output/'status.json').read_text())
    assert status['storage_mode']=='continuation-windows' and status['cases']==8 and status['members']==56
    assert status['measured_accounted_peak_bytes']<=250e9


def test_real_subprocess_exceeding_allowance_fails_before_more_cases(tmp_path,monkeypatch):
    import verify_track,windowed_evaluation
    _,output=main_fixture(tmp_path,monkeypatch);seen=[]
    def case(index,device,package,root,output,protocol_path,p,storage,execute,model_python,tracker_bin,scales,budget,**options):
        seen.append(index)
        # A sparse file tests accounting without consuming 250 GB physically.
        code="import pathlib,time; p=pathlib.Path(__import__('sys').argv[1]); f=p.open('wb'); f.truncate(250000000000); f.close(); time.sleep(30)"
        execute([sys.executable,'-c',code,str(output/'oversized')],'oversized-output')
    monkeypatch.setattr(windowed_evaluation,'evaluate_case',case)
    with pytest.raises(RuntimeError,match='Disk allowance exceeded'):verify_track.main()
    assert seen==[0]
    assert json.loads((output/'status.json').read_text())['status']=='failed'
    assert json.loads((output/'reward.json').read_text())['reward']==0


def test_existing_starter_upgrade_preserves_weights_and_original(tmp_path):
    import shutil
    from upgrade_inference import upgrade
    from training_data import sha256
    source=tmp_path/'source';source.mkdir();here=Path(__file__).parent
    for name in ('native_model.py','forecast_store.py'):
        shutil.copyfile(here/name,source/name)
    (source/'weights.npz').write_bytes(b'existing trained parameters')
    (source/'predict.py').write_text('old entry point')
    (source/'package.json').write_text(json.dumps(dict(schema='wn2-starter-package/v1',
        files={f.name:sha256(f) for f in source.iterdir() if f.is_file()})))
    original=sha256(source/'package.json');weight=sha256(source/'weights.npz')
    result=upgrade(source,tmp_path/'upgraded')
    assert result==dict(source_package_sha256=original,weights_sha256=weight,training_rerun=False)
    assert sha256(source/'package.json')==original and sha256(tmp_path/'upgraded/weights.npz')==weight
    assert (source/'predict.py').read_text()=='old entry point'
    assert (tmp_path/'upgraded/inference_state.py').exists()
    with pytest.raises(ValueError,match='new output'):upgrade(source,source/'nested')


def test_rain_boundary_accepts_current_valid_time_and_compact_previous_state():
    from precipitation import prepend_interval
    rain=xr.DataArray([2.,3.],dims='lead',coords={'lead':np.array([12,24],dtype='timedelta64[h]'),
        'valid_time':('lead',np.array(['2022-01-01T12','2022-01-02T00'],dtype='datetime64[h]'))}).chunk({'lead':1})
    previous=rain.isel(lead=[0]).drop_vars('valid_time')
    joined=prepend_interval(rain.isel(lead=[1]),previous)
    actual=accumulate_24h(joined,allow_partial=True).compute()
    assert actual.item()==5.
    with pytest.raises(ValueError,match='adjacent'):
        prepend_interval(rain.isel(lead=[1]),previous.assign_coords(lead=np.array([0],dtype='timedelta64[h]')))


def test_truth_identity_is_independent_of_disk_window_size(tmp_path):
    from window_scores import canonical_truth_sha256
    leads=[12,24,36,48];all_truth={str(h):f'digest-{h}' for h in leads}
    for width in (1,2,4):
        paths=[]
        for start in range(0,len(leads),width):
            subset=leads[start:start+width];truth={str(h):all_truth[str(h)] for h in subset}
            report=dict(status='passed',lead_hours=subset,truth_lead_sha256=truth,truth_values_sha256=canonical_truth_sha256(truth),per_case=[])
            path=tmp_path/f'{width}-{start}.json';path.write_text(json.dumps(report));paths.append(path)
        merged=merge_windows(paths,leads)
        assert merged['truth_values_sha256']==canonical_truth_sha256(all_truth)
    spectral=[]
    for h in leads:
        path=tmp_path/f'spectra-{h}.json'
        path.write_text(json.dumps(dict(status='passed',lead_hours=[h],truth_values_sha256='full-truth',rows=[])))
        spectral.append(path)
    assert merge_windows(spectral,leads)['truth_values_sha256']=='full-truth'


def test_concurrent_plan_shares_resources_and_splits_the_rest():
    p=json.loads(Path(__file__).with_name('protocol.json').read_text())
    one=plan(p,500,58838549659,780970016)
    two=plan(p,500,58838549659,780970016,cases=2)
    four=plan(p,1000,58838549659,780970016,cases=4)
    assert one['concurrent_cases']==1 and (two['concurrent_cases'],four['concurrent_cases'])==(2,4)
    for result in (two,four):
        assert 1<=result['lead_window']<one['lead_window']
        assert [h for w in result['windows'] for h in w]==p['lead_hours']
        cases=result['concurrent_cases']
        assert result['fixed_bytes']+cases*result['bytes_per_lead']*result['lead_window']<=result['disk_allowance_bytes']
    with pytest.raises(ValueError,match='too small'):plan(p,500,58838549659,780970016,cases=4)


def test_lead_chunks_merge_to_the_unsplit_window(tmp_path):
    from window_scores import merge_lead_chunks,canonical_truth_sha256
    leads=[12,24,36,48]
    def report(part):
        per_lead={str(h):f't{h}' for h in part}
        return dict(status='passed',protocol_sha256='p',lead_hours=part,members=56,
            prediction_manifests_sha256=['m0','m1'],truth_lead_sha256=per_lead,
            truth_values_sha256=canonical_truth_sha256(per_lead),
            per_case=[dict(lead_hours=h,variable='t',crps=float(h)) for h in part])
    whole=tmp_path/'whole.json';whole.write_text(json.dumps(report(leads)))
    paths=[]
    for part in ([12],[24,36],[48]):
        path=tmp_path/f'part-{part[0]}.json';path.write_text(json.dumps(report(part)));paths.append(path)
    merged=merge_lead_chunks(paths,leads);unsplit=json.loads(whole.read_text())
    assert len(merged.pop('lead_chunk_reports_sha256'))==3
    assert merged==unsplit
    r=json.loads(paths[1].read_text());r['prediction_manifests_sha256']=['other'];paths[1].write_text(json.dumps(r))
    with pytest.raises(ValueError,match='different forecasts'):merge_lead_chunks(paths,leads)


def test_concurrent_cases_use_distinct_gpus_and_cover_every_case(tmp_path,monkeypatch):
    import threading,time
    import verify_track,windowed_evaluation
    from training_data import sha256
    p,output=main_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1,2,3')
    monkeypatch.setattr(sys,'argv',sys.argv+['--disk-allowance-gb','1000','--score-workers','3'])
    seen=[];active={};peak=[0];lock=threading.Lock()
    def case(index,device,package,root,output,protocol_path,p,storage,execute,model_python,tracker_bin,scales,budget,**options):
        with lock:
            assert device not in active.values();active[index]=device;peak[0]=max(peak[0],len(active))
        assert options['score_chunk']==1
        options['run_jobs']([lambda:time.sleep(.05)]*4)
        for endpoint in verify_track.ENDPOINTS:
            path=output/endpoint;path.mkdir(exist_ok=True)
            r=dict(status='passed',protocol_sha256=sha256(protocol_path),members=p['members'])
            if endpoint=='scores':
                r['per_case']=[dict(case=p['evaluation_initializations'][index],variable=n,level_hpa=level,lead_hours=h,crps=1.)
                    for n,s in p['variables'].items() for level in (p['pressure_levels_hpa'] if s['pressure_levels'] else [None]) for h in p['lead_hours']]
            (path/f'case-{index:03d}.json').write_text(json.dumps(r))
        replay=output/f'replay-{index:03d}';replay.mkdir();(replay/'replay.json').write_text('{}')
        time.sleep(.1)
        with lock:seen.append(index);del active[index]
    monkeypatch.setattr(windowed_evaluation,'evaluate_case',case)
    verify_track.main()
    assert sorted(seen)==list(range(8)) and peak[0]==4
    status=json.loads((output/'status.json').read_text())
    assert status['max_inflight_cases']==4 and json.loads((output/'reward.json').read_text())['reward']==.5


def test_first_failure_is_reported_and_stops_other_cases(tmp_path,monkeypatch):
    import time
    import verify_track,windowed_evaluation
    _,output=main_fixture(tmp_path,monkeypatch)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    started=[]
    def case(index,device,package,root,output,protocol_path,p,storage,execute,model_python,tracker_bin,scales,budget,**options):
        started.append(index)
        if index==0:raise ValueError('root cause')
        execute([sys.executable,'-c','import time; time.sleep(30)'],f'case-{index}-long')
    monkeypatch.setattr(windowed_evaluation,'evaluate_case',case)
    begin=time.monotonic()
    with pytest.raises(ValueError,match='root cause'):verify_track.main()
    assert time.monotonic()-begin<20 and len(started)<8
    assert 'root cause' in json.loads((output/'status.json').read_text())['error']
