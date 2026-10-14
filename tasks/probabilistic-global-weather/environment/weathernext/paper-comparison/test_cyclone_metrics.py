import numpy as np
from cyclone_metrics import distance_km,mean_position,match_initial,position_rows,paired_position_rows,cell_events,cell_confusion
from rev import potential_value

INIT=np.datetime64('2022-01-05T00','ns')
NEXT=INIT+np.timedelta64(12,'h')


def test_dateline_mean_and_known_distance():
    np.testing.assert_allclose(distance_km([0,0],[0,90]),np.pi*6371.0088/2)
    p=mean_position([[0,359],[0,1]])
    assert distance_km(p,[0,0])<1e-8


def test_matching_is_unique_and_uses_only_initialization():
    truth={'a':{INIT:[0,0],NEXT:[0,180]},'b':{INIT:[0,1],NEXT:[0,0]}}
    tracks={'x':{INIT:[0,.2],NEXT:[0,0]},'y':{INIT:[0,.8],NEXT:[0,180]},'far':{INIT:[40,40]}}
    assert match_initial(tracks,truth,INIT)=={'a':'x','b':'y'}
    # Maximum cardinality matters when two tracks both prefer one storm.
    tracks={'x':{INIT:[0,.2]},'y':{INIT:[0,-.5]}}
    assert match_initial(tracks,truth,INIT)=={'b':'x','a':'y'}


def test_missing_members_and_paired_intersection():
    truth={'a':{INIT:[0,0],NEXT:[0,0]}}
    live={'t':{INIT:[0,0],NEXT:[0,1]}}
    dead={'t':{INIT:[0,0]}}
    assert position_rows([live,dead,dead,dead],truth,INIT,[12])==[]
    rows=position_rows([live,live,dead,dead],truth,INIT,[12])
    assert rows[0]['members_present']==2
    assert len(paired_position_rows(rows,rows))==1
    assert paired_position_rows(rows,[])==[]


def test_cells_wrap_and_count_members_not_tracks():
    tracks={'a':{INIT:[0,-.1]},'b':{INIT:[.5,359.9]},'north':{INIT:[90,360]}}
    grid=cell_events(tracks,INIT)
    assert grid.sum()==2 and grid[90,359] and grid[179,0]
    matrix=cell_confusion([tracks,tracks],tracks,INIT)
    np.testing.assert_allclose(potential_value(matrix,[.001,.01,.5]),1,atol=1e-12)


def test_ibtracs_iso_time_preserves_intended_second_and_rejects_disagreement():
    import pytest
    from prepare_named_cyclones import canonical_times
    text=np.array([b'2022-01-05 00:00:00',b''])
    numeric=np.array([INIT+np.timedelta64(39936,'ns'),np.datetime64('NaT','ns')])
    result=canonical_times(text,numeric)
    assert result[0]==INIT and np.isnat(result[1])
    numeric[0]+=np.timedelta64(1,'s')
    with pytest.raises(ValueError,match='Inconsistent'):canonical_times(text,numeric)


def test_cyclone_comparison_preserves_paired_support_and_rejects_changed_truth():
    import copy
    import pytest
    from compare_cyclones import compare,MATCH
    from rev import confusion
    matrix=confusion(np.tile([0.,1.],(56,1)),np.array([0.,1.]),np.array([.5,.5]),np.ones(2)).tolist()
    row=dict(storm='a',lead_hours=12,truth_position=[0.,0.],error_km=100.)
    base=dict(**{key:'same' for key in MATCH},status='passed',case='d0',lineage={'seed':0},
        observed_tracks_receipt_sha256='truth0',position_rows=[row],
        rev_rows=[dict(lead_hours=lead,confusion=matrix) for lead in range(12,217,12)])
    base['members']=56
    other=copy.deepcopy(base);other.update(case='d1',observed_tracks_receipt_sha256='truth1',position_rows=[])
    candidates=copy.deepcopy([base,other]);candidates[0]['position_rows'][0]['error_km']=80.
    result=compare([base,other],candidates,['d0','d1'])
    assert result['position'][0]['paired_storm_initializations']==1
    assert result['position'][0]['candidate_mean_error_km']==80.
    assert result['position'][1]['baseline_mean_error_km'] is None
    np.testing.assert_allclose(result['rev'][0]['baseline_potential_rev'],1,atol=1e-12)
    candidates[0]['observed_tracks_receipt_sha256']='changed'
    with pytest.raises(ValueError,match='Incompatible'):compare([base,other],candidates,['d0','d1'])
