import copy
import pytest
from ar_extension import validate_branch
from plan_schedule import STAGES


def fixture():
    p={'stages':[s|dict(short_updates=1,warmup_equivalent_updates=s['warmup']/s['updates']) for s in STAGES]}
    c=copy.deepcopy(p)
    for s in c['stages'][5:]:s.update(short_updates=4,warmup_equivalent_updates=4*s['warmup']/s['updates'])
    h=dict(stage_index=4,stage_complete=True,stage_updates=1,global_updates=5)
    return h,p,c


def test_branch_preserves_prefix_and_native_configuration():
    h,p,c=fixture();validate_branch(h,p,c,5)
    c['stages'][6]['peak_lr']*=2
    with pytest.raises(ValueError,match='definition'):validate_branch(h,p,c,5)


def test_rejects_wrong_checkpoint_and_changed_prefix():
    h,p,c=fixture();h['stage_complete']=False
    with pytest.raises(ValueError,match='completed AR2'):validate_branch(h,p,c,5)
    h['stage_complete']=True;c['stages'][4]['short_updates']=2
    with pytest.raises(ValueError,match='earlier'):validate_branch(h,p,c,5)
