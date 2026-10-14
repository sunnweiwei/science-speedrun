import pytest
from plan_schedule import STAGES, plan, learning_rate


def profiles():
    return {s['name']:dict(status='passed',devices=4,effective_batch=64,includes_io_and_communication=True,
                          update_seconds=1,overhead_seconds=10) for s in STAGES}


def test_common_factor_budget_and_nonzero_short_stage_lr():
    result=plan(profiles(),training_seconds=2000)
    assert result['training_wall_seconds']<=2000
    for s in result['stages']:
        assert s['short_updates']>=1
        assert 0<learning_rate(s,0)<=s['peak_lr']
        assert 0<learning_rate(s,s['short_updates']-1)<=s['peak_lr']
        assert s['warmup_equivalent_updates']<s['short_updates']


def test_insufficient_or_missing_measurements_cannot_freeze():
    p=profiles();p.pop(next(iter(p)))
    with pytest.raises(ValueError,match='Every'):plan(p)
    with pytest.raises(ValueError,match='smallest'):plan(profiles(),training_seconds=1)
    p=profiles();p[next(iter(p))]['effective_batch']=1
    with pytest.raises(ValueError,match='batch-64'):plan(p)
