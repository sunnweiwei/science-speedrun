import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from calibrated_reward_v2 import score_context,aggregate_contexts


def test_no_signal_and_repeat_controls_define_endpoints():
    assert score_context(loss=2.,negative_loss=2.,repeat_loss=.5,gap_lower95=.1)['reward']==0
    assert score_context(loss=.5,negative_loss=2.,repeat_loss=.5,gap_lower95=.1)['reward']==1
    assert score_context(loss=0.,negative_loss=2.,repeat_loss=.5,gap_lower95=.1)['raw_skill']>1


def test_worse_predictions_never_benefit_from_failure():
    good=score_context(loss=1.,negative_loss=2.,repeat_loss=.5,gap_lower95=.1)
    worse=score_context(loss=5.,negative_loss=2.,repeat_loss=.5,gap_lower95=.1)
    invalid=score_context(loss=None,negative_loss=2.,repeat_loss=.5,gap_lower95=.1,status='invalid_submission')
    assert good['reward']>worse['reward']==invalid['reward']==0
    assert worse['raw_skill']<0 and invalid['raw_skill'] is None


def test_infrastructure_and_bad_reference_cannot_become_zero_scores():
    with pytest.raises(RuntimeError):score_context(loss=None,negative_loss=2.,repeat_loss=.5,gap_lower95=.1,status='infrastructure_error')
    for negative,repeat in [(1.,1.),(1.,2.),(float('nan'),0.),(float('inf'),0.)]:
        with pytest.raises(ValueError):score_context(loss=0.,negative_loss=negative,repeat_loss=repeat,gap_lower95=.1)
    for lower in [0.,float('nan'),2.]:
        with pytest.raises(ValueError):score_context(loss=1.,negative_loss=2.,repeat_loss=.5,gap_lower95=lower)
    with pytest.raises(ValueError):score_context(loss=float('nan'),negative_loss=2.,repeat_loss=.5,gap_lower95=.1)


def test_missing_hard_context_cannot_increase_overall():
    one={'loss':.5,'negative_loss':2.,'repeat_loss':.5,'gap_lower95':.1}
    bad={'loss':None,'negative_loss':2.,'repeat_loss':.5,'gap_lower95':.1,'status':'invalid_submission'}
    r=aggregate_contexts({'a':one,'b':bad},expected_contexts=['a','b'])
    assert r['reward']==.5
    with pytest.raises(ValueError):aggregate_contexts({'a':one},expected_contexts=['a','b'])


def test_anonymous_names_and_context_order_do_not_change_score():
    rows={'a':{'loss':1.,'negative_loss':2.,'repeat_loss':.5,'gap_lower95':.1},'b':{'loss':2.,'negative_loss':3.,'repeat_loss':1.,'gap_lower95':.1}}
    a=aggregate_contexts(rows,expected_contexts=['a','b'])
    b=aggregate_contexts(dict(reversed(list(rows.items()))),expected_contexts=['b','a'])
    assert math.isclose(a['reward'],b['reward'])
