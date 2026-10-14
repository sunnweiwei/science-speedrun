"""Validate an explicit AR3–AR8 schedule branch from a completed AR2 state."""


def validate_branch(header, parent, child, stage_index):
    if stage_index != 5 or header.get('stage_index') != 4 or not header.get('stage_complete'):
        raise ValueError('AR extension must branch from completed AR2 into AR3')
    if len(parent['stages']) != 11 or len(child['stages']) != 11:
        raise ValueError('Expected eleven stages')
    if parent['stages'][:5] != child['stages'][:5]:
        raise ValueError('AR extension changes earlier stages')
    if header.get('stage_updates') != parent['stages'][4]['short_updates']:
        raise ValueError('AR2 checkpoint does not finish parent stage')
    if header.get('global_updates') != sum(s['short_updates'] for s in parent['stages'][:5]):
        raise ValueError('Checkpoint update count differs from parent prefix')
    for old,new in zip(parent['stages'][5:],child['stages'][5:]):
        changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        if changed-{'short_updates','warmup_equivalent_updates'}:
            raise ValueError('AR extension changes model, optimizer or stage definition')
        if type(new['short_updates']) is not int or new['short_updates'] <= old['short_updates']:
            raise ValueError('Each AR3–AR8 stage must gain updates')
        expected=new['short_updates']*new['warmup']/new['updates']
        if abs(new['warmup_equivalent_updates']-expected)>1e-12:
            raise ValueError('Wrong extended warmup metadata')
