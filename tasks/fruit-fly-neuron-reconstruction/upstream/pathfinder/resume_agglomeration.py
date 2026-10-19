"""Reuse completed FFN candidate scores only under identical input and shard identity."""
from ffn_agglomeration import accept_pair


def resume_rows(prior, decisions, signature, shard_index, shards):
    if prior.get('schema')!='ffn-agglomeration-shard/v1' or prior.get('ground_truth_used') is not False:
        raise ValueError('Prediction-only prior candidate receipt required')
    if prior.get('shard_index')!=shard_index or prior.get('shards')!=shards:
        raise ValueError('Prior candidate shard changed')
    for name,digest in signature.items():
        if prior.get(name)!=digest:raise ValueError('Prior candidate input changed: '+name)
    expected={tuple(row['pair']):row for row in decisions}
    if len(expected)!=len(decisions) or prior.get('requested_candidates')!=len(decisions):
        raise ValueError('Prior candidate set changed')
    retained=[];seen=set()
    for row in prior['results']:
        pair=tuple(row['pair'])
        if pair not in expected or pair in seen or row.get('center_zyx')!=expected[pair]['center_zyx']:
            raise ValueError('Unexpected, duplicate, or moved candidate')
        seen.add(pair)
        if type(row.get('complete')) is not bool or type(row.get('accepted')) is not bool:
            raise ValueError('Explicit candidate completion and acceptance required')
        if row['complete']:
            if accept_pair(row,complete=True)!=row['accepted']:raise ValueError('Prior candidate gate changed')
            retained.append(row)
        elif row['accepted']:raise ValueError('Incomplete candidate was accepted')
    if prior.get('completed_candidates')!=len(retained):raise ValueError('Prior completion count disagrees')
    completed={tuple(row['pair']) for row in retained}
    return retained,[row for row in decisions if tuple(row['pair']) not in completed]


class PendingRows:
    def __init__(self,decisions,completed,count):
        self.decisions=decisions;self.completed=completed;self.count=count
    def __len__(self):return self.count
    def __iter__(self):
        for index in range(len(self.decisions)):
            if not self.completed[index]:yield self.decisions[index]


def resume_to_sink(prior,decisions,signature,shard_index,shards,sink):
    """Validate and stream retained scores; two bytes per assigned decision."""
    from decision_table import DecisionRows
    if not isinstance(decisions,DecisionRows):raise ValueError('Disk decisions required')
    if prior.get('schema')!='ffn-agglomeration-shard/v1' or prior.get('ground_truth_used') is not False:
        raise ValueError('Prediction-only prior candidate receipt required')
    if prior.get('shard_index')!=shard_index or prior.get('shards')!=shards:
        raise ValueError('Prior candidate shard changed')
    for name,digest in signature.items():
        if prior.get(name)!=digest:raise ValueError('Prior candidate input changed: '+name)
    if prior.get('requested_candidates')!=len(decisions):raise ValueError('Prior candidate set changed')
    seen=bytearray(len(decisions));completed=bytearray(len(decisions));retained=accepted=0
    for row in prior['results']:
        index=decisions.pair_index(row['pair'])
        if seen[index] or row.get('center_zyx')!=decisions[index]['center_zyx']:
            raise ValueError('Unexpected, duplicate, or moved candidate')
        seen[index]=1
        if type(row.get('complete')) is not bool or type(row.get('accepted')) is not bool:
            raise ValueError('Explicit candidate completion and acceptance required')
        if row['complete']:
            if accept_pair(row,complete=True)!=row['accepted']:raise ValueError('Prior candidate gate changed')
            sink(row);completed[index]=1;retained+=1;accepted+=row['accepted']
        elif row['accepted']:raise ValueError('Incomplete candidate was accepted')
    if prior.get('completed_candidates')!=retained:raise ValueError('Prior completion count disagrees')
    return dict(count=retained,accepted=accepted),PendingRows(decisions,completed,len(decisions)-retained)
