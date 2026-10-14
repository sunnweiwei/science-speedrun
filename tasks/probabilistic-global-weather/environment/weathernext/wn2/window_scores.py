"""Validate and merge disjoint lead-window reports without averaging windows."""
import copy
import hashlib
import json
from pathlib import Path

LIST_FIELDS=('per_case','pooled_per_case','rev_per_case','seeps_per_case','rows')
VARIABLE_FIELDS=set(LIST_FIELDS)|{'lead_hours','truth_values_sha256','prediction_manifests_sha256',
    'rev_threshold_receipts','seeps_climatology_receipts','truth_lead_sha256','lead_chunk_reports_sha256'}


def canonical_truth_sha256(per_lead):
    return hashlib.sha256(json.dumps(per_lead,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def checked_leads(requested, canonical):
    leads=canonical if requested is None else requested
    if not leads or leads!=sorted(set(leads)) or any(x not in canonical for x in leads):
        raise ValueError('Invalid scoring window')
    if leads!=list(range(leads[0],leads[-1]+1,12)):
        raise ValueError('Scoring window must be consecutive')
    return leads


def merge_windows(paths, canonical):
    reports=[json.loads(Path(path).read_text()) for path in paths]
    if not reports or [lead for r in reports for lead in r['lead_hours']]!=canonical:
        raise ValueError('Missing, overlapping or reordered score windows')
    for r in reports:
        if r.get('status')!='passed':raise ValueError('Failed score window')
        for key in reports[0].keys()-VARIABLE_FIELDS:
            if r.get(key)!=reports[0][key]:raise ValueError(f'Changed score identity: {key}')
        for key in LIST_FIELDS:
            for row in r.get(key,[]):
                if row['lead_hours'] not in r['lead_hours']:raise ValueError('Row outside owned score window')
    result=copy.deepcopy(reports[0]);result['lead_hours']=canonical
    for key in LIST_FIELDS:
        if key in result:result[key]=[row for r in reports for row in r[key]]
    for key in ('rev_threshold_receipts','seeps_climatology_receipts'):
        if key in result:
            merged={}
            for r in reports:
                for name,digest in r[key].items():
                    if name in merged and merged[name]!=digest:raise ValueError('Changed climatology receipt')
                    merged[name]=digest
            result[key]=merged
    # Truth identity must not depend on the disk allowance/window partition.
    if 'truth_lead_sha256' in result:
        combined={}
        for r in reports:
            per_lead=r['truth_lead_sha256']
            if (set(per_lead)!={str(h) for h in r['lead_hours']}
                    or canonical_truth_sha256(per_lead)!=r['truth_values_sha256']):
                raise ValueError('Invalid per-lead truth inventory')
            combined.update(per_lead)
        result['truth_lead_sha256']=combined
        result['truth_values_sha256']=canonical_truth_sha256(combined)
    elif 'truth_values_sha256' in result:
        # Spectral windows all reference the same immutable full truth report.
        if any(r['truth_values_sha256']!=result['truth_values_sha256'] for r in reports):
            raise ValueError('Changed spectral truth identity')
    if 'prediction_manifests_sha256' in result:
        result['prediction_manifests_sha256']=[v for r in reports for v in r['prediction_manifests_sha256']]
    if any('lead_chunk_reports_sha256' in r for r in reports):
        # A window short enough to need no split has no chunks of its own.
        result['lead_chunk_reports_sha256']=[v for r in reports for v in r.get('lead_chunk_reports_sha256',[])]
    result['window_reports_sha256']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    return result


def merge_lead_chunks(paths, leads):
    """One window's report from processes that each scored a consecutive part of its leads."""
    reports=[json.loads(Path(path).read_text()) for path in paths]
    if any(r.get('prediction_manifests_sha256')!=reports[0].get('prediction_manifests_sha256') for r in reports):
        raise ValueError('Lead chunks scored different forecasts')
    result=merge_windows(paths,leads)
    # Every chunk read the same window, so its manifest list is recorded once, as for an unsplit window.
    if 'prediction_manifests_sha256' in result:
        result['prediction_manifests_sha256']=reports[0]['prediction_manifests_sha256']
    result['lead_chunk_reports_sha256']=result.pop('window_reports_sha256')
    return result
