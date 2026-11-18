"""Model-blind population metrics of the candidate v2 reward (METRIC_VALIDITY_V2.md section 2).

WMSE and squared min-max absolute t-score weights follow Shift Bioscience's
Nature Biotechnology calibration paper, DOI 10.1038/s41587-026-03307-w,
official implementation commit 15361aec8392fb45a75c8c9da88ca5e6510fca97.
The SSI cross-fitting and target exclusions are explicit local adaptations.
grader.py scores every context against a reference bundle built here from the
private responses only; predictions and model names never enter the bundle.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

SEED = 20261114
SPLITS = 5
BOOTSTRAP = 5000
BUNDLE_VERSION = 'ssi-vcc-wmse-reference-bundle/1'


def finite_matrix(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or min(x.shape)==0 or not np.isfinite(x).all():
        raise ValueError('Expected a nonempty finite matrix')
    return x


def reference_weights(sums,squares,counts,excluded):
    """One-vs-rest t-test_overestim_var; neither predictions nor model IDs enter."""
    sums=finite_matrix(sums);squares=finite_matrix(squares)
    n=np.asarray(counts,dtype=np.float64)[:,None]
    if squares.shape!=sums.shape or n.shape!=(sums.shape[0],1) or (n<2).any():
        raise ValueError('Insufficient or inconsistent reference statistics')
    other_n=n.sum()-n
    mean=sums/n;other_mean=(sums.sum(0)-sums)/other_n
    var=np.maximum((squares-sums*sums/n)/(n-1),0)
    other_var=np.maximum((squares.sum(0)-squares-(sums.sum(0)-sums)**2/other_n)/(other_n-1),0)
    # Scanpy overestimates rest variance by using the target's n for both terms.
    denominator=np.sqrt((var+other_var)/n)
    if np.any((denominator==0)&(np.abs(mean-other_mean)>1e-12)):
        raise ValueError('Degenerate reference variance with nonzero effect')
    t=np.divide(mean-other_mean,denominator,out=np.zeros_like(mean),where=denominator>0)
    if not np.isfinite(t).all():raise ValueError('Nonfinite reference t statistic')
    a=np.abs(t);span=a.max(1)-a.min(1)
    weights=np.divide(a-a.min(1)[:,None],span[:,None],out=np.zeros_like(a),where=span[:,None]>0)**2
    for i,j in enumerate(excluded):
        if j is not None:weights[i,j]=0
    total=weights.sum(1)
    if (total<=0).any():raise ValueError('Uninformative reference weights; cannot invent a score')
    return weights/total[:,None]


def wmse_rows(prediction,truth,weights):
    p=finite_matrix(prediction);y=finite_matrix(truth);w=finite_matrix(weights)
    if p.shape!=y.shape or p.shape!=w.shape or (w<0).any() or not np.allclose(w.sum(1),1):
        raise ValueError('Metric shapes/weights mismatch')
    return np.sum(w*(p-y)**2,axis=1)


def nir_rows(prediction,truth,keep):
    """Euclidean NIR with explicit half-credit ties; fixed panel for every model."""
    p=finite_matrix(prediction);y=finite_matrix(truth)
    if p.shape!=y.shape or len(p)<2 or not np.any(keep):raise ValueError('Invalid retrieval panel')
    d=cdist(p[:,keep],y[:,keep]);correct=np.diag(d)[:,None]
    win=(d>correct).astype(float)+.5*np.isclose(d,correct,rtol=1e-12,atol=1e-12)
    # Do not count a near-tie as both a win and a half tie.
    win[np.isclose(d,correct,rtol=1e-12,atol=1e-12)]=.5
    np.fill_diagonal(win,0)
    return win.sum(1)/(len(p)-1)


def log1p_cp10k(x):
    """Per-cell CP10k then log1p (float64 CSR). Only all-zero rows use total 1, so the
    result is invariant to scaling any positive row, including rows whose total is < 1."""
    x=sparse.csr_matrix(x,dtype=np.float64)
    if not np.isfinite(x.data).all() or (x.data<0).any():
        raise ValueError('Expression must be finite and nonnegative')
    total=np.asarray(x.sum(axis=1)).ravel()
    if not np.isfinite(total).all():raise ValueError('Nonfinite per-cell totals')
    y=sparse.diags(1e4/np.where(total==0,1.,total))@x
    y.data=np.log1p(y.data)
    return y.tocsr()


def target_means(matrix,labels,targets,chunk=16384):
    """Mean log1p(CP10k) profile per target; rows of `matrix` (in memory or backed) are read in chunks."""
    targets=list(targets);index={t:i for i,t in enumerate(targets)}
    if len(index)!=len(targets):raise ValueError('Duplicate targets')
    rows=np.array([index.get(x,-1) for x in np.asarray(labels).astype(str)],dtype=np.int64)
    if len(rows)!=matrix.shape[0]:raise ValueError('Labels do not match rows')
    counts=np.bincount(rows[rows>=0],minlength=len(targets))
    if (counts==0).any():raise ValueError('A target has no cells')
    sums=np.zeros((len(targets),matrix.shape[1]))
    for start in range(0,len(rows),chunk):
        r=rows[start:start+chunk];keep=np.flatnonzero(r>=0)
        block=log1p_cp10k(matrix[start:start+len(r)])  # also validates rows outside the targets
        member=sparse.csr_matrix((np.ones(len(keep)),(r[keep],keep)),shape=(len(targets),len(r)))
        sums+=(member@block).toarray()
    return sums/counts[:,None]


def build_reference_bundle(log_expression,labels,targets,excluded,*,seed=SEED,splits=SPLITS,n_boot=BOOTSTRAP):
    """Reference-only constants for one context. `log_expression` is log1p(CP10k) of the real
    cells of the cohort targets (controls excluded). Per split/direction: weights from one
    half, truth means of the other; N = equal-target mean of the weight half, P = its same-target mean."""
    labels=np.asarray(labels).astype(str);targets=list(targets)
    index=[np.flatnonzero(labels==t) for t in targets]
    if min(map(len,index))<4:raise ValueError('Insufficient private reference cells')
    weights,truths,negative,repeat,half_cells=[],[],[],[],[]
    for split in range(splits):
        halves=[[],[]]
        for k,ix in enumerate(index):
            order=np.random.default_rng(seed+split*10000+k).permutation(ix);n=len(ix)//2
            halves[0].append(order[:n]);halves[1].append(order[n:2*n])
        stats=[]
        for blocks in halves:
            sums=[];squares=[];counts=[]
            for ix in blocks:
                x=log_expression[ix];sums.append(np.asarray(x.sum(0)).ravel())
                sq=x.copy();sq.data**=2;squares.append(np.asarray(sq.sum(0)).ravel());counts.append(len(ix))
            sums=np.stack(sums);squares=np.stack(squares);counts=np.asarray(counts)
            stats.append((sums,squares,counts,sums/counts[:,None]))
        for direction in (0,1):
            a,b=stats[direction],stats[1-direction]
            w=reference_weights(*a[:3],excluded);truth=b[3]
            weights.append(w);truths.append(truth);half_cells.append(a[2])
            negative.append(wmse_rows(np.tile(a[3].mean(0),(len(targets),1)),truth,w))
            repeat.append(wmse_rows(a[3],truth,w))
    negative=np.mean(negative,axis=0);repeat=np.mean(repeat,axis=0)
    return {'weights':np.stack(weights),'truth':np.stack(truths),'half_cells':np.stack(half_cells),
            'negative_per_target':negative,'repeat_per_target':repeat,**anchor_constants(negative,repeat,seed=seed,n_boot=n_boot)}


def anchor_constants(negative,repeat,*,seed=SEED,n_boot=BOOTSTRAP):
    """N, P and the paired target-bootstrap 95% interval of the gap N-P."""
    gap=np.asarray(negative)-np.asarray(repeat)
    boot=np.random.default_rng(seed).integers(len(gap),size=(n_boot,len(gap)))
    low,high=np.quantile(gap[boot].mean(1),[.025,.975])
    return {'negative_loss':float(np.mean(negative)),'repeat_loss':float(np.mean(repeat)),
            'gap':float(gap.mean()),'gap_lower95':float(low),'gap_upper95':float(high)}


def retrieval_keep_mask(bundle):
    """Boolean gene mask removing all panel target genes on the roster (METRIC_VALIDITY_V2.md section 2)."""
    keep=np.ones(bundle['truth'].shape[2],dtype=bool)
    for x in bundle.get('roster',()):
        j=x.get('readout_gene_index',-1)
        if j is not None and j>=0:keep[j]=False
    return keep


def bundle_repeat_nir(bundle):
    """Cross-fitted split-half repeatability NIR on the non-target gene panel."""
    keep=retrieval_keep_mask(bundle)
    truths=bundle['truth']
    per_target=np.mean([nir_rows(truths[m^1],truths[m],keep) for m in range(len(truths))],axis=0)
    return {'repeat_nir':float(per_target.mean()),'repeat_per_target_nir':per_target.tolist()}


def score_against_bundle(prediction_means,bundle):
    """Mean WMSE and Euclidean NIR (panel target genes excluded) over the fixed split/direction pairs and targets."""
    per_target=np.mean([wmse_rows(prediction_means,t,w) for t,w in zip(bundle['truth'],bundle['weights'])],axis=0)
    keep=retrieval_keep_mask(bundle)
    per_target_nir=np.mean([nir_rows(prediction_means,t,keep) for t in bundle['truth']],axis=0)
    rep=bundle_repeat_nir(bundle)
    return {'loss':float(per_target.mean()),'per_target_loss':per_target.tolist(),
            'nir':float(per_target_nir.mean()),'per_target_nir':per_target_nir.tolist(),
            'negative_nir':0.5,**rep}


def _sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<24),b''):h.update(block)
    return h.hexdigest()


def _panel_sha256(var_names):
    return hashlib.sha256('\n'.join(map(str,var_names)).encode()).hexdigest()


def _read_requests(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['request_id'] for r in rows})!=len(rows):raise ValueError('duplicate request_id')
    return rows


def reference_roster(private_ctx,requests_path,*,development=False):
    """Trusted roster: final-subset requests sorted by request_id (independent of file order);
    the readout gene of each target is resolved through its Ensembl ID, never the symbol alone."""
    import anndata as ad
    private_ctx=Path(private_ctx)
    requests=_read_requests(requests_path)
    if development:
        rows=sorted(requests,key=lambda r:r['request_id'])
        responses_digest=_sha256(private_ctx/'responses.h5ad')
        allowed_exclusions=set()
    else:
        manifest=json.loads((private_ctx/'manifest.json').read_text())
        final={r['request_id'] for r in manifest['requests'] if r['subset']=='final'}
        rows=sorted((r for r in requests if r['request_id'] in final),key=lambda r:r['request_id'])
        if len(rows)!=len(final):raise ValueError('final requests missing from requests.jsonl')
        responses_digest=manifest['responses_sha256']
        # The operator-only roster exception belongs in the private evaluator image.
        rules=Path(__file__).with_name('wmse_roster_exclusions.json')
        allowed_exclusions=set(json.loads(rules.read_text()).get(private_ctx.name,[])) if rules.is_file() else set()
    basal=ad.read_h5ad(Path(private_ctx)/'basal.h5ad',backed='r')
    responses=ad.read_h5ad(Path(private_ctx)/'responses.h5ad',backed='r')
    try:
        if list(map(str,responses.var_names))!=list(map(str,basal.var_names)):raise ValueError('responses genes/order differ from basal')
        if 'target_ensembl' not in responses.obs or 'ensembl_id' not in basal.var:raise ValueError('Missing trusted stable gene IDs')
        panel_ids=basal.var['ensembl_id'].astype(str).str.split('.').str[0]
        if panel_ids.duplicated().any():raise ValueError('Ambiguous gene IDs in panel')
        position={g:i for i,g in enumerate(panel_ids)};symbols={g:i for i,g in enumerate(map(str,basal.var_names))}
        obs=responses.obs[['target_gene','target_ensembl']].astype(str)
        stable=obs.groupby('target_gene',observed=True)['target_ensembl'].agg(lambda s:sorted(set(s.str.split('.').str[0])))
        roster,excluded_targets=[],[]
        for r in rows:
            target=r['perturbation']['target'];ids=stable.get(target,[])
            if len(ids)!=1 or not ids[0].startswith('ENSG'):
                if target not in allowed_exclusions and not (development and len(ids)>0):
                    raise ValueError('Ambiguous target identity: '+target)
                excluded_targets.append({'target':target,'request_id':r['request_id'],'target_ensembl':list(ids),
                                         'reason':'No unique stable gene ID; frozen roster exclusion'})
                continue
            gene=position.get(ids[0])
            if target in symbols and gene!=symbols[target]:raise ValueError('Symbol/Ensembl conflict: '+target)
            roster.append({'request_id':r['request_id'],'target':target,'readout_gene_index':-1 if gene is None else int(gene)})
        if len({x['target'] for x in roster})!=len(roster):raise ValueError('Duplicate target in roster')
        return {'roster':roster,'excluded_targets':excluded_targets,'panel_sha256':_panel_sha256(basal.var_names),
                'responses_sha256':responses_digest}
    finally:
        basal.file.close();responses.file.close()


def load_or_build_bundle(private_ctx,requests_path,cache_dir=None,*,development=False):
    """Load the cached bundle (checked against the roster, panel, responses and its own hash)
    or build it once from the private responses and cache it, by default under private_ctx/wmse_v2."""
    import anndata as ad
    private_ctx=Path(private_ctx);cache=Path(cache_dir) if cache_dir else private_ctx/'wmse_v2'
    info=reference_roster(private_ctx,requests_path,development=development)
    meta_path,array_path=cache/'bundle.json',cache/'bundle.npz'
    expected={'version':BUNDLE_VERSION,'seed':SEED,'splits':SPLITS,'bootstrap':BOOTSTRAP,
              'roster':info['roster'],'excluded_targets':info['excluded_targets'],
              'panel_sha256':info['panel_sha256'],'responses_sha256':info['responses_sha256']}
    if meta_path.is_file() and array_path.is_file():
        meta=json.loads(meta_path.read_text())
        if any(meta.get(k)!=v for k,v in expected.items()) or meta.get('npz_sha256')!=_sha256(array_path):
            raise ValueError(f'{cache}: cached reference bundle does not match the private reference')
        with np.load(array_path) as z:arrays={k:z[k] for k in z.files}
        scalars=anchor_constants(arrays['negative_per_target'],arrays['repeat_per_target'])
        if any(meta.get(k)!=v for k,v in scalars.items()):raise ValueError(f'{cache}: cached anchor constants do not match their arrays')
        return {**meta,**scalars,**arrays,'cache':str(cache)}
    if _sha256(private_ctx/'responses.h5ad')!=info['responses_sha256']:raise ValueError('private response checksum mismatch')
    targets=[x['target'] for x in info['roster']]
    responses=ad.read_h5ad(private_ctx/'responses.h5ad',backed='r')
    try:
        part=responses[responses.obs['target_gene'].astype(str).isin(targets).to_numpy()].to_memory()
    finally:
        responses.file.close()
    labels=part.obs['target_gene'].astype(str).to_numpy()
    excluded=[None if x['readout_gene_index']<0 else x['readout_gene_index'] for x in info['roster']]
    built=build_reference_bundle(log1p_cp10k(part.X),labels,targets,excluded)
    del part
    arrays={k:np.asarray(built[k]) for k in ('weights','truth','half_cells','negative_per_target','repeat_per_target')}
    scalars={k:built[k] for k in ('negative_loss','repeat_loss','gap','gap_lower95','gap_upper95')}
    bundle={**expected,**scalars,**arrays,'cache':'memory'}
    try:
        cache.mkdir(parents=True,exist_ok=True)
        import fcntl
        from tempfile import TemporaryDirectory
        with (cache/'.publish.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not meta_path.exists():
                with TemporaryDirectory(dir=cache) as staging:
                    staged=Path(staging)
                    np.savez(staged/'bundle.npz',**arrays)
                    meta={**expected,**scalars,'npz_sha256':_sha256(staged/'bundle.npz')}
                    (staged/'bundle.json').write_text(json.dumps(meta,indent=1)+'\n')
                    (staged/'bundle.npz').replace(array_path)
                    (staged/'bundle.json').replace(meta_path)
            bundle['cache']=str(cache)
    except OSError:
        pass  # read-only private mount: keep the in-memory bundle; the operator stage pre-builds it
    return bundle


def prediction_means(predictions,basal_path,requests_path,roster):
    """Validate the full prediction contract, then return the roster's mean log1p(CP10k) profiles."""
    import anndata as ad
    from prediction_safety import validate_prediction_storage
    validate_prediction_storage(predictions)
    requests=_read_requests(requests_path)
    basal=ad.read_h5ad(basal_path,backed='r');genes=list(map(str,basal.var_names));basal.file.close()
    pred=ad.read_h5ad(predictions,backed='r')
    try:
        if pred.var_names.has_duplicates or list(map(str,pred.var_names))!=genes:raise ValueError('prediction genes/order differ from basal input')
        if 'request_id' not in pred.obs:raise ValueError('predictions need obs.request_id')
        ids=pred.obs['request_id'].astype(str).to_numpy()
        values,counts=np.unique(ids,return_counts=True)
        if dict(zip(values.tolist(),counts.tolist()))!={r['request_id']:int(r['n_cells']) for r in requests}:
            raise ValueError('prediction request IDs or per-request cell counts differ from requests.jsonl')
        target_of={x['request_id']:x['target'] for x in roster}
        labels=np.array([target_of.get(i,'') for i in ids],dtype=object)
        matrix=pred.X.to_memory().tocsr() if getattr(pred.X,'format',None)=='csc' else pred.X
        return target_means(matrix,labels,[x['target'] for x in roster])
    finally:
        if pred.isbacked:pred.file.close()


def main():
    import argparse
    ap=argparse.ArgumentParser(description='Operator stage: build the v2 reference bundles under PRIVATE/<ctx>/wmse_v2')
    ap.add_argument('--private',required=True,type=Path);ap.add_argument('--benchmark',required=True,type=Path)
    a=ap.parse_args()
    for priv in sorted(p for p in a.private.iterdir() if (p/'responses.h5ad').exists()):
        b=load_or_build_bundle(priv,a.benchmark/priv.name/'public'/'requests.jsonl')
        print(priv.name,len(b['roster']),'targets N',b['negative_loss'],'P',b['repeat_loss'],'gap95',b['gap_lower95'],b['cache'],flush=True)


if __name__=='__main__':
    main()
