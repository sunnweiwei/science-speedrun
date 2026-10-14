"""Area-weighted potential REV, GenCast Appendix E.2.6 equations E11--E17.

Aggregate confusion matrices over dates before maximizing over decision
thresholds. No evaluation-period climatology is used for extreme thresholds.
"""
import numpy as np


def confusion(prediction, truth, event_threshold, weights, high=True):
    prediction=np.asarray(prediction)
    truth=np.asarray(truth)
    threshold=np.asarray(event_threshold)
    weights=np.broadcast_to(np.asarray(weights),truth.shape)
    if prediction.shape[1:]!=truth.shape or threshold.shape!=truth.shape:
        raise ValueError('Misaligned event panels')
    if len(prediction)<2 or not np.isfinite(prediction).all() or not np.isfinite(threshold).all():
        raise ValueError('Invalid ensemble/threshold')
    if np.isinf(truth).any() or not np.isfinite(weights).all() or np.any(weights<0):
        raise ValueError('Invalid truth/weights')
    valid=np.isfinite(truth)
    weights=np.where(valid,weights,0.)
    if weights.sum()<=0: raise ValueError('Empty verification mask')
    counts=np.sum(prediction>threshold if high else prediction<threshold,axis=0)
    observed=truth>threshold if high else truth<threshold
    n=len(prediction)
    positive=np.bincount(counts[valid],weights=weights[valid]*observed[valid],minlength=n+1)
    negative=np.bincount(counts[valid],weights=weights[valid]*(~observed[valid]),minlength=n+1)
    tp=np.append(np.cumsum(positive[::-1])[::-1],0.)/weights.sum()
    fp=np.append(np.cumsum(negative[::-1])[::-1],0.)/weights.sum()
    base=positive.sum()/weights.sum()
    return np.stack([tp,fp,base-tp,1-base-fp],axis=-1)


def potential_value(mean_confusion,cost_loss_ratios):
    matrix=np.asarray(mean_confusion,dtype=float)
    costs=np.asarray(cost_loss_ratios,dtype=float)
    if matrix.ndim!=2 or matrix.shape[1]!=4 or not np.isfinite(matrix).all():
        raise ValueError('Expected decision-threshold x (TP, FP, FN, TN)')
    if np.any(costs<=0) or np.any(costs>=1): raise ValueError('Cost/loss ratios must lie inside (0,1)')
    np.testing.assert_allclose(matrix.sum(axis=-1),1.,atol=1e-10)
    tp,fp,fn,tn=matrix.T
    rates=tp+fn
    np.testing.assert_allclose(rates,rates[0],atol=1e-10)
    base=rates[0]
    climatology=np.minimum(base,costs)
    perfect=base*costs
    denominator=climatology-perfect
    forecast=(tp+fp)[:,None]*costs[None,:]+fn[:,None]
    value=np.divide(climatology[None,:]-forecast,denominator[None,:],
                    out=np.full_like(forecast,np.nan),where=denominator[None,:]>0)
    # An all-positive or all-negative truth sample does not identify REV.
    return np.max(value,axis=0)


class Thresholds:
    """Read only verified paper-period monthly/hourly climatology groups."""
    def __init__(self, root):
        import json
        from pathlib import Path
        import hashlib
        self.root=Path(root)
        source=self.root/'source.json'
        self.source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
        spec=json.loads(source.read_text())
        if spec['years']!=[1993,2016] or spec['method']!='linear':
            raise ValueError('REV climatology differs from frozen method')
        self.spec=spec
        self.cache={}
        self.receipts={}

    def get(self, valid_time):
        import json
        import hashlib
        import xarray as xr
        stamp=np.datetime64(valid_time,'h')
        month=int(str(stamp.astype('datetime64[M]'))[-2:])
        hour=int((stamp-stamp.astype('datetime64[D]'))/np.timedelta64(1,'h'))
        key=(month,hour)
        if key not in self.cache:
            stem=f'month-{month:02d}-hour-{hour:02d}'
            path=self.root/f'{stem}.nc';receipt=self.root/f'{stem}.json'
            record=json.loads(receipt.read_text())
            if record['status']!='passed' or record['spec']['years']!=[1993,2016] or record['spec']['month']!=month or record['spec']['hour']!=hour:
                raise ValueError('Invalid REV threshold receipt')
            for name in ('years','method','quantiles','fields','source'):
                if record['spec'][name]!=self.spec[name]:raise ValueError('Mixed climatology methods/sources')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('Corrupt REV thresholds')
            self.cache[key]=xr.load_dataset(path)
            self.receipts[receipt.name]=hashlib.sha256(receipt.read_bytes()).hexdigest()
        return self.cache[key]


def panel_rows(prediction,truth,weights,thresholds,case,lead,variable):
    """Preserve additive confusion matrices for later cross-date aggregation."""
    import xarray as xr
    directions={'2m_temperature':[True,False], '10m_wind_speed':[True], 'mean_sea_level_pressure':[False]}
    if variable not in directions:return []
    source=thresholds.get(np.datetime64(case)+np.timedelta64(lead,'h'))
    rows=[]
    for high in directions[variable]:
        for quantile in ([.99,.999,.9999] if high else [.01,.001,.0001]):
            threshold=source[variable].sel(quantile=quantile,drop=True)
            xr.align(truth,threshold,join='exact',copy=False)
            matrix=confusion(prediction.transpose('member','lat','lon').values,
                truth.transpose('lat','lon').values,threshold.transpose('lat','lon').values,weights,high=high)
            rows.append(dict(case=case,variable=variable,lead_hours=lead,quantile=quantile,
                direction='high' if high else 'low',confusion=matrix.tolist()))
    return rows
