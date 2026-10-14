"""SEEPS for ensemble-mean rain using the GraphCast paper's monthly climatology.

Rain and wet thresholds are metres. p1 and wet thresholds are selected from
1980-2015 monthly statistics, not estimated from the evaluation date subset.
The paper defines dry as strictly less than the threshold. WeatherBench-X uses
less-than-or-equal at this boundary; its matrix is an oracle away from that edge.
"""
import numpy as np


class Climatology:
    """Load checksummed monthly paper-period statistics on demand."""
    def __init__(self,root):
        import hashlib,json
        from pathlib import Path
        self.root=Path(root);here=Path(__file__).resolve().parent
        self.protocol_sha256=hashlib.sha256((here/'seeps-protocol.json').read_bytes()).hexdigest()
        self.source_sha256=hashlib.sha256((here/'seeps-climatology-source.lock.json').read_bytes()).hexdigest()
        self.cache={};self.receipts={};self.identity=None

    def get(self,valid_time,hours,dry_mm):
        import hashlib,json
        import xarray as xr
        month=int(str(np.datetime64(valid_time,'M'))[-2:])
        if month not in self.cache:
            file=self.root/f'month-{month:02d}.nc';receipt=file.with_suffix('.json')
            record=json.loads(receipt.read_text());identity=record['identity']
            if (record['status']!='passed' or identity['month']!=month or identity['protocol_sha256']!=self.protocol_sha256
                    or identity['source_lock_sha256']!=self.source_sha256):raise ValueError('Wrong monthly SEEPS method')
            if hashlib.sha256(file.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Corrupt SEEPS climatology')
            years={f'year-{y}.json' for y in range(1980,2016)}
            if record['rain_year_receipts_sha256'].keys()!=years:raise ValueError('Incomplete paper climatology years')
            stable=(identity['runner_sha256'],record['rain_year_receipts_sha256'],record['source_metadata_sha256'])
            if self.identity is not None and stable!=self.identity:raise ValueError('Mixed monthly climatology sources')
            self.identity=stable
            ds=xr.load_dataset(file)
            np.testing.assert_array_equal(ds.lat,np.linspace(-90,90,721));np.testing.assert_array_equal(ds.lon,np.arange(1440)/4)
            np.testing.assert_array_equal(ds.accumulation_hours,[12,24]);np.testing.assert_array_equal(ds.dry_threshold_mm,[.1,.25])
            if ds.wet_threshold.attrs.get('units')!='m':raise ValueError('Wrong wet-threshold units')
            p1=ds.dry_fraction.values;wet=ds.wet_threshold.values
            if not np.isfinite(p1).all() or np.any((p1<0)|(p1>1)) or np.any(~np.isfinite(wet)&(p1!=1)):
                raise ValueError('Invalid climatology values')
            self.cache[month]=ds;self.receipts[receipt.name]=hashlib.sha256(receipt.read_bytes()).hexdigest()
        return self.cache[month].sel(accumulation_hours=hours,dry_threshold_mm=dry_mm,drop=True)


def score(prediction,truth,wet_threshold,climatological_dry_fraction,weights,dry_threshold_mm):
    arrays=np.broadcast_arrays(*[np.asarray(x,dtype=float) for x in
        (prediction,truth,wet_threshold,climatological_dry_fraction,weights)])
    prediction,truth,wet,p1,weights=arrays
    if dry_threshold_mm not in (.1,.25):raise ValueError('Unfrozen dry threshold')
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():raise ValueError('Nonfinite rain')
    if not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('Invalid area weights')
    if np.any(np.isfinite(p1)&((p1<0)|(p1>1))):raise ValueError('Invalid dry fraction')
    valid=np.isfinite(p1)&np.isfinite(wet)&(p1>=.1)&(p1<=.85)
    dry=dry_threshold_mm/1000
    if np.any(valid&(wet<dry)):raise ValueError('Wet threshold below dry threshold')
    # Strict dry boundary follows the paper and the climatology construction.
    def categories(values):
        return np.stack([values<dry,(values>=dry)&(values<wet),values>=wet],axis=-1)
    f=categories(prediction);t=categories(truth)
    q=np.where(valid,p1,.5)
    matrix=np.zeros(q.shape+(3,3),dtype=float)
    matrix[...,0,1]=1/(1-q);matrix[...,0,2]=4/(1-q)
    matrix[...,1,0]=1/q;matrix[...,1,2]=3/(1-q)
    matrix[...,2,0]=1/q+3/(2+q);matrix[...,2,1]=3/(2+q)
    values=.5*np.einsum('...i,...ij,...j->...',f,matrix,t)
    area=np.where(valid,weights,0.);fraction=float(area.sum()/weights.sum())
    return dict(seeps=float(np.sum(np.where(valid,values,0.)*area)/area.sum()) if area.sum()>0 else None,
        valid_area_fraction=fraction,valid_grid_points=int(valid.sum()),dry_threshold_mm=dry_threshold_mm)
