"""Explicit decimal-GB storage planning and measured fail-closed disk allowance."""
import math
from pathlib import Path
import threading
import time

GIB=1024**3


def tree_bytes(root):
    root=Path(root)
    if not root.exists():return 0
    total=0
    for path in root.rglob('*'):
        try:
            if path.is_file():total+=path.stat().st_size
        except FileNotFoundError:pass  # released by a concurrent case between listing and stat
    return total


def plan(protocol, allowance_gb, resource_bytes, package_bytes, reserve_gb=25, cases=1):
    """Windows for `cases` cases evaluated at once; each holds its own model copies, states and
    tracker buffer, and all share the benchmark resources and the reserve."""
    if not math.isfinite(allowance_gb) or not math.isfinite(reserve_gb) or allowance_gb<=0 or reserve_gb<0:
        raise ValueError('Invalid disk allowance/reserve')
    if cases<1:raise ValueError('At least one case at a time')
    cells=math.prod(protocol['grid']);members=protocol['members']
    planes=sum(len(protocol['pressure_levels_hpa']) if s['pressure_levels'] else 1 for s in protocol['variables'].values())+1
    lead_bytes=cells*4*planes*members
    # Two immutable generations of 56 two-frame native float32 states plus
    # forcings/coordinates. Each submission must respect the same state limit.
    state_bytes=math.ceil(cells*4*190*members*1.1)
    tracking_bytes=cells*4*3*len(protocol['lead_hours'])*members
    per_case=2*package_bytes+2*state_bytes+tracking_bytes+cells*4*members
    fixed=resource_bytes+cases*per_case+int(reserve_gb*1e9)
    count=min(len(protocol['lead_hours']),int((allowance_gb*1e9-fixed)//(cases*lead_bytes)))
    if count<1:
        raise ValueError(f'Disk allowance too small: need at least {(fixed+cases*lead_bytes)/1e9:.2f} GB '
                         f'with this reserve/package for {cases} concurrent case(s)')
    return dict(disk_allowance_bytes=int(allowance_gb*1e9),resource_bytes=resource_bytes,
                package_bytes=package_bytes,reserve_bytes=int(reserve_gb*1e9),state_limit_bytes=state_bytes,
                bytes_per_lead=lead_bytes,fixed_bytes=fixed,lead_window=count,concurrent_cases=cases,
                windows=[protocol['lead_hours'][i:i+count] for i in range(0,len(protocol['lead_hours']),count)],
                forecast_recomputation=False)


class DiskBudget:
    def __init__(self, output, plan, interval=0.0):
        self.output=Path(output);self.plan=plan;self.peak=0
        # Concurrent cases all poll this; one walk of the tree per interval serves them all.
        self.interval=interval;self.lock=threading.Lock();self.measured=-math.inf;self.used=0

    def check(self):
        with self.lock:
            if time.monotonic()-self.measured>=self.interval:
                self.used=tree_bytes(self.output)+self.plan['resource_bytes']+self.plan['package_bytes']+self.plan['reserve_bytes']
                self.measured=time.monotonic()
            used=self.used
            self.peak=max(self.peak,used)
        if used>self.plan['disk_allowance_bytes']:
            raise RuntimeError(f'Disk allowance exceeded: {used} > {self.plan["disk_allowance_bytes"]} bytes')
        return used


if __name__=='__main__':
    import argparse
    import json
    ap=argparse.ArgumentParser(description='Plan evaluation windows without training or allocating a GPU')
    ap.add_argument('--benchmark',required=True,type=Path)
    ap.add_argument('--package',required=True,type=Path)
    ap.add_argument('--disk-allowance-gb',type=float,default=500)
    ap.add_argument('--disk-reserve-gb',type=float,default=25)
    a=ap.parse_args()
    p=json.loads(Path(__file__).with_name('protocol.json').read_text())
    resources=json.loads((a.benchmark/'resources.json').read_text())
    size=sum((a.benchmark/name).stat().st_size for name in resources['files'])+(a.benchmark/'resources.json').stat().st_size
    print(json.dumps(plan(p,a.disk_allowance_gb,size,tree_bytes(a.package),a.disk_reserve_gb),indent=2))
