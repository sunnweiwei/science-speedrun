"""Measure exact-byte archive verification on a real retained artifact, one process per mode."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--mode',choices=('materialized','streaming'),required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh benchmark output required')
    sys.path.insert(0,str(a.source))
    import window_io
    record=json.loads(a.receipt.read_text())['storage']
    start=time.monotonic();cpu=time.process_time()
    if a.mode=='streaming':result=window_io.verify_compressed(record)
    else:
        with window_io.window_array(record) as array:
            result=dict(sha256=record['sha256'],shape=list(array.shape),uncompressed_bytes=record['uncompressed_bytes'])
    result.update(mode=a.mode,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  artifact=record['path'],archive_sha256=record['archive_sha256'],
                  verifier_source=str(Path(window_io.__file__).resolve()),verifier_sha256=window_io.sha256(Path(window_io.__file__)),
                  compressed_bytes=Path(record['path']).stat().st_size)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
