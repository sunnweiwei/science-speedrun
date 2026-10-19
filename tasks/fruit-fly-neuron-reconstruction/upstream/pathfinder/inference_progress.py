"""Rate-limited local JSONL progress, independent of model-quality measurement."""
import json
from pathlib import Path
import time


class Progress:
    def __init__(self,path,stage,shard_index,interval=60):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stage=stage;self.shard=shard_index;self.interval=interval;self.started=time.monotonic();self.last=None
        # Never append a second run to an existing log.
        with self.path.open('x'):pass
    def emit(self,force=False,**values):
        now=time.monotonic()
        if not force and self.last is not None and now-self.last<self.interval:return
        row=dict(stage=self.stage,shard_index=self.shard,elapsed_seconds=now-self.started,
            utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),quality_metric=False,**values)
        with self.path.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        self.last=now
