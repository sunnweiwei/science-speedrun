"""Hash-bound JSONL score records; incomplete files never become valid receipts."""
import json
from pathlib import Path
from dataset_contract import sha256

class ScoreRows:
    def __init__(self,path,count):self.path=path;self.count=count
    def __len__(self):return self.count
    def __iter__(self):
        count=0
        with self.path.open() as f:
            for line in f:
                row=json.loads(line)
                if not isinstance(row,dict):raise ValueError('Invalid score record')
                count+=1
                if count>self.count:raise ValueError('Score count mismatch')
                yield row
        if count!=self.count:raise ValueError('Incomplete score stream')

class ScoreWriter:
    def __init__(self,path):
        self.path=Path(path);self.partial=self.path.with_suffix(self.path.suffix+'.partial')
        if self.path.exists() or self.partial.exists():raise ValueError('New score stream required')
        self.file=self.partial.open('x');self.count=0
    def append(self,row):
        self.file.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');self.count+=1
    def close(self):self.file.close()
    def finish(self):
        self.close();self.partial.replace(self.path)
        return dict(storage='ffn-score-jsonl/v1',file=self.path.name,count=self.count,sha256=sha256(self.path))


def load_shard(path):
    path=Path(path);shard=json.loads(path.read_text())
    if 'result_stream' not in shard:return shard
    if 'results' in shard:raise ValueError('Ambiguous score storage')
    d=shard['result_stream'];name=d['file']
    if (d['storage']!='ffn-score-jsonl/v1' or Path(name).name!=name
        or type(d['count']) is not int or d['count']<0):raise ValueError('Invalid score stream descriptor')
    data=path.parent/name
    if data.is_symlink() or sha256(data)!=d['sha256']:raise ValueError('Score stream identity mismatch')
    shard['results']=ScoreRows(data,d['count'])
    return shard
