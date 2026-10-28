"""Controller-side data builders. Original labels never enter private/public inputs."""
import argparse,csv,hashlib,hmac,json,os,pickle,re,secrets,sys,traceback,fcntl
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

SPECS={'faced':(32,10,9),'tuev':(16,5,6),'ad65':(19,10,3),'openbmi':(62,4,2),'longitudinal':(58,2,2)}
def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def subject_split(subjects,labels=None):
    subjects=sorted(subjects)
    ys=None if labels is None else [labels[s] for s in subjects]
    trval,test=train_test_split(subjects,test_size=.2,random_state=3407,stratify=ys)
    ys=None if labels is None else [labels[s] for s in trval]
    train,val=train_test_split(trval,test_size=.25,random_state=3407,stratify=ys)
    splits={k:sorted(v) for k,v in [('train',train),('val',val),('test',test)]}
    assert not (set(train)&set(val) or set(train)&set(test) or set(val)&set(test))
    return splits
class Writer:
    def __init__(self,dest,track,split):
        self.track,self.split=track,split
        records=dest/'records';records.mkdir(parents=True,exist_ok=True)
        keyfile=records/'opaque-id-key'
        # Private key prevents dictionary attacks on class-bearing TUEV filenames.
        with open(records/'opaque-id-key.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not keyfile.exists():
                fd=os.open(keyfile,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(fd,'wb') as keyout:keyout.write(os.urandom(32))
            self.id_key=keyfile.read_bytes()
            if len(self.id_key)!=32:raise ValueError('invalid private ID key')
        self.root=dest/('private' if split=='test' else 'development')/track/('public' if split=='test' else split)
        if (self.root/'manifest.json').exists():raise FileExistsError('already materialized: '+str(self.root))
        self.root.mkdir(parents=True,exist_ok=True)
        if split=='test':
            # The controller and verifier are root; candidate code must not be
            # able to traverse into private inputs or labels on a shared mount.
            (dest/'private').chmod(0o700)
        self.rows=[];self.buffer=[];self.pending=[];self.shards=[];self.ids=set();self.config={}
        self.truth=dest/'private'/track/'truth.jsonl' if split=='test' else None
        if track=='faced':
            self.config={'channels':[f'FACED-source-index-{i}' for i in range(32)],'units':'uV','sampling_rate':200,'shape':[32,10,200],'channel_note':'Preserve official processed array order; named montage not supplied.'}
        elif track=='tuev':
            self.config={'channels':['FP1-F7','F7-T3','T3-T5','T5-O1','FP2-F8','F8-T4','T4-T6','T6-O2','FP1-F3','F3-C3','C3-P3','P3-O1','FP2-F4','F4-C4','C4-P4','P4-O2'],'units':'uV','sampling_rate':200,'shape':[16,5,200]}
    def add(self,x,label,source,subject,session='1',channels=None):
        c,s,k=SPECS[self.track]
        x=np.asarray(x)
        if x.shape!=(c,s,200) or not np.isfinite(x).all():raise ValueError('invalid sample '+source+' '+str(x.shape))
        if not 0<=int(label)<k or int(label)!=label:raise ValueError('invalid label')
        sid=hmac.new(self.id_key,(self.track+'|'+self.split+'|'+source).encode(),hashlib.sha256).hexdigest()
        if sid in self.ids:raise ValueError('duplicate sample '+source)
        self.ids.add(sid)
        if channels:
            if self.config and self.config.get('channels')!=channels:raise ValueError('inconsistent channel order')
            self.config={'channels':channels,'units':'uV','sampling_rate':200,'shape':[c,s,200]}
        self.buffer.append(x)
        self.pending.append({'id':sid,'label':int(label),'subject':str(subject),'session':str(session),'source':source})
        if self.split!='test' and len(self.buffer)>=128:self.flush()
    def flush(self):
        if not self.buffer:return
        name=f'shard-{len(self.shards):06d}.npy';p=self.root/name
        np.save(p,np.stack(self.buffer),allow_pickle=False)
        h=hashlib.sha256(p.read_bytes()).hexdigest();self.shards.append({'path':name,'sha256':h,'bytes':p.stat().st_size})
        for i,r in enumerate(self.pending):self.rows.append({**r,'path':name,'index':i})
        self.buffer=[];self.pending=[]
    def finish(self):
        if self.split=='test':
            # Shuffle the entire split BEFORE assigning public shard/row positions.
            # Neither the private random permutation nor its RNG state is published.
            order=list(range(len(self.buffer)))
            secrets.SystemRandom().shuffle(order)
            arrays,pending=self.buffer,self.pending
            self.buffer,self.pending=[],[]
            for i in order:
                self.buffer.append(arrays[i]);self.pending.append(pending[i])
                if len(self.buffer)>=128:self.flush()
        self.flush()
        if not self.rows:raise ValueError('empty split '+self.split)
        if self.truth:
            self.truth.parent.mkdir(parents=True,exist_ok=True)
            self.truth.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
            self.truth.chmod(0o600)
            public=[{k:r[k] for k in ('id','path','index')} for r in self.rows]
        else:public=self.rows
        # Stable opaque-ID ordering decouples interface order from class/subject.
        public=sorted(public,key=lambda r:r['id'])
        write_json(self.root/'manifest.json',{'schema':'bci-input-v1','storage_layout':'global-random-v2' if self.split=='test' else 'development-v1','track':self.track,'config':self.config,'samples':public})
        write_json(self.root/'checksums.json',self.shards)
        return {'samples':len(self.rows),'subjects':len({r['subject'] for r in self.rows}),'labels':dict(Counter(r['label'] for r in self.rows))}
def faced(src,w):
    import lmdb
    db=lmdb.open(str(src),readonly=True,lock=False,readahead=False)
    with db.begin() as txn:
        splits=pickle.loads(txn.get(b'__keys__'))
        seen={}
        for split in ('train','val','test'):
            for key in splits[split]:
                key=key.decode() if isinstance(key,bytes) else key
                pair=pickle.loads(txn.get(key.encode()))
                subject=key.rsplit('-',2)[0]
                if subject in seen and seen[subject]!=split:raise ValueError('FACED subject leakage')
                seen[subject]=split
                w[split].add(pair['sample'],int(pair['label']),key,subject)
    db.close()
    return {k:sorted(s for s,v in seen.items() if v==k) for k in w}
TUEV_DEVELOPMENT_SHA256 = '96f5f65c05527be1626d858917e758e375debd90e5061082038c8e8066e0e7bb'
TUEV_SUBJECTS_SHA256 = 'baf3ed5bd371dffbd57b2ab15541d6d4a88c27a99d007b6e3d37a091dcaafd81'

def validate_tuev_layout(layout):
    supported={(66050,17882,29421),(68445,15487,29421),(83932,0,29421)}
    if tuple(layout) not in supported:raise ValueError('unsupported TUEV source layout '+str(layout))

def tuev_plan(src):
    paths={}
    for folder in ('processed_train','processed_eval','processed_test'):
        paths[folder]=sorted((src/folder).glob('*.pkl'))
    layout=tuple(len(paths[k]) for k in ('processed_train','processed_eval','processed_test'))
    validate_tuev_layout(layout)
    development=paths['processed_train']+paths['processed_eval']
    if len({p.name for p in development})!=len(development):raise ValueError('duplicate development filenames')
    union_digest=hashlib.sha256('\n'.join(sorted(p.name for p in development)).encode()).hexdigest()
    if union_digest!=TUEV_DEVELOPMENT_SHA256:
        raise ValueError('TUEV development union differs from frozen source inventory')
    subjects=sorted({p.name.split('_')[0] for p in development})
    if len(subjects)!=290:raise ValueError('expected 290 TUEV development subjects')
    subject_digest=hashlib.sha256('\n'.join(subjects).encode()).hexdigest()
    if subject_digest!=TUEV_SUBJECTS_SHA256:
        raise ValueError('TUEV development subject membership differs from frozen source')
    train=set(subjects[:int(len(subjects)*.8)])
    # The provider defines the patient-independent train/evaluation split.
    # official-test is a partition marker, not a patient identifier.
    subsets={'train':sorted(train),'val':sorted(set(subjects)-train),'test':['official-test']}
    planned=[]
    for p in sorted(development,key=lambda p:p.name)+paths['processed_test']:
        subject=p.name.split('_')[0]
        split='test' if p.parent.name=='processed_test' else ('train' if subject in train else 'val')
        if split=='test':subject='official-test'
        planned.append((p,split,subject))
    if Counter(split for _,split,_ in planned)!={'train':68445,'val':15487,'test':29421}:
        raise ValueError('current official TUEV split mismatch')
    return planned,subsets

def tuev(src,w):
    planned,subsets=tuev_plan(src)
    counts=Counter()
    for p,split,subject in planned:
        pair=pickle.loads(p.read_bytes());x=np.asarray(pair['signal']).reshape(16,5,200)
        label=int(np.asarray(pair['label']).reshape(-1)[0])-1
        w[split].add(x,label,str(p.relative_to(src)),subject)
        counts[split]+=1
    if dict(counts)!={'train':68445,'val':15487,'test':29421}:raise ValueError('current official TUEV split mismatch '+str(counts))
    return subsets
def epochs(raw,seconds,filter_signal):
    import mne
    # EEGLAB boundary annotations must reject windows, not join discontinuous time.
    descriptions=[('BAD_boundary' if 'boundary' in d.lower() else d) for d in raw.annotations.description]
    raw.annotations.description=np.array(descriptions)
    if filter_signal:raw.filter(.5,45,skip_by_annotation=('edge','bad'),verbose='ERROR')
    raw.resample(200,verbose='ERROR')
    ep=mne.make_fixed_length_epochs(raw,duration=seconds,preload=True,reject_by_annotation=True,verbose='ERROR')
    return ep.get_data(units='uV').astype(np.float32),{'candidate_windows':int(raw.n_times//(seconds*200)),'kept_windows':len(ep),'window_start_samples_200hz':ep.events[:,0].tolist(),'tail_samples':int(raw.n_times%(seconds*200))}
def ad65(src,w):
    import mne
    table=list(csv.DictReader(open(src/'participants.tsv'),delimiter='\t'))
    mapping={'A':0,'F':1,'C':2};labels={r['participant_id']:mapping[r['Group']] for r in table}
    if len(labels)!=88:raise ValueError('AD expected 88 subjects')
    splits=subject_split(labels,labels);stats={}
    for split,subs in splits.items():
        for sub in subs:
            p=src/'derivatives'/sub/'eeg'/f'{sub}_task-eyesclosed_eeg.set'
            raw=mne.io.read_raw_eeglab(p,preload=True,verbose='ERROR')
            raw.pick('eeg')
            if len(raw.ch_names)!=19:raise ValueError('AD channel count')
            x,st=epochs(raw,10,False);stats[sub]=st
            for i,a in enumerate(x):w[split].add(a.reshape(19,10,200),labels[sub],sub+':'+str(i),sub,channels=raw.ch_names)
    return splits,stats
def scalar(v):return np.asarray(v).squeeze().item()
def strings(v):
    result=[]
    for x in np.asarray(v).ravel():
        while isinstance(x,np.ndarray) and x.size==1:x=x.item()
        result.append(str(x))
    return result
def openbmi(src,w):
    import mne
    from scipy.io import loadmat
    splits=subject_split([f'{i:02d}' for i in range(1,55)]);stats={}
    for split,subs in splits.items():
        for sub in subs:
            for ses in (1,2):
                p=src/f'session{ses}'/f's{int(sub)}'/f'sess{ses:02d}_subj{int(sub):02d}_EEG_MI.mat'
                mat=loadmat(p)
                for run in ('train','test'):
                    d=mat['EEG_MI_'+run][0,0]
                    x=np.asarray(d['x'],dtype=float);names=strings(d['chan']);fs=float(scalar(d['fs']))
                    if x.ndim!=2 or x.shape[1]!=62 or len(names)!=62:raise ValueError('OpenBMI channels '+str(p))
                    labels=np.asarray(d['y_dec']).reshape(-1).astype(int);onsets=np.asarray(d['t']).reshape(-1).astype(int)
                    if len(labels)!=len(onsets) or set(labels)!={1,2}:raise ValueError('OpenBMI labels missing')
                    raw=mne.io.RawArray(x.T*1e-6,mne.create_info(names,fs,'eeg'),verbose='ERROR')
                    raw.filter(.5,45,verbose='ERROR');raw.resample(200,verbose='ERROR');arr=raw.get_data(units='uV')
                    for i,(t,label) in enumerate(zip(onsets,labels)):
                        # MATLAB sample indices are 1-based; preserve event alignment.
                        start=int(round((int(t)-1)*200/fs));a=arr[:,start:start+800]
                        if start<0 or a.shape!=(62,800):raise ValueError('trial outside recording')
                        w[split].add(a.astype(np.float32).reshape(62,4,200),{2:0,1:1}[int(label)],f'{sub}:{ses}:{run}:{i}',sub,str(ses),names)
                    stats[f'{sub}:{ses}:{run}']=len(labels)
    return splits,stats
def longitudinal(src,w):
    import mne
    paths=sorted(src.glob('sub-*/ses-*/eeg/*_eeg.vhdr'))
    subjects=sorted({p.parts[-4] for p in paths});splits=subject_split(subjects)
    # Both conditions must exist in every included subject/session.
    conditions=defaultdict(set)
    for p in paths:
        sub,ses=p.parts[-4:-2];task=re.search(r'_task-(CE|OE)_',p.name)
        if not task:raise ValueError('unknown condition '+str(p))
        conditions[(sub,ses)].add(task[1])
    if any(v!={'CE','OE'} for v in conditions.values()):raise ValueError('incomplete session conditions')
    assignment={s:k for k,v in splits.items() for s in v};stats={}
    for p in paths:
        sub,ses=p.parts[-4:-2];task=re.search(r'_task-(CE|OE)_',p.name)[1]
        side=p.with_name(p.name.replace('_eeg.vhdr','_channels.tsv'))
        channels=[r['name'] for r in csv.DictReader(open(side),delimiter='\t') if r['type'].upper()=='EEG']
        if len(channels)!=58:raise ValueError('Longitudinal expected 58 EEG channels, got '+str(len(channels)))
        raw=mne.io.read_raw_brainvision(p,preload=True,verbose='ERROR');raw.pick(channels)
        x,st=epochs(raw,2,True);stats[str(p.relative_to(src))]=st
        for i,a in enumerate(x):w[assignment[sub]].add(a.reshape(58,2,200),{'CE':0,'OE':1}[task],f'{sub}:{ses}:{task}:{i}',sub,ses,channels)
    return splits,stats
def main():
    ap=argparse.ArgumentParser();ap.add_argument('track',choices=SPECS);ap.add_argument('--source',required=True,type=Path);ap.add_argument('--destination',required=True,type=Path);a=ap.parse_args()
    w={k:Writer(a.destination,a.track,k) for k in ('train','val','test')}
    try:
        result=globals()[a.track](a.source,w)
        splits,stats=result if isinstance(result,tuple) else (result,{})
        summary={k:writer.finish() for k,writer in w.items()}
        write_json(a.destination/'records'/a.track/'subjects.json',splits)
        write_json(a.destination/'records'/a.track/'preprocessing.json',stats)
        write_json(a.destination/'records'/a.track/'receipt.json',{'status':'complete','track':a.track,'splits':summary,'seed':3407,'source':str(a.source)})
        print(json.dumps(summary),flush=True)
    except Exception:
        p=a.destination/'records'/a.track/'ERROR.txt';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(traceback.format_exc());raise
if __name__=='__main__':main()
