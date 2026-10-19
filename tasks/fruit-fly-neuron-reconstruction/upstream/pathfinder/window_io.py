"""Lossless, size-bounded storage of unsigned FFN prediction windows."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import zipfile
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import external,sha256


def reserved_window_bytes(shape,compressed_limit=0):
    if type(compressed_limit) is not int or compressed_limit<0:raise ValueError('Nonnegative compressed window limit required')
    raw=int(np.prod(shape))*4+128
    return min(compressed_limit,raw+raw//100+65536) if compressed_limit else raw


class LimitedFile:
    def __init__(self,stream,limit):self.stream=stream;self.limit=limit
    def write(self,data):
        if self.stream.tell()+len(data)>self.limit:raise ValueError('Compressed window exceeds reserved bytes')
        return self.stream.write(data)
    def seek(self,*args):return self.stream.seek(*args)
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()


class HashedWriter:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256();self.count=0
    def write(self,data):
        self.digest.update(data);self.count+=len(data);return self.stream.write(data)
    def flush(self):self.stream.flush()


def save_compressed(array,path,limit,*,compression_level=1):
    path=external(path)
    if path.exists() or type(limit) is not int or limit<1:raise ValueError('New output and positive byte limit required')
    if type(compression_level) is not int or not 0<=compression_level<=9:raise ValueError('Deflate compression level must be an integer from zero through nine')
    if array.ndim!=3 or array.dtype!=np.uint32 or not array.flags.c_contiguous:raise ValueError('C-order uint32 prediction required')
    temporary=path.with_suffix(path.suffix+'.partial')
    if temporary.exists():raise ValueError('Partial output already exists')
    try:
        with temporary.open('xb') as stream:
            with zipfile.ZipFile(LimitedFile(stream,limit),'w',compression=zipfile.ZIP_DEFLATED,compresslevel=compression_level) as archive:
                with archive.open('segmentation.npy','w',force_zip64=True) as member:
                    writer=HashedWriter(member);np.lib.format.write_array(writer,array,allow_pickle=False)
        record=dict(storage='npz',path=str(temporary),sha256=writer.digest.hexdigest(),uncompressed_bytes=writer.count,
                    archive_sha256=sha256(temporary),shape=list(array.shape),reserved_bytes=limit,compression_level=compression_level)
        # Verify decompressed NPY bytes before publishing the compressed artifact.
        verify_compressed(record)
        temporary.replace(path);record['path']=str(path);return record
    except BaseException:
        temporary.unlink(missing_ok=True);raise


@contextmanager
def compressed_source(record):
    """Yield validated NPY header and payload stream without materializing labels."""
    path=Path(record['path'])
    if record['storage']!='npz' or sha256(path)!=record['archive_sha256']:raise ValueError('Compressed window integrity mismatch')
    if type(record.get('reserved_bytes')) is not int or path.stat().st_size>record['reserved_bytes']:
        raise ValueError('Compressed window exceeds its reservation')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()!=['segmentation.npy']:raise ValueError('Single NPY member required')
        info=archive.getinfo('segmentation.npy');shape=record['shape']
        if len(shape)!=3 or any(type(n) is not int or n<1 for n in shape):raise ValueError('Positive 3D shape required')
        import math
        payload=math.prod(shape)*4
        if info.file_size!=record['uncompressed_bytes'] or not payload<info.file_size<=payload+65536:raise ValueError('Unexpected decompressed length')
        with archive.open(info) as source:
            header=source.read(info.file_size-payload)
            import io
            header_stream=io.BytesIO(header);version=np.lib.format.read_magic(header_stream)
            if version==(1,0):actual,fortran,dtype=np.lib.format.read_array_header_1_0(header_stream)
            elif version==(2,0):actual,fortran,dtype=np.lib.format.read_array_header_2_0(header_stream)
            else:raise ValueError('Unsupported NPY version')
            if list(actual)!=shape or fortran or dtype!=np.uint32 or header_stream.tell()!=len(header):raise ValueError('Window header mismatch')
            yield source,header,payload


def verify_compressed(record,*,chunk_bytes=8*1024*1024):
    """Verify exact original NPY bytes with bounded decompression memory."""
    if type(chunk_bytes) is not int or not 1<=chunk_bytes<=64*1024*1024:raise ValueError('Bounded positive chunk size required')
    with compressed_source(record) as (source,header,payload):
        digest=hashlib.sha256(header);remaining=payload
        while remaining:
            chunk=source.read(min(chunk_bytes,remaining))
            if not chunk:raise ValueError('Truncated window data')
            digest.update(chunk);remaining-=len(chunk)
        if source.read(1):raise ValueError('Extra window data')
        if digest.hexdigest()!=record['sha256']:raise ValueError('Decompressed window hash mismatch')
    return dict(sha256=digest.hexdigest(),uncompressed_bytes=len(header)+payload,shape=record['shape'])


@contextmanager
def window_array(record):
    path=Path(record['path'])
    if record.get('storage','npy')=='npy':
        if sha256(path)!=record['sha256']:raise ValueError('Window hash mismatch')
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        try:yield array
        finally:array._mmap.close()
        return
    with compressed_source(record) as (source,header,payload):
        data=source.read(payload)
        if len(data)!=payload or source.read(1):raise ValueError('Truncated or extra window data')
        digest=hashlib.sha256(header);digest.update(data)
        if digest.hexdigest()!=record['sha256']:raise ValueError('Decompressed window hash mismatch')
    yield np.frombuffer(data,dtype=np.uint32).reshape(record['shape'])
