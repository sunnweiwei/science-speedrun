"""Atomically publish a validated data-only package; retain the old directory."""
import argparse
import ctypes
import os
from pathlib import Path
from model_package import verify


def exchange(source,target):
    """Same-filesystem Linux atomic exchange; no unsafe delete/copy fallback."""
    source=Path(source);target=Path(target)
    if source.is_symlink() or target.is_symlink() or not source.is_dir():
        raise ValueError('Regular package directories required')
    source=source.resolve();target=target.absolute()
    if source==target or source in target.parents or target in source.parents:
        raise ValueError('Independent staging and publication directories required')
    if not target.exists():os.rename(source,target);return
    if not target.is_dir():raise ValueError('Publication target is not a directory')
    libc=ctypes.CDLL(None,use_errno=True)
    fn=libc.renameat2
    fn.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    fn.restype=ctypes.c_int
    if fn(-100,os.fsencode(source),-100,os.fsencode(target),2):
        error=ctypes.get_errno()
        raise OSError(error,'Atomic publication unavailable; both directories retained',str(target))


def publish(source,target):
    check=verify(source)
    if 'ffn_ensemble' in check or check['search_threshold']!=.8:
        raise ValueError('Single-checkpoint protocol required')
    exchange(source,target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged',type=Path,required=True);p.add_argument('--submission',type=Path,default=Path('/workspace/submission'))
    a=p.parse_args();publish(a.staged,a.submission)
