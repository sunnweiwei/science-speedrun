"""Hashing and atomic-write helpers shared by the inference supervisor and the grader.

The physical drivers and scorers live in environment/evaluation (copied to /evaluation), so the
agent's validation runs and the grader use the same code.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(8*1024*1024): digest.update(chunk)
    return digest.hexdigest()

def tree_sha256(path:Path)->str:
    digest=hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file()):
        digest.update(str(item.relative_to(path)).encode()); digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()

def write_json(path:Path,value:object)->None:
    temporary=path.with_name(path.name+".tmp"); temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); temporary.replace(path)
