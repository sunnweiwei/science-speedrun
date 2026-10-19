"""Publish complete JSON under a hard byte cap; never truncate a result."""
import json
from pathlib import Path


def write_bounded(path,value,limit):
    path=Path(path)
    if type(limit) is not int or limit<1 or path.exists():raise ValueError('New output and positive JSON byte cap required')
    temporary=path.with_suffix(path.suffix+'.tmp')
    # Exclusive open precedes cleanup ownership: never delete another writer's file.
    stream=temporary.open('xb')
    count=0
    try:
        with stream:
            def write(chunk):
                nonlocal count
                data=chunk.encode('utf-8')
                if count+len(data)>limit:raise ValueError('Complete JSON exceeds declared byte cap')
                stream.write(data);count+=len(data)
            for chunk in json.JSONEncoder(indent=2,allow_nan=False).iterencode(value):write(chunk)
            write('\n')
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count
