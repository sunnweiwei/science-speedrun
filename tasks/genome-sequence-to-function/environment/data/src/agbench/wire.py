"""Zero-copy reader for the parts of tf.train.Example / TensorProto that agbench uses.

A source record is several hundred MB after decompression. Parsing it with protobuf copies it
into an Example, then each tensor again into a TensorProto, before NumPy sees it; with ~100
concurrent workers those copies saturated memory bandwidth (1.55 s per ATAC record under load,
0.14 s with views). Byte-identical results are checked against the protobuf path in
tests/test_wire.py and on real records of every bundle.
"""
import numpy as np

def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if b < 0x80: return result, pos
        shift += 7

def _fields(buf, start, end):
    """Yield (field_number, wire_type, value_or_(start,end)) without copying length-delimited data."""
    pos = start
    while pos < end:
        key, pos = _varint(buf, pos); field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos); yield field, wire, value
        elif wire == 2:
            length, pos = _varint(buf, pos); yield field, wire, (pos, pos + length); pos += length
        elif wire == 5: yield field, wire, (pos, pos + 4); pos += 4
        elif wire == 1: yield field, wire, (pos, pos + 8); pos += 8
        else: raise ValueError(f"unsupported wire type {wire}")

def example_features(buf):
    """name -> ('bytes', (s,e)) | ('int64', [values]) from a serialized tf.train.Example."""
    out = {}
    for f, w, span in _fields(buf, 0, len(buf)):
        if f != 1: continue  # Example.features
        for f2, w2, entry in _fields(buf, *span):
            if f2 != 1: continue  # Features.feature map entry
            key = None; value = None
            for f3, w3, v in _fields(buf, *entry):
                if f3 == 1: key = bytes(buf[v[0]:v[1]]).decode()
                elif f3 == 2: value = v
            for f4, w4, v in _fields(buf, *value):  # Feature oneof
                if f4 == 1:  # BytesList
                    for f5, w5, b in _fields(buf, *v):
                        if f5 == 1: out[key] = ("bytes", b)
                elif f4 == 3:  # Int64List
                    vals = []
                    for f5, w5, b in _fields(buf, *v):
                        if f5 == 1:
                            if w5 == 0: vals.append(b)
                            else:
                                p = b[0]
                                while p < b[1]:
                                    x, p = _varint(buf, p); vals.append(x)
                    # int64 is two's complement on the wire: interval starts can be negative
                    # (windows padded past a chromosome start).
                    out[key] = ("int64", [v - (1 << 64) if v >= (1 << 63) else v for v in vals])
    return out

def tensor_view(buf, span):
    """(dtype_enum, shape, memoryview of tensor_content) of a serialized TensorProto."""
    dtype = None; shape = []; content = None
    for f, w, v in _fields(buf, *span):
        if f == 1: dtype = v
        elif f == 2:
            for f2, w2, dim in _fields(buf, *v):
                if f2 == 2:
                    for f3, w3, size in _fields(buf, *dim):
                        if f3 == 1: shape.append(size)
        elif f == 4: content = memoryview(buf)[v[0]:v[1]]
    return dtype, tuple(shape), content
