"""Generic strict reader for the twelve benchmark source bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
import urllib.request

import numpy as np

from .errors import ContractError
from .protocol import BUNDLES, INPUT_BP, TARGET_BP
from .wire import example_features, tensor_view


@dataclass
class SourceRecord:
    chromosome: str
    start: int
    end: int
    tensors: dict[str, np.ndarray]
    dna: np.ndarray | None
    dna_serialized_sha256: str
    feature_names: tuple[str, ...]
    source_record_index: int | None = None

    @property
    def coordinate(self) -> tuple[str, int, int]:
        return self.chromosome, self.start, self.end


def _tf_dtype(tf, name: str):
    return {
        "bfloat16": tf.bfloat16,
        "float16": tf.float16,
        "float32": tf.float32,
        "bool": tf.bool,
        "int32": tf.int32,
    }[name]


def _view_tensor(tf, record: bytes, span: tuple[int, int], dtype_name: str) -> np.ndarray:
    """A NumPy view of a serialized TensorProto inside the record (see wire.py)."""
    dtype = _tf_dtype(tf, dtype_name)
    enum, shape, content = tensor_view(record, span)
    if enum != dtype.as_datatype_enum:
        raise ContractError(f"TensorProto dtype {enum} differs from {dtype_name}")
    if content is None or len(content) == 0:
        # Tiny arrays may use repeated scalar fields instead of tensor_content.
        return _decoded_tensor(tf, bytes(memoryview(record)[span[0]:span[1]]), dtype_name)
    result = np.frombuffer(content, dtype=dtype.as_numpy_dtype)
    if result.size != int(np.prod(shape, dtype=np.int64)):
        raise ContractError(f"TensorProto content length differs from shape {shape}")
    return result.reshape(shape)


def _decoded_tensor(tf, serialized: bytes, dtype_name: str) -> np.ndarray:
    """Read serialize_tensor output without constructing a second full TF tensor."""
    from tensorflow.core.framework import tensor_pb2

    proto = tensor_pb2.TensorProto.FromString(serialized)
    dtype = _tf_dtype(tf, dtype_name)
    if proto.dtype != dtype.as_datatype_enum:
        raise ContractError(f"TensorProto dtype {proto.dtype} differs from {dtype_name}")
    shape = tuple(int(dim.size) for dim in proto.tensor_shape.dim)
    if proto.tensor_content:
        result = np.frombuffer(proto.tensor_content, dtype=dtype.as_numpy_dtype)
        if result.size != int(np.prod(shape, dtype=np.int64)):
            raise ContractError(f"TensorProto content length differs from shape {shape}")
        return result.reshape(shape)
    # Some TensorProto encoders use repeated scalar fields for tiny arrays.
    return np.asarray(tf.make_ndarray(proto))


_GCS_HTTPS = "https://storage.googleapis.com/"
_READ_BUFFER = 64 * 1024 * 1024


def _open_gzip_stream(uri: str):
    """A sequential gzip stream over a local file or a public gs:// object.

    TensorFlow's GCS reader sustained about 6.5 MB/s per process here and stalled under
    many processes; one HTTPS stream reaches about 140 MB/s. ISA-L's igzip decompresses several
    times faster than zlib, and the gzip trailer CRC still guards every byte.
    """
    try:
        from isal import igzip as gzip_module
    except ImportError:  # pragma: no cover - the materializer image installs isal
        import gzip as gzip_module
    if uri.startswith("gs://"):
        raw = urllib.request.urlopen(_GCS_HTTPS + uri[len("gs://"):], timeout=600)
    else:
        raw = open(uri, "rb", buffering=_READ_BUFFER)
    return gzip_module.GzipFile(fileobj=raw, mode="rb"), raw


def _tfrecord_payloads(uri: str, attempts: int = 6):
    """Yield TFRecord payloads: little-endian u64 length, u32 length CRC, data, u32 data CRC.

    A remote stream that breaks (connection reset, truncated gzip) is reopened from the start
    and the records already yielded are skipped, so one dropped connection among hundreds of
    concurrent multi-GB downloads does not abort the build.
    """
    import time
    import urllib.error
    import zlib

    yielded = 0
    for attempt in range(attempts):
        stream = raw = None
        try:
            stream, raw = _open_gzip_stream(uri)
            index = 0
            while True:
                header = stream.read(12)
                if not header:
                    return
                if len(header) != 12:
                    raise EOFError(f"Truncated TFRecord header in {uri}")
                (length,) = struct.unpack("<Q", header[:8])
                data = stream.read(length)
                if len(data) != length or len(stream.read(4)) != 4:
                    raise EOFError(f"Truncated TFRecord payload in {uri}")
                if index >= yielded:
                    yielded += 1
                    yield data
                index += 1
        except (EOFError, OSError, zlib.error, urllib.error.URLError) as error:
            if attempt == attempts - 1 or not uri.startswith("gs://"):
                raise ContractError(f"Could not read {uri} after {attempt + 1} attempts: {error}") from error
            time.sleep(min(60, 5 * 2 ** attempt))
        finally:
            if stream is not None:
                stream.close()
            if raw is not None:
                raw.close()


def iter_records(
    uri: str,
    bundle: str,
    *,
    decode_dna: bool = False,
    record_indices: set[int] | None = None,
    compact_target: bool = False,
):
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required to read AlphaGenome TFRecords") from exc
    try:
        spec = BUNDLES[bundle]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark bundle {bundle!r}") from exc
    required = {
        "dna_sequence",
        "interval/chromosome",
        "interval/start",
        "interval/end",
        *(name for name, _ in spec.tensors),
    }
    for record_index, serialized in enumerate(_tfrecord_payloads(uri)):
        # Filtering before Example parsing and parse_tensor is essential for the
        # 512x512x734 junction bundle. Gzip is still streamed sequentially, but
        # unselected tensor payloads never become dense NumPy arrays.
        if record_indices is not None and record_index not in record_indices:
            continue
        features = example_features(serialized)
        names = set(features)
        if names != required:
            raise ContractError(
                f"Unexpected {bundle} fields at {uri}: "
                f"found={sorted(names)}, expected={sorted(required)}"
            )

        def scalar_bytes(name):
            kind, span = features[name]
            return memoryview(serialized)[span[0]:span[1]]

        chromosome = bytes(scalar_bytes("interval/chromosome")).decode("utf-8")
        start = int(features["interval/start"][1][0])
        end = int(features["interval/end"][1][0])
        source_bp = end - start
        tensors = {}
        for name, dtype in spec.tensors:
            tensor = _view_tensor(tf, serialized, features[name][1], dtype)
            was_cropped = False
            if compact_target:
                if bundle == "CONTACT_MAPS":
                    source_bins = source_bp // 2_048
                    if tensor.shape[0] == source_bins and source_bins != 512:
                        offset = (source_bp - INPUT_BP) // 4_096
                        tensor = tensor[offset:offset + 512, offset:offset + 512]
                        was_cropped = True
                elif bundle in {"ATAC", "DNASE", "PROCAP", "CAGE", "RNA_SEQ",
                                "CHIP_TF", "CHIP_HISTONE", "SPLICE_SITES",
                                "SPLICE_SITE_USAGE"} and name != "rna_seq_strand":
                    resolution = spec.native_resolution
                    source_bins = source_bp // resolution
                    target_bins = TARGET_BP // resolution
                    if tensor.shape[0] == source_bins and source_bins != target_bins:
                        offset = (source_bp - TARGET_BP) // (2 * resolution)
                        tensor = tensor[offset:offset + target_bins]
                        was_cropped = True
            # Views point into the decompressed record; copy what is kept so the record can go.
            tensors[name] = np.array(tensor, copy=True)
            del tensor
        dna_bytes = scalar_bytes("dna_sequence")
        dna = None
        if decode_dna:
            dna = np.asarray(
                tf.io.parse_tensor(bytes(dna_bytes), out_type=tf.float32), dtype=np.float32
            )
        dna_hash = hashlib.sha256(dna_bytes).hexdigest()
        del features, serialized, scalar_bytes, dna_bytes
        yield SourceRecord(
            chromosome=chromosome,
            start=start,
            end=end,
            tensors=tensors,
            dna=dna,
            dna_serialized_sha256=dna_hash,
            feature_names=tuple(sorted(names)),
            source_record_index=record_index,
        )
