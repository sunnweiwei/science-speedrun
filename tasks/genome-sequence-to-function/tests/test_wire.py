"""The zero-copy wire reader matches protobuf, including negative int64 coordinates."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from agbench.wire import example_features, tensor_view  # noqa: E402


def test_example_and_tensor_views_match_protobuf():
    tensor = tf.io.serialize_tensor(tf.constant(np.arange(24, dtype=np.float32).reshape(6, 4))).numpy()
    example = tf.train.Example(features=tf.train.Features(feature={
        "interval/chromosome": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"chr9"])),
        "interval/start": tf.train.Feature(int64_list=tf.train.Int64List(value=[-122994])),
        "interval/end": tf.train.Feature(int64_list=tf.train.Int64List(value=[929678])),
        "atac": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor])),
    })).SerializeToString()
    features = example_features(example)
    assert features["interval/start"] == ("int64", [-122994])
    assert features["interval/end"] == ("int64", [929678])
    kind, span = features["interval/chromosome"]
    assert bytes(memoryview(example)[span[0]:span[1]]) == b"chr9"
    dtype, shape, content = tensor_view(example, features["atac"][1])
    assert dtype == tf.float32.as_datatype_enum and shape == (6, 4)
    np.testing.assert_array_equal(np.frombuffer(content, np.float32).reshape(shape),
                                  np.arange(24, dtype=np.float32).reshape(6, 4))
