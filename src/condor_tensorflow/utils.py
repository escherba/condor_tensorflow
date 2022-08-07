"""
Utilities module
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes

from .types import IntArray, FloatArray


def encode_ordinal_labels_numpy(
        array: IntArray,
        num_classes: int,
        dtype: type = np.float32) -> FloatArray:
    """Encoder ordinal data to one-hot type

    Example:

        >>> labels = np.arange(3)
        >>> encode_ordinal_labels_numpy(labels, num_classes=3)
        array([[0., 0.],
               [1., 0.],
               [1., 1.]], dtype=float32)
    """
    compare_to = np.arange(num_classes)
    mask = array[:, None] >= compare_to
    return mask[:, 1:].astype(dtype)


def encode_ordinal_labels_v1(
        labels: tf.Tensor,
        num_classes: int,
        dtype: dtypes.DType = tf.float32) -> tf.Tensor:
    """Convert ordinal label to one-host representation

    Args:
        labels (tf.Tensor): a tensor of ordinal labels (starting with zero)
        num_classes (int): assumed number of classes
        dtype (dtypes.DType): result data type

    Returns:
        tf.Tensor: a tensor of levels (one-hot-encoded labels)

    Example:

        >>> labels = tf.constant([0, 1, 2], dtype=tf.float32)
        >>> encode_ordinal_labels_v1(labels, num_classes=3)
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
        array([[0., 0.],
               [1., 0.],
               [1., 1.]], dtype=float32)>

    Calling this is equivalent to:

        levels = [1] * label + [0] * (num_classes - 1 - label)
    """
    def _func(label: tf.Tensor) -> tf.Tensor:
        # Original code that we are trying to replicate:
        # levels = [1] * label + [0] * (num_classes - 1 - label)
        label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))

        # This line requires that label values begin at 0. If they start at a higher
        # value it will yield an error.
        num_zeros = num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)
        zero_vec = tf.zeros(shape=num_zeros, dtype=tf.int32)
        return tf.cast(tf.concat([label_vec, zero_vec], 0), dtype)
    labels = tf.cast(labels, tf.float32)
    return tf.map_fn(_func, labels)


def encode_ordinal_labels_v2(
        labels: tf.Tensor,
        num_classes: int,
        dtype: dtypes.DType = tf.float32) -> tf.Tensor:
    """Convert ordinal label to one-hot representation

    Args:
        labels (tf.Tensor): a tensor of ordinal labels (starting with zero)
        num_classes (int): assumed number of classes
        dtype (dtypes.DType): result data type

    Returns:
        tf.Tensor: a tensor of levels (one-hot-encoded labels)

    Example:

        >>> labels = tf.constant([0, 1, 2], dtype=tf.float32)
        >>> encode_ordinal_labels_v2(labels, num_classes=3)
        <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
        array([[0., 0.],
               [1., 0.],
               [1., 1.]], dtype=float32)>

    Calling this is equivalent to:

        levels = [1] * label + [0] * (num_classes - 1 - label)
    """
    # This function uses tf.sequence_mask(), which is vectorized, and avoids
    # map_fn() call.
    return tf.sequence_mask(labels, maxlen=num_classes - 1, dtype=dtype)


encode_ordinal_labels = encode_ordinal_labels_v2
