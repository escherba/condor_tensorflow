"""
Unit tests for ordinal metrics module
"""
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
import pytest
import numpy as np
import tensorflow as tf

from condor_tensorflow.metrics import OrdinalMeanAbsoluteError
from condor_tensorflow.metrics import OrdinalAccuracy
from condor_tensorflow.utils import encode_ordinal_labels_numpy


@pytest.mark.parametrize(
    "klass",
    [OrdinalMeanAbsoluteError, OrdinalAccuracy]
)
def test_sparse_order_invariance(klass: type) -> None:
    """test order invariance (equal after shuffling)"""
    for _ in range(10):
        num_classes = np.random.randint(2, 8)
        loss = klass(sparse=True)
        y_true = np.random.randint(0, num_classes, 20)
        y_pred1 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed1 = loss(y_true, y_pred1)
        np.random.shuffle(y_true)
        y_pred2 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed2 = loss(y_true, y_pred2)
        tf.debugging.assert_near(observed1, observed2)


@pytest.mark.parametrize(
    "klass,condition",
    [(OrdinalMeanAbsoluteError, "le"), (OrdinalAccuracy, "ge")]
)
def test_sparse_inequality1(klass: type, condition: str) -> None:
    """test expected inequality (equal or worse after shuffling)"""
    for _ in range(10):
        num_classes = np.random.randint(2, 8)
        loss = klass(sparse=True)
        y_true = np.random.randint(0, num_classes, 20)
        y_true_orig = y_true.copy()
        y_pred1 = encode_ordinal_labels_numpy(
            y_true_orig, num_classes=num_classes)
        observed1 = loss(y_true_orig, y_pred1)
        np.random.shuffle(y_true)
        y_pred2 = encode_ordinal_labels_numpy(
            y_true, num_classes=num_classes)
        observed2 = loss(y_true_orig, y_pred2)
        if condition == "le":
            tf.debugging.assert_less_equal(observed1, observed2)
        elif condition == "ge":
            tf.debugging.assert_greater_equal(observed1, observed2)
        else:
            assert False


def test_OrdinalMeanAbsoluteError() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalMeanAbsoluteError1() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError1() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalMeanAbsoluteError2() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError2() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([0]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy() -> None:
    loss = OrdinalAccuracy(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy1() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy1() -> None:
    loss = OrdinalAccuracy(sparse=True, tolerance=1)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy2() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy2() -> None:
    loss = OrdinalAccuracy(sparse=True)
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy12() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy12() -> None:
    loss = OrdinalAccuracy(sparse=True, tolerance=1)
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalMeanAbsoluteError3() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 1.], [1., 1.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError3() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([[2], [2]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalMeanAbsoluteError13() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 0.], [1., 0.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError13() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([[1], [1]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalMeanAbsoluteError23() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0., 0.], [0., 0.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalMeanAbsoluteError23() -> None:
    loss = OrdinalMeanAbsoluteError(sparse=True)
    val = loss(tf.constant([[0], [0]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy3() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 1.], [1., 1.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy3() -> None:
    loss = OrdinalAccuracy(sparse=True)
    val = loss(tf.constant([[2], [2]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy13() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 1.], [1., 1.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy13() -> None:
    loss = OrdinalAccuracy(sparse=True, tolerance=1)
    val = loss(tf.constant([[2], [2]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy23() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 0.], [1., 0.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy23() -> None:
    loss = OrdinalAccuracy(sparse=True)
    val = loss(tf.constant([[1], [1]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalAccuracy123() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 0.], [1., 0.]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalAccuracy123() -> None:
    loss = OrdinalAccuracy(sparse=True, tolerance=1)
    val = loss(tf.constant([[1], [1]]),
               tf.constant([[-1., 1.], [-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
