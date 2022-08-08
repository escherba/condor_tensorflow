"""
Unit tests for ordinal loss module
"""
# pylint: disable=missing-function-docstring
import pytest
import numpy as np
import tensorflow as tf

from condor_tensorflow.loss import CondorNegLogLikelihood
from condor_tensorflow.loss import CondorOrdinalCrossEntropy
from condor_tensorflow.loss import OrdinalEarthMoversDistance
from condor_tensorflow.utils import encode_ordinal_labels_numpy


def test_dense_condor_nll_mismatch() -> None:
    loss = CondorNegLogLikelihood(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.81326175)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_condor_nll_mismatch() -> None:
    loss = CondorNegLogLikelihood(sparse=True)
    val = loss(tf.constant([2.]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.81326175)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_condor_ce_mismatch() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.4698925)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_condor_ce_mismatch() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.4698925)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_emd_mismatch() -> None:
    loss = OrdinalEarthMoversDistance(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.51148224)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_emd_mismatch() -> None:
    loss = OrdinalEarthMoversDistance(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.51148224)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_condor_nll_match() -> None:
    loss = CondorNegLogLikelihood(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.31326172)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_condor_nll_match() -> None:
    loss = CondorNegLogLikelihood(sparse=True)
    val = loss(tf.constant([2.]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.31326172)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_condor_ce_match() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.46989256)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_condor_ce_match() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.46989256)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_dense_emd_match() -> None:
    loss = OrdinalEarthMoversDistance(sparse=False)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.24483162)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_sparse_emd_match() -> None:
    loss = OrdinalEarthMoversDistance(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.24483162)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "klass",
    [CondorNegLogLikelihood, CondorOrdinalCrossEntropy, OrdinalEarthMoversDistance],
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
    "klass",
    [CondorNegLogLikelihood, CondorOrdinalCrossEntropy, OrdinalEarthMoversDistance],
)
def test_sparse_inequality(klass: type) -> None:
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
        tf.debugging.assert_less_equal(observed1, observed2)
