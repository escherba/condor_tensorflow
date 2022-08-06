"""
Unit tests for ordinal loss module
"""
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
import tensorflow as tf

from condor_tensorflow.loss import CondorNegLogLikelihood
from condor_tensorflow.loss import CondorOrdinalCrossEntropy
from condor_tensorflow.loss import OrdinalEarthMoversDistance


def test_CondorNegLogLikelihood() -> None:
    loss = CondorNegLogLikelihood()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCondorNegLogLikelihood() -> None:
    loss = CondorNegLogLikelihood(sparse=True)
    val = loss(tf.constant([2.]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_CondorOrdinalCrossEntropy() -> None:
    loss = CondorOrdinalCrossEntropy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCondorOrdinalCrossEntropy() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalEarthMoversDistance() -> None:
    loss = OrdinalEarthMoversDistance()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.5344467)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalEarthMoversDistance() -> None:
    loss = OrdinalEarthMoversDistance(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.5344467)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_CondorNegLogLikelihoodMatch() -> None:
    loss = CondorNegLogLikelihood()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCondorNegLogLikelihoodMatch() -> None:
    loss = CondorNegLogLikelihood(sparse=True)
    val = loss(tf.constant([2.]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.6265235)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_CondorOrdinalCrossEntropyMatch() -> None:
    loss = CondorOrdinalCrossEntropy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseCondorOrdinalCrossEntropyMatch() -> None:
    loss = CondorOrdinalCrossEntropy(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.9397845)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_OrdinalEarthMoversDistanceMatch() -> None:
    loss = OrdinalEarthMoversDistance()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.73449475)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)


def test_SparseOrdinalEarthMoversDistanceMatch() -> None:
    loss = OrdinalEarthMoversDistance(sparse=True)
    val = loss(tf.constant([2]), tf.constant([[1., 1.]]))
    expect = tf.constant(0.73449475)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
