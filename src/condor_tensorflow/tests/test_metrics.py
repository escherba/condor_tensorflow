from condor_tensorflow.metrics import OrdinalMeanAbsoluteError
from condor_tensorflow.metrics import SparseOrdinalMeanAbsoluteError
from condor_tensorflow.metrics import OrdinalAccuracy
from condor_tensorflow.metrics import SparseOrdinalAccuracy
import pytest
import tensorflow as tf


def test_OrdinalMeanAbsoluteError() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalMeanAbsoluteError1() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError1() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalMeanAbsoluteError2() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError2() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([0]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy() -> None:
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy1() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 1.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy1() -> None:
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([2]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy2() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy2() -> None:
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy12() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 0.]]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy12() -> None:
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([1]), tf.constant([[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)





def test_OrdinalMeanAbsoluteError3() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 1.],[1., 1.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError3() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([[2],[2]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(2.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalMeanAbsoluteError13() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1., 0.],[1., 0.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError13() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([[1],[1]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalMeanAbsoluteError23() -> None:
    loss = OrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0., 0.],[0., 0.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalMeanAbsoluteError23() -> None:
    loss = SparseOrdinalMeanAbsoluteError()
    val = loss(tf.constant([[0],[0]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy3() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 1.],[1., 1.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy3() -> None:
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([[2],[2]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy13() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 1.],[1., 1.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy13() -> None:
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[2],[2]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy23() -> None:
    loss = OrdinalAccuracy()
    val = loss(tf.constant([[1., 0.],[1., 0.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy23() -> None:
    loss = SparseOrdinalAccuracy()
    val = loss(tf.constant([[1],[1]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(0.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_OrdinalAccuracy123() -> None:
    loss = OrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1., 0.],[1., 0.]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)

def test_SparseOrdinalAccuracy123() -> None:
    loss = SparseOrdinalAccuracy(tolerance=1)
    val = loss(tf.constant([[1],[1]]),
               tf.constant([[-1., 1.],[-1., 1.]]))
    expect = tf.constant(1.0)
    tf.debugging.assert_near(val, expect, rtol=1e-5, atol=1e-5)
