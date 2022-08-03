"""
Test module for LabelEncoder
"""
# pylint: disable=missing-function-docstring
import numpy as np

from condor_tensorflow.labelencoder import CondorOrdinalEncoder


def test_labelencoder_basic() -> None:
    num_classes = 5
    labels = np.arange(num_classes)
    enc_labs = CondorOrdinalEncoder(nclasses=num_classes).fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced1() -> None:
    labels = np.array(['a', 'b', 'c', 'd', 'e'])
    enc_labs = CondorOrdinalEncoder().fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced2() -> None:
    labels = ['a', 'b', 'c', 'd', 'e']
    enc_labs = CondorOrdinalEncoder().fit_transform(labels)
    expected = np.array([[0., 0., 0., 0.],
                         [1., 0., 0., 0.],
                         [1., 1., 0., 0.],
                         [1., 1., 1., 0.],
                         [1., 1., 1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)


def test_labelencoder_advanced3() -> None:
    labels = ['low', 'med', 'high']
    enc = CondorOrdinalEncoder(categories=[['low', 'med', 'high']])
    enc_labs = enc.fit_transform(labels)
    expected = np.array([[0., 0.],
                         [1., 0.],
                         [1., 1.]])
    np.testing.assert_allclose(enc_labs, expected)
