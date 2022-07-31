"""
Ordinal label encoder
"""
# pylint: disable=attribute-defined-outside-init
# pylint: disable=unused-argument
from typing import Any, Dict, Optional, Union, Sequence, List

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

from .types import IntArray, ObjectArray


class CondorOrdinalEncoder(BaseEstimator, TransformerMixin):

    """Ordinal label encoder"""

    nclasses: int
    dtype: type
    kwargs: Dict[str, Any]
    feature_names_in_: List[str]

    def __init__(
            self,
            nclasses: int = 0,
            dtype: type = np.int32,
            **kwargs: Any) -> None:
        self.nclasses = nclasses
        self.dtype = dtype
        self.kwargs = kwargs

    def fit(self,
            X: pd.DataFrame,
            y: Optional[ObjectArray] = None) -> 'CondorOrdinalEncoder':
        """Fit the CondorOrdinalEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.
        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`~sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
        """
        if hasattr(X, "columns"):
            # pandas dataframes
            self.feature_names_in_ = X.columns.tolist()
        elif hasattr(X, "iloc"):
            # pandas series
            self.feature_names_in_ = [X.name]
        elif hasattr(X, "shape") or isinstance(X, list):
            # numpy array
            self.feature_names_in_ = ["X"]

        if self.nclasses > 0:
            pass  # expecting 0,1,...,nclasses-1
        else:
            self._enc = OrdinalEncoder(dtype=self.dtype, **self.kwargs)
            if isinstance(X, list):
                X = np.array(X)
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self._enc.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> IntArray:
        """Transform X to ordinal arrays.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            The labels data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_classes-1)
            Transformed input.
        """
        if isinstance(X, list):
            X = np.array(X)
        if self.nclasses == 0:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
                X = np.array(self._enc.transform(X)[:, 0],
                             dtype=self.dtype)
            self.nclasses_converted = len(self._enc.categories_[0])
        else:
            self.nclasses_converted = self.nclasses
            X = np.array(X, dtype=self.dtype)

        # now X always has values 0,1,...,nclasses-1
        # first make one-hot encoding
        output = np.zeros((X.shape[0], self.nclasses_converted))
        output[np.arange(X.size), X] = 1

        # now drop first column
        output = output[:, 1:]

        # and use cumsum to fill
        return np.flip(np.flip(output, 1).cumsum(axis=1), 1)

    def get_feature_names_out(
            self,
            input_features: Union[None, str, Sequence[str]] = None
            ) -> ObjectArray:
        """feature names transformation.
        Parameters
        ----------
        input_features : str, a list of str of the same length as features fitted, or None.

        If input_features is None, then feature_names_in_ is used as feature
        names in. If feature_names_in_ is not defined, then the following input
        feature names are generated: ["x1", "x2", ..., "x(nclasses - 1)"].  If
        input_features is an array-like, then input_features must match
        feature_names_in_ if feature_names_in_ is defined.

        Returns
        -------
        ndarray of shape (n_classes-1) transformed feature names.
        """
        if isinstance(input_features, str):
            input_features = [input_features for _ in self.feature_names_in_]
        if input_features is None:
            input_features = self.feature_names_in_

        assert len(input_features) == len(self.feature_names_in_), \
            'length of input features not equal to length of fitted features'

        return np.array([
            featurename + str(level)
            for level in range(1, self.nclasses)
            for featurename in input_features
        ], dtype=object)
