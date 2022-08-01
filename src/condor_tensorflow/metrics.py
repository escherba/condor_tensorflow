"""
Ordinal metrics module
"""
from typing import Any, Optional, Dict
import tensorflow as tf
from tensorflow.keras import metrics


class OrdinalMeanAbsoluteError(metrics.Metric):
    """Computes mean absolute error for ordinal labels."""

    sparse: bool

    def __init__(
            self,
            sparse: bool = False,
            name: str = "mean_absolute_error_labels",
            **kwargs: Any):
        """Creates a `OrdinalMeanAbsoluteError` instance."""
        super().__init__(name=name, **kwargs)
        self.sparse = sparse
        self.maes = self.add_weight(name='maes', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> None:
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """
        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        if not self.sparse:
            # Sum across columns to estimate how many cumulative thresholds are
            # passed.
            y_true = tf.reduce_sum(y_true, axis=1)

        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        if sample_weight is not None:
            values = tf.abs(y_true - labels_v2)
            sample_weight = tf.cast(tf.squeeze(sample_weight), y_pred.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            self.maes.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.maes.assign_add(tf.reduce_sum(tf.abs(y_true - labels_v2)))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.maes, self.count)

    def reset_state(self) -> None:
        """Resets all of the metric state variables at the start of each epoch."""
        self.maes.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            'sparse': self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}


class OrdinalAccuracy(metrics.Metric):
    """Computes accuracy for ordinal labels (tolerance is allowed rank
    distance to be considered 'correct' predictions)."""

    sparse: bool

    def __init__(
            self,
            sparse: bool = False,
            name: Optional[str] = None,
            tolerance: float = 0.,
            **kwargs: Any) -> None:
        """Creates a `OrdinalAccuracy` instance."""
        if name is None:
            name = f"ordinal_accuracy_tol{tolerance}"
        super().__init__(name=name, **kwargs)
        self.sparse = sparse
        self.accs = self.add_weight(name='accs', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.tolerance = tolerance

    def update_state(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight: Optional[tf.Tensor] = None) -> None:
        """Computes accuracy for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
          sample_weight (optional): Not implemented.
        """

        # Predict the label as in Cao et al. - using cumulative probabilities
        cum_probs = tf.math.cumprod(
            tf.math.sigmoid(y_pred),
            axis=1)  # tf.map_fn(tf.math.sigmoid, y_pred)

        # Calculate the labels using the style of Cao et al.
        above_thresh = tf.map_fn(
            lambda x: tf.cast(
                x > 0.5,
                tf.float32),
            cum_probs)

        # Sum across columns to estimate how many cumulative thresholds are
        # passed.
        labels_v2 = tf.reduce_sum(above_thresh, axis=1)

        if not self.sparse:
            y_true = tf.reduce_sum(y_true, axis=1)

        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        y_true = tf.squeeze(y_true)

        if sample_weight is not None:
            values = tf.cast(tf.less_equal(
                tf.abs(y_true - labels_v2), tf.cast(self.tolerance, y_pred.dtype)),
                y_pred.dtype)
            sample_weight = tf.cast(tf.squeeze(sample_weight), y_pred.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
            self.accs.assign_add(tf.reduce_sum(values))
            self.count.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.accs.assign_add(tf.reduce_sum(tf.cast(tf.less_equal(
                tf.abs(y_true - labels_v2), tf.cast(self.tolerance, y_pred.dtype)),
                y_pred.dtype)))
            self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self) -> tf.Tensor:
        return tf.math.divide_no_nan(self.accs, self.count)

    def reset_state(self) -> None:
        """Resets all of the metric state variables at the start of each epoch."""
        self.accs.assign(0.0)
        self.count.assign(0.0)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            'sparse': self.sparse,
            'tolerance': self.tolerance,
        }
        base_config = super().get_config()
        return {**base_config, **config}
