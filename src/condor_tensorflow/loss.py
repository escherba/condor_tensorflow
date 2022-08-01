"""
Loss function definitions
"""
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras import losses
from tensorflow.keras import backend as K

from .activations import ordinal_softmax
from .types import TensorLike


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

        >>> encode_ordinal_labels_v1(tf.constant([0, 1, 2], dtype=tf.float32), num_classes=3)
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
        zero_vec = tf.zeros(shape=(num_zeros), dtype=tf.int32)
        return tf.cast(tf.concat([label_vec, zero_vec], 0), dtype)
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

        >>> encode_ordinal_labels_v2([0, 1, 2], num_classes=3)
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


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class CondorNegLogLikelihood(losses.Loss):

    """Ordinal Negative Log-likelihood Loss
    """
    sparse: bool
    from_type: str

    def __init__(
            self,
            from_type: str = "ordinal_logits",
            sparse: bool = False,
            name: str = "ordinal_nll",
            **kwargs: Any) -> None:
        """Negative log likelihood loss designed for ordinal outcomes.

        Parameters
        ----------
        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        super().__init__(name=name, **kwargs)
        self.from_type = from_type
        self.sparse = sparse

    # Modifed from: https://github.com/tensorflow/tensorflow/blob/6dcd6fcea73ad613e78039bd1f696c35e63abb32/tensorflow/python/ops/nn_impl.py#L112-L148
    def ordinal_loss(
            self,
            logits: tf.Tensor,
            labels: tf.Tensor,
            name: Optional[str] = None) -> tf.Tensor:
        """Negative log likelihood loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        levels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        with ops.name_scope(name, "logistic_loss", [logits, labels]) as scope:
            if isinstance(logits, tf.Tensor):
                logits = tf.cast(logits, dtype=tf.float32, name="logits")
            else:
                logits = ops.convert_to_tensor(logits, dtype=tf.float32, name="logits")
            if isinstance(labels, tf.Tensor):
                labs = tf.cast(labels, dtype=tf.float32, name="labs")
            else:
                labs = ops.convert_to_tensor(labels, dtype=tf.float32, name="labs")
            pi_labels = tf.concat([tf.ones((tf.shape(labs)[0], 1)), labs[:, :-1]], 1)

            # The logistic loss formula from above is
            #   x - x * z + log(1 + exp(-x))
            # For x < 0, a more numerically stable formula is
            #   -x * z + log(1 + exp(x))
            # Note that these two expressions can be combined into the following:
            #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
            # To allow computing gradients at zero, we define custom versions of max and
            # abs functions.
            zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
            cond = logits >= zeros
            cond2 = pi_labels > zeros
            relu_logits = array_ops.where(cond, logits, zeros)
            neg_abs_logits = array_ops.where(cond, -logits, logits)
            temp = math_ops.add(
                relu_logits - logits * labs,
                math_ops.log1p(math_ops.exp(neg_abs_logits)),
            )
            return tf.math.reduce_sum(
                array_ops.where(cond2, temp, zeros), axis=1, name=scope
            )

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Forward pass"""

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.sparse:
            # get number of classes
            num_classes = tf.shape(y_pred)[1] + 1

            # Convert each true label to a vector of ordinal level indicators.
            y_true = encode_ordinal_labels(y_true, num_classes)

        from_type = self.from_type
        if from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, y_true)
        if from_type == "probs":
            raise NotImplementedError("not yet implemented")
        if from_type == "logits":
            raise NotImplementedError("not yet implemented")
        raise ValueError(f"Unknown from_type value {from_type}")

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serializing"""
        config = {
            "from_type": self.from_type,
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class CondorOrdinalCrossEntropy(losses.Loss):

    """Ordinal cross-entropy loss
    """

    importance_weights: Optional[TensorLike]
    sparse: bool
    from_type: str

    def __init__(
            self,
            importance_weights: Optional[TensorLike] = None,
            sparse: bool = False,
            from_type: str = "ordinal_logits",
            name: str = "ordinal_crossent",
            **kwargs: Any) -> None:
        """Cross-entropy loss designed for ordinal outcomes.

        Parameters
        ----------
        importance_weights: tf or np array of floats, shape(numclasses-1,)
            (Optional) importance weights for each binary classification task.

        from_type: one of "ordinal_logits" (default), or "probs".
          Ordinal logits are the output of a Dense(num_classes-1) layer with no activation.
          (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        self.importance_weights = importance_weights
        self.sparse = sparse
        self.from_type = from_type
        super().__init__(name=name, **kwargs)

    def ordinal_loss(
            self,
            logits: tf.Tensor,
            levels: tf.Tensor,
            importance_weights: TensorLike) -> tf.Tensor:
        """Cross-entropy loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        levels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        importance_weights: tf or np array of floats, shape(numclasses-1,)
            importance weights for each binary classification task.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        logprobs = tf.math.cumsum(tf.math.log_sigmoid(logits), axis=1)
        eps = K.epsilon()
        return -tf.reduce_sum(
            importance_weights
            * (
                logprobs * levels
                + (tf.math.log(1 - tf.math.exp(logprobs) + eps) * (1 - levels))
            ),
            axis=1,
        )

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Forward pass logic"""
        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        num_classes = tf.shape(y_pred)[1] + 1

        if self.sparse:
            y_true = encode_ordinal_labels(y_true, num_classes)

        if self.importance_weights is None:
            importance_weights = tf.ones(num_classes - 1, dtype=tf.float32)
        else:
            importance_weights = tf.cast(self.importance_weights, dtype=tf.float32)

        from_type = self.from_type
        if from_type == "ordinal_logits":
            return self.ordinal_loss(y_pred, y_true, importance_weights)
        if from_type == "probs":
            raise NotImplementedError("not yet implemented")
        if from_type == "logits":
            raise NotImplementedError("not yet implemented")
        raise ValueError(f"Unknown from_type value {from_type}")

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serializing"""
        config = {
            "importance_weights": self.importance_weights,
            "sparse": self.sparse,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class OrdinalEarthMoversDistance(losses.Loss):
    """Computes earth movers distance for ordinal labels."""

    sparse: bool

    def __init__(
            self,
            sparse: bool = False,
            name: str = "earth_movers_distance",
            **kwargs: Any) -> None:
        """Creates a `OrdinalEarthMoversDistance` instance."""
        super().__init__(name=name, **kwargs)
        self.sparse = sparse

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulatiuve logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
        """

        y_pred = tf.convert_to_tensor(y_pred)

        # basic setup
        cum_probs = ordinal_softmax(y_pred)
        num_classes = tf.shape(cum_probs)[1]

        if not self.sparse:
            # not sparse: obtain labels from levels
            y_true = tf.reduce_sum(y_true, axis=1)

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_true = tf.cast(y_true, y_pred.dtype)

        # remove all dimensions of size 1 (e.g., from [[1], [2]], to [1, 2])
        # y_true = tf.squeeze(y_true)

        y_dist = tf.map_fn(
            fn=lambda y: tf.abs(y - tf.range(num_classes, dtype=y_pred.dtype)),
            elems=y_true,
        )

        return tf.reduce_sum(tf.math.multiply(y_dist, cum_probs), axis=1)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}
