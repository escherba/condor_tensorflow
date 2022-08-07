"""
Loss function definitions
"""
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras.losses import Reduction

from keras import losses
from keras import backend as K

from .activations import ordinal_softmax
from .types import FloatArray
from .utils import encode_ordinal_labels


def _reduce_losses(
        values: tf.Tensor,
        reduction: Reduction) -> tf.Tensor:
    """Reduces loss values to specified reduction."""
    if reduction == Reduction.NONE:
        return values

    if reduction in [
        Reduction.AUTO,
        Reduction.SUM_OVER_BATCH_SIZE,
    ]:
        return tf.reduce_mean(values)

    if reduction == Reduction.SUM:
        return tf.reduce_sum(values)

    raise ValueError(f"'{reduction}' is not a valid reduction.")


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class CondorNegLogLikelihood(losses.Loss):

    """Ordinal Negative Log-likelihood Loss
    """
    num_classes: Optional[int]
    sparse: bool
    from_type: str

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = False,
            from_type: str = "ordinal_logits",
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
        self.num_classes = num_classes
        self.sparse = sparse
        self.from_type = from_type

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "from_type": self.from_type,
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    # Modifed from: https://github.com/tensorflow/tensorflow/blob/6dcd6fcea73ad613e78039bd1f696c35e63abb32/tensorflow/python/ops/nn_impl.py#L112-L148
    @staticmethod
    def ordinal_loss(
            logits: tf.Tensor,
            labels: tf.Tensor) -> tf.Tensor:
        """Negative log likelihood loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        labels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        Returns
        ----------
        loss: tf.Tensor, shape=(num_samples,)
            Loss vector, note that tensorflow will reduce it to a single number
            automatically.
        """
        logits = tf.cast(logits, dtype=tf.float32)
        labs = tf.cast(labels, dtype=tf.float32)
        # line below makes loss work with 3D tensors
        # pi_labels = tf.concat([tf.ones((tf.shape(labs)[0], 1)), labs[:, -1][:, :-1]], 1)
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
        relu_logits = array_ops.where(cond, logits, zeros)
        neg_abs_logits = array_ops.where(cond, -logits, logits)
        temp = math_ops.add(
            relu_logits - logits * labs,
            math_ops.log1p(math_ops.exp(neg_abs_logits)),
        )
        # line below makes loss work with 3D tensors
        # array_ops.where(pi_labels > zeros, temp[:, -1], zeros)
        return array_ops.where(pi_labels > zeros, temp, zeros)

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Forward pass"""

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if self.sparse:
            # Convert each true label to a vector of ordinal level indicators.
            y_true = encode_ordinal_labels(y_true, self.num_classes)

        from_type = self.from_type
        if from_type == "ordinal_logits":
            loss_values = self.ordinal_loss(y_pred, y_true)
        elif from_type == "probs":
            raise NotImplementedError("not yet implemented")
        elif from_type == "logits":
            raise NotImplementedError("not yet implemented")
        else:
            raise ValueError(f"Unknown from_type value {from_type}")
        return _reduce_losses(loss_values, self.reduction)


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class CondorOrdinalCrossEntropy(losses.Loss):

    """Ordinal cross-entropy loss
    """

    num_classes: Optional[int]
    sparse: bool
    importance_weights: Optional[FloatArray]
    from_type: str

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = False,
            importance_weights: Optional[FloatArray] = None,
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
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.importance_weights = importance_weights
        self.sparse = sparse
        self.from_type = from_type

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serializing"""
        config = {
            "num_classes": self.num_classes,
            "importance_weights": self.importance_weights,
            "sparse": self.sparse,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    @staticmethod
    def ordinal_loss(
            logits: tf.Tensor,
            levels: tf.Tensor,
            importance_weights: Optional[FloatArray] = None) -> tf.Tensor:
        """Cross-entropy loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples, num_classes-1)
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
        loss_values = (
            logprobs * levels
            + (tf.math.log(1 - tf.math.exp(logprobs) + eps) * (1 - levels))
        )
        if importance_weights is not None:
            importance_weights = tf.cast(importance_weights, dtype=loss_values.dtype)
            loss_values = tf.multiply(loss_values, importance_weights)
        return loss_values

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Forward pass logic"""
        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)

        # get number of classes
        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if self.sparse:
            y_true = encode_ordinal_labels(y_true, self.num_classes)

        from_type = self.from_type
        if from_type == "ordinal_logits":
            loss_values = self.ordinal_loss(y_pred, y_true, self.importance_weights)
        elif from_type == "probs":
            raise NotImplementedError("not yet implemented")
        elif from_type == "logits":
            raise NotImplementedError("not yet implemented")
        else:
            raise ValueError(f"Unknown from_type value {from_type}")
        return -_reduce_losses(loss_values, self.reduction)


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
class OrdinalEarthMoversDistance(losses.Loss):
    """Computes earth movers distance for ordinal labels.
    """

    num_classes: Optional[int]
    sparse: bool

    def __init__(
            self,
            num_classes: Optional[int] = None,
            sparse: bool = False,
            name: str = "earth_movers_distance",
            **kwargs: Any) -> None:
        """Creates a `OrdinalEarthMoversDistance` instance."""
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.sparse = sparse

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            "num_classes": self.num_classes,
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes mean absolute error for ordinal labels.

        Args:
          y_true: Cumulative logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        dtype = y_pred.dtype

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_true = tf.cast(y_true, dtype=dtype)

        # basic setup
        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        if not self.sparse:
            # not sparse: obtain labels from levels
            y_true = tf.reduce_sum(y_true, axis=1)

        if len(y_true.get_shape()) == 1:
            y_true = tf.expand_dims(y_true, axis=1)

        class_probs = ordinal_softmax(y_pred)
        y_dist = tf.abs(y_true - tf.range(self.num_classes, dtype=dtype))
        loss_values = tf.math.multiply(y_dist, class_probs)
        return _reduce_losses(loss_values, self.reduction)
