"""
Loss function definitions
"""
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, math_ops
from tensorflow.keras import losses
from tensorflow.keras import backend as K

from .activations import ordinal_softmax
from .types import TensorLike
from .utils import encode_ordinal_labels


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
    @staticmethod
    def ordinal_loss(
            logits: tf.Tensor,
            labels: tf.Tensor,
            name: Optional[str] = None) -> tf.Tensor:
        """Negative log likelihood loss function designed for ordinal outcomes.

        Parameters
        ----------
        logits: tf.Tensor, shape=(num_samples,num_classes-1)
            Logit output of the final Dense(num_classes-1) layer.

        labels: tf.Tensor, shape=(num_samples, num_classes-1)
            Encoded lables provided by CondorOrdinalEncoder.

        name: str
            execution scope name

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
            return tf.math.reduce_sum(
                # line below makes loss work with 3D tensors
                # array_ops.where(pi_labels > zeros, temp[:, -1], zeros), axis=1, name=scope
                array_ops.where(pi_labels > zeros, temp, zeros), axis=1, name=scope
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

    @staticmethod
    def ordinal_loss(
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
    """Computes earth movers distance for ordinal labels.

    See https://arxiv.org/abs/1611.05916
    """

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
          y_true: Cumulative logits from CondorOrdinal layer.
          y_pred: CondorOrdinal Encoded Labels.
        """
        y_pred = tf.convert_to_tensor(y_pred)
        dtype = y_pred.dtype

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_true = tf.convert_to_tensor(y_true)
        y_true = tf.cast(y_true, dtype=dtype)

        # basic setup
        cum_probs = ordinal_softmax(y_pred)
        num_classes = tf.shape(cum_probs)[1]

        if not self.sparse:
            # not sparse: obtain labels from levels
            y_true = tf.reduce_sum(y_true, axis=1)

        if y_true.ndim == 1:
            y_true = tf.expand_dims(y_true, axis=1)
        y_dist = tf.abs(y_true - tf.range(num_classes, dtype=dtype))
        return tf.reduce_sum(tf.math.multiply(y_dist, cum_probs), axis=1)

    def get_config(self) -> Dict[str, Any]:
        """Returns the serializable config of the metric."""
        config = {
            "sparse": self.sparse,
        }
        base_config = super().get_config()
        return {**base_config, **config}
