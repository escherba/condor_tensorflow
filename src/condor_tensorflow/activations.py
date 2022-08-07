"""
Activation functions
"""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="condor_tensorflow")
def ordinal_softmax(logits: tf.Tensor) -> tf.Tensor:
    """ Convert the ordinal logit output of CondorOrdinal() to label probabilities.

      Parameters
      ----------
      logits: tf.Tensor, shape=(num_samples,num_classes-1)
          Logit output of the final Dense(num_classes-1) layer.

      Returns
      ----------
      probs_tensor: tf.Tensor, shape=(num_samples, num_classes)
          Probabilities of each class (columns) for each
          sample (rows).

    """

    # Convert the ordinal logits into cumulative probabilities.
    cum_probs = tf.concat(
        [
            tf.ones((tf.shape(logits)[0], 1), dtype=tf.float32),
            tf.math.cumprod(tf.math.sigmoid(tf.cast(logits, tf.float32)), axis=1),
            tf.zeros((tf.shape(logits)[0], 1), dtype=tf.float32),
        ], 1)
    return cum_probs[:, 0:-1] - cum_probs[:, 1:]
