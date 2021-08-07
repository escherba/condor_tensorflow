import tensorflow as tf

# The outer function is a constructor to create a loss function using a certain number of classes.
class CondorOrdinalCrossEntropy(tf.keras.losses.Loss):

  def __init__(self,
               num_classes,
               importance_weights = None,
               from_type = "ordinal_logits",
               name = "ordinal_crossent",
               **kwargs):
    """ Cross-entropy loss designed for ordinal outcomes.

    Parameters
    ----------
    num_classes: int
        number of ranks (aka labels or values) in the ordinal variable

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
    self.num_classes = num_classes
    self.importance_weights = importance_weights
    self.from_type = from_type

    super().__init__(name = name, **kwargs)

  def ordinal_loss(self, logits, levels, importance):
      """ Cross-entropy loss function designed for ordinal outcomes.

      Parameters
      ----------
      logits: tf.Tensor, shape=(num_samples,num_classes-1)
          Logit output of the final Dense(num_classes-1) layer.

      levels: tf.Tensor, shape=(num_samples, num_classes-1)
          Encoded lables provided by CondorOrdinalEncoder.

      importance_weights: tf or np array of floats, shape(numclasses-1,)
          Importance weights for each binary classification task.

      Returns
      ----------
      loss: tf.Tensor, shape=(num_samples,)
          Loss vector, note that tensorflow will reduce it to a single number
          automatically.
      """
      logprobs = tf.math.cumsum(tf.math.log_sigmoid(logits), axis = 1)
      eps = tf.keras.backend.epsilon()
      val = (-tf.reduce_sum( importance *( logprobs * levels
                              + (tf.math.log(1 - tf.math.exp(logprobs)+eps)*
                                (1 - levels)) ),
                            axis = 1))
      return val

  # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
  def call(self, y_true, y_pred):

    # Ensure that y_true is the same type as y_pred (presumably a float).
    #y_pred = ops.convert_to_tensor_v2(y_pred)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # we are not sparse here, so labels are encoded already
    tf_levels = y_true

    if self.importance_weights is None:
      importance_weights = tf.ones(self.num_classes - 1, dtype = tf.float32)
    else:
      importance_weights = tf.cast(self.importance_weights, dtype = tf.float32)

    if self.from_type == "ordinal_logits":
      return self.ordinal_loss(y_pred, tf_levels, importance_weights)
    elif self.from_type == "probs":
      raise Exception("not yet implemented")
    elif self.from_type == "logits":
      raise Exception("not yet implemented")
    else:
      raise Exception("Unknown from_type value " + self.from_type +
                      " in CondorOrdinalCrossEntropy()")

  def get_config(self):
    base_config = super().get_config()
    config = {
        "num_classes": self.num_classes,
        "importance_weights": self.importance_weights,
        "from_type": self.from_type,
    }
    return {**base_config, **config}


# The outer function is a constructor to create a loss function using a certain number of classes.
class SparseCondorOrdinalCrossEntropy(CondorOrdinalCrossEntropy):

  def __init__(self,
               num_classes,
               importance_weights = None,
               from_type = "ordinal_logits",
               name = "ordinal_crossent",
               **kwargs):
    """ Cross-entropy loss designed for ordinal outcomes.

    Parameters
    ----------
    num_classes: int
        number of ranks (aka labels or values) in the ordinal variable

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
    super().__init__(name = name,
                     num_classes = num_classes,
                     importance_weights = importance_weights,
                     from_type = from_type,
                     **kwargs)

  def label_to_levels(self, label):
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))

    # This line requires that label values begin at 0. If they start at a higher
    # value it will yield an error.
    num_zeros = self.num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)

    zero_vec = tf.zeros(shape = (num_zeros), dtype = tf.int32)

    levels = tf.concat([label_vec, zero_vec], axis = 0)

    return tf.cast(levels, tf.float32)


  # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
  def call(self, y_true, y_pred):

    # Ensure that y_true is the same type as y_pred (presumably a float).
    #y_pred = ops.convert_to_tensor_v2(y_pred)
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # Convert each true label to a vector of ordinal level indicators.
    tf_levels = tf.map_fn(self.label_to_levels, y_true)

    if self.importance_weights is None:
      importance_weights = tf.ones(self.num_classes - 1, dtype = tf.float32)
    else:
      importance_weights = tf.cast(self.importance_weights, dtype = tf.float32)

    if self.from_type == "ordinal_logits":
      return self.ordinal_loss(y_pred, tf_levels, importance_weights)
    elif self.from_type == "probs":
      raise Exception("not yet implemented")
    elif self.from_type == "logits":
      raise Exception("not yet implemented")
    else:
      raise Exception("Unknown from_type value " + self.from_type +
                      " in CondorOrdinalCrossEntropy()")

