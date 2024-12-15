import tensorflow as tf
from tensorflow.keras import backend as K


def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the masked accuracy, ignoring samples where the true labels are masked.

    Args:
        y_true (tf.Tensor): The ground truth labels, typically one-hot encoded.
        y_pred (tf.Tensor): The predicted labels, typically as logits or probabilities.

    Returns:
        tf.Tensor: The computed masked accuracy as a scalar tensor.
    """
    mask = K.cast(K.not_equal(K.argmax(y_true, axis=-1), 0), K.floatx())
    correct = K.cast(
        K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx()
    )
    accuracy = K.sum(correct * mask) / K.sum(mask)
    return accuracy
