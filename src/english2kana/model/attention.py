from typing import Any, Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.types.experimental import TensorLike


class DotAttention(Layer):  # type: ignore
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(
        self,
        inputs: tuple[TensorLike, TensorLike],
        mask: Optional[tuple[Optional[TensorLike], Optional[TensorLike]]] = None,
    ) -> tuple[TensorLike, TensorLike]:
        decoder_outputs, encoder_outputs = inputs
        encoder_outputs_transposed = tf.transpose(encoder_outputs, perm=[0, 2, 1])
        score = tf.matmul(decoder_outputs, encoder_outputs_transposed)

        if mask is not None:
            _, encoder_mask = mask
            if encoder_mask is not None:
                encoder_mask = tf.cast(encoder_mask, dtype=score.dtype)
                encoder_mask = tf.expand_dims(encoder_mask, axis=1)
                score += (1.0 - encoder_mask) * -1e9

        attention_weights = tf.nn.softmax(score, axis=-1)
        context_vector = tf.matmul(attention_weights, encoder_outputs)

        return context_vector, attention_weights

    def compute_mask(
        self,
        inputs: tuple[TensorLike, TensorLike],
        mask: Optional[tuple[Optional[TensorLike], Optional[TensorLike]]] = None,
    ) -> Optional[TensorLike]:
        if mask is None:
            return None
        decoder_mask, _ = mask
        return decoder_mask
