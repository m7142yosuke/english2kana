from typing import Any

import tensorflow as tf
from tensorflow.keras.layers import Layer


class DotAttention(Layer):  # type: ignore
    """
    Luong-style Dot Attention

    Input:
        decoder_outputs: (batch, dec_timesteps, latent_dim)
        encoder_outputs: (batch, enc_timesteps, latent_dim)

    Output:
        context_vector: (batch, dec_timesteps, latent_dim)
        attention_weights: (batch, dec_timesteps, enc_timesteps)

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def call(self, decoder_outputs: Any, encoder_outputs: Any) -> Any:
        encoder_outputs_transposed = tf.transpose(encoder_outputs, perm=[0, 2, 1])

        score = tf.matmul(decoder_outputs, encoder_outputs_transposed)

        attention_weights = tf.nn.softmax(score, axis=-1)

        context_vector = tf.matmul(attention_weights, encoder_outputs)

        return context_vector, attention_weights
