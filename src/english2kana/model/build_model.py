from tensorflow.keras.layers import LSTM, Concatenate, Dense, Embedding, Input
from tensorflow.keras.models import Model

from .attention import DotAttention


def build_model(
    num_chars_english: int,
    num_chars_kana: int,
    embedding_dim: int,
    latent_dim: int,
    max_len_english: int,
    max_len_kana: int,
) -> Model:
    """
    Builds and returns a seq2seq model for English-to-Kana conversion.

    Args:
        num_chars_english (int): Size of the input character dictionary (English).
        num_chars_kana (int): Size of the output character dictionary (Kana).
        embedding_dim (int): Dimension of the output from the embedding layer.
        latent_dim (int): Dimension of the hidden state in the LSTM.
        max_len_english (int): Maximum sequence length on the input side.
        max_len_kana (int): Maximum sequence length on the output side.

    Returns:
        Model: A constructed Keras model (encoder-decoder with attention).
    """

    # encoder
    encoder_inputs = Input(shape=(max_len_english,), name="encoder_input")
    enc_emb = Embedding(
        input_dim=num_chars_english,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )(encoder_inputs)
    encoder_lstm = LSTM(
        latent_dim, return_sequences=True, return_state=True, name="encoder_lstm"
    )
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = Input(shape=(max_len_kana,), name="decoder_input")
    dec_emb_layer = Embedding(
        input_dim=num_chars_kana,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )
    dec_emb = dec_emb_layer(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # attention
    context_vector, _ = DotAttention()([decoder_outputs, encoder_outputs])

    decoder_concat_input = Concatenate(axis=-1, name="decoder_concat")(
        [decoder_outputs, context_vector]
    )

    decoder_dense = Dense(num_chars_kana, activation="softmax", name="decoder_output")
    decoder_outputs = decoder_dense(decoder_concat_input)

    model = Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model"
    )

    return model
