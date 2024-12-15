import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from datasets import load_dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.english2kana.data_processing.preprocess import pipeline
from src.english2kana.model.build_model import build_model
from src.english2kana.model.losses import masked_accuracy

CONFIG_PATH = "src/english2kana/configs/config.yaml"
HUGGINGFACE_DATASET_NAME = "m7142yosuke/english2kana-v1"

START_TOKEN = "<s>"
END_TOKEN = "<e>"


def load_config() -> Any:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset(HUGGINGFACE_DATASET_NAME)
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    # Convert stringified 'None' values back to 'null' because the original 'null'
    # strings were interpreted as None during dataset loading.
    train_df["name"] = train_df["name"].astype(str).replace("None", "null")
    test_df["name"] = test_df["name"].astype(str).replace("None", "null")

    return train_df, test_df


def prepare_tokenizers(
    df: pd.DataFrame, start_token: str = START_TOKEN, end_token: str = END_TOKEN
) -> tuple[Any, Any, int, int]:
    kana = [start_token + k + end_token for k in df["name_kana"].astype(str).values]
    english = df["name"].apply(pipeline).astype(str).values

    tokenizer_english = Tokenizer(char_level=True, lower=False, filters="")
    tokenizer_kana = Tokenizer(char_level=True, lower=False, filters="")

    tokenizer_english.fit_on_texts(english)
    tokenizer_kana.fit_on_texts(kana)

    num_chars_english = len(tokenizer_english.word_index) + 1
    num_chars_kana = len(tokenizer_kana.word_index) + 1

    print(f"Number of unique characters in name: {num_chars_english}")
    print(f"Number of unique characters in name_kana: {num_chars_kana}")

    return tokenizer_english, tokenizer_kana, num_chars_english, num_chars_kana


def encode_and_split_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer_english: Tokenizer,
    tokenizer_kana: Tokenizer,
    start_token: str = START_TOKEN,
    end_token: str = END_TOKEN,
) -> Any:
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_english = combined_df["name"].apply(pipeline).astype(str).values
    combined_kana = [
        start_token + k + end_token for k in combined_df["name_kana"].astype(str).values
    ]

    combined_seq_english = tokenizer_english.texts_to_sequences(combined_english)
    combined_seq_kana = tokenizer_kana.texts_to_sequences(combined_kana)

    max_len_english = max(len(seq) for seq in combined_seq_english)
    max_len_kana = max(len(seq) for seq in combined_seq_kana)

    X_train, decoder_input_train, y_train = process_single_df(
        train_df,
        tokenizer_english,
        tokenizer_kana,
        max_len_english,
        max_len_kana,
        start_token,
        end_token,
    )
    X_test, decoder_input_test, y_test = process_single_df(
        test_df,
        tokenizer_english,
        tokenizer_kana,
        max_len_english,
        max_len_kana,
        start_token,
        end_token,
    )

    return (
        X_train,
        decoder_input_train,
        y_train,
        X_test,
        decoder_input_test,
        y_test,
        max_len_english,
        max_len_kana,
    )


def process_single_df(
    df: pd.DataFrame,
    tokenizer_english: Tokenizer,
    tokenizer_kana: Tokenizer,
    max_len_english: int,
    max_len_kana: int,
    start_token: str,
    end_token: str,
) -> Any:
    english = df["name"].apply(pipeline).astype(str).values
    kana = [start_token + k + end_token for k in df["name_kana"].astype(str).values]

    sequences_english = tokenizer_english.texts_to_sequences(english)
    sequences_kana = tokenizer_kana.texts_to_sequences(kana)

    X = pad_sequences(sequences_english, maxlen=max_len_english, padding="post")
    y = pad_sequences(sequences_kana, maxlen=max_len_kana, padding="post")

    decoder_input = np.zeros_like(y)
    decoder_input[:, 1:] = y[:, :-1]
    decoder_input[:, 0] = tokenizer_kana.word_index.get(start_token, 1)
    y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer_kana.word_index) + 1)

    return X, decoder_input, y


def save_tokenizer(tokenizer: Tokenizer, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())


def decode_sequence(
    sequence: Any, tokenizer: Tokenizer, end_token: str = END_TOKEN
) -> str:
    end_idx = tokenizer.word_index.get(end_token, 0)
    decoded = []
    for idx in sequence:
        if idx == end_idx:
            break
        if idx != 0:
            decoded.append(tokenizer.index_word.get(idx, ""))
    return "".join(decoded)


def main() -> None:
    config = load_config()

    embedding_dim = config["model"]["embedding_dim"]
    latent_dim = config["model"]["latent_dim"]
    batch_size = config["training"]["batch_size"]
    epochs = config["training"]["epochs"]
    learning_rate = config["training"]["learning_rate"]

    train_df, test_df = load_raw_data()

    tokenizer_english, tokenizer_kana, num_chars_english, num_chars_kana = (
        prepare_tokenizers(pd.concat([train_df, test_df]))
    )
    (
        X_train,
        decoder_input_train,
        y_train,
        X_test,
        decoder_input_test,
        y_test,
        max_len_english,
        max_len_kana,
    ) = encode_and_split_data(train_df, test_df, tokenizer_english, tokenizer_kana)

    model = build_model(
        num_chars_english,
        num_chars_kana,
        embedding_dim,
        latent_dim,
        max_len_english,
        max_len_kana,
    )
    print(
        f"{num_chars_english=}, {num_chars_kana=}, {embedding_dim=}, {latent_dim=}, {max_len_english=}, {max_len_kana=}"
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=[masked_accuracy]
    )

    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/best_model.keras"
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        mode="min",
        verbose=1,
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    callbacks = [early_stopping, model_checkpoint, reduce_lr]

    model.fit(
        [X_train, decoder_input_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_test, decoder_input_test], y_test),
        callbacks=callbacks,
    )

    loss, accuracy = model.evaluate(
        [X_test, decoder_input_test], y_test, batch_size=batch_size
    )
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    model.save("models/final_model.keras")
    save_tokenizer(tokenizer_english, "models/tokenizer_english.json")
    save_tokenizer(tokenizer_kana, "models/tokenizer_kana.json")

    # Inference
    y_pred = model.predict([X_test, decoder_input_test])
    y_pred_int = np.argmax(y_pred, axis=-1)
    y_test_int = np.argmax(y_test, axis=-1)

    for i in range(min(100, len(X_test))):
        input_seq = X_test[i]
        pred_seq = y_pred_int[i]
        true_seq = y_test_int[i]

        input_str = decode_sequence(input_seq, tokenizer_english)
        pred_str = decode_sequence(pred_seq, tokenizer_kana)
        true_str = decode_sequence(true_seq, tokenizer_kana)

        print(f"Input (English): {input_str}")
        print(f"Predicted (Katakana): {pred_str}")
        print(f"True (Katakana): {true_str}")
        print("---")


if __name__ == "__main__":
    main()
