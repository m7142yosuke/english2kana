import json
import tempfile

import numpy as np
import tensorflow as tf
import yaml
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from english2kana.model.attention import DotAttention

from english2kana.data_processing.preprocess import pipeline

CONFIG_PATH = "english2kana/configs/config.yaml"
HUGGINGFACE_MODEL_REPO_ID = "m7142yosuke/english2kana"

START_TOKEN = "<s>"
END_TOKEN = "<e>"


class English2KanaInferer:
    def __init__(self):
        self.model = None
        self.tokenizer_english = None
        self.tokenizer_kana = None
        self.config = self._load_config()

        self.max_len_english = self.config["model"]["max_len_english"]
        self.max_len_kana = self.config["model"]["max_len_kana"]

    def _load_config(self):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config

    def load_model(self, version: str = "latest") -> None:
        cache_dir = tempfile.mkdtemp()

        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL_REPO_ID,
            filename=f"english2kana-{version}.keras",
            cache_dir=cache_dir,
        )
        tokenizer_english_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL_REPO_ID,
            filename="tokenizer_english.json",
            cache_dir=cache_dir,
        )
        tokenizer_kana_path = hf_hub_download(
            repo_id=HUGGINGFACE_MODEL_REPO_ID,
            filename="tokenizer_kana.json",
            cache_dir=cache_dir,
        )

        # Load the model
        self.model = tf.keras.models.load_model(model_path, compile=False, custom_objects={"DotAttention": DotAttention})

        # Load tokenizers
        with open(tokenizer_english_path, encoding="utf-8") as f:
            tokenizer_english_data = json.load(f)
            self.tokenizer_english = tokenizer_from_json(
                json.dumps(tokenizer_english_data)
            )

        with open(tokenizer_kana_path, encoding="utf-8") as f:
            tokenizer_kana_data = json.load(f)
            self.tokenizer_kana = tokenizer_from_json(json.dumps(tokenizer_kana_data))

    def translate(self, english_text: str) -> str:
        if self.model is None:
            raise ValueError("Model not loaded. Please call `load_model()` first.")

        english_text = pipeline(english_text)

        # Tokenize and pad the input
        seq = self.tokenizer_english.texts_to_sequences([english_text])
        X = pad_sequences(seq, maxlen=self.max_len_english, padding="post")

        # Greedy decoding
        kana_seq = [self.tokenizer_kana.word_index.get(START_TOKEN, 1)]
        for _ in range(self.max_len_kana - 1):
            dec_in = pad_sequences([kana_seq], maxlen=self.max_len_kana, padding="post")
            pred = self.model.predict([X, dec_in], verbose=0)
            pred_id = np.argmax(pred[0, len(kana_seq) - 1])
            if (
                pred_id == self.tokenizer_kana.word_index.get(END_TOKEN, 0)
                or pred_id == 0
            ):
                break
            kana_seq.append(pred_id)

        return self._decode_sequence(kana_seq, self.tokenizer_kana, END_TOKEN)

    @staticmethod
    def _decode_sequence(sequence, tokenizer, end_token=END_TOKEN):
        decoded = []
        for idx in sequence:
            if idx != 0:
                decoded.append(tokenizer.index_word.get(idx, ""))
        kana = "".join(decoded).lstrip(START_TOKEN).rstrip(END_TOKEN)
        sanitized_kana = kana.replace("<", "").replace(">", "")
        return sanitized_kana
