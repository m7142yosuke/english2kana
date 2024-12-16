# english2kana

[![test](https://github.com/m7142yosuke/english2kana/actions/workflows/build-test.yaml/badge.svg)](https://github.com/m7142yosuke/english2kana/actions/workflows/build-test.yaml)

english2kana is a Python library designed to convert English words and romanized Japanese text into their Katakana representations. It leverages a sequence-to-sequence model with an attention mechanism, trained on data derived from English company names and their corresponding Katakana forms found on the Japan Corporate Number Publication Site.

Key Features:
- Converts English words to Katakana.
- Employs a seq2seq model architecture with attention.
- Achieves high accuracy with the given dataset.

Model Performance:
- Test Loss: 0.1440
- Test Accuracy: 0.9552

### Requirements
Python 3.11 or higher

### Installation
```bash
pip install english2kana
```

### Usage
```python
from english2kana import english2kana

# Initialize the translator
e2k = english2kana()
# Load the pretrained model
e2k.load_model()

# Translate an English word into Katakana
output = e2k.translate('simple')
print(output)  # シンプル
```

### Data
The training data is sourced from the Japan Corporate Number Publication Site. The dataset includes a wide array of English corporate names along with their correct Katakana representations, ensuring the model is exposed to various letter combinations and phonetic patterns.

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to help improve this library.

### License
The MIT License (MIT)
