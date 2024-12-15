import pytest

from src.english2kana.data_processing.preprocess import (
    check_halfwidth_english,
    convert_fullwidth_to_halfwidth,
    pipeline,
    to_lower,
)


class TestPreprocessFunctions:
    def test_convert_fullwidth_to_halfwidth(self):
        input_text = "ＡＢＣＤ"
        expected = "ABCD"
        assert convert_fullwidth_to_halfwidth(input_text) == expected

    def test_check_halfwidth_english_valid(self):
        input_text = "Hello"
        assert check_halfwidth_english(input_text) == input_text

    def test_check_halfwidth_english_invalid(self):
        input_text = "Hello世界"
        with pytest.raises(ValueError, match=r"non-halfwidth-English"):
            check_halfwidth_english(input_text)

    def test_to_lower(self):
        assert to_lower("ABC") == "abc"
        assert to_lower("Example_Word") == "example_word"

    def test_pipeline(self):
        input_text = "Ｅｘａｍｐｌｅ"  # Fullwidth chars
        expected = "example"
        result = pipeline(input_text)
        assert result == expected
