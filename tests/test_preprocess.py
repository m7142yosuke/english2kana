import pytest

from src.english2kana.data_processing.preprocess import (
    Preprocess,
    camel_to_sep,
    check_halfwidth_english,
    convert_fullwidth_to_halfwidth,
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

    @pytest.mark.parametrize(
        "input_text, sep, expected",
        [
            ("ExampleWord", "_", "Example_Word"),
            ("exampleWord", "_", "example_Word"),
            ("example", "_", "example"),  # No split needed
            (
                "exampleWORLD",
                "_",
                "exampleWORLD",
            ),  # Doesn't match camel/pascal pattern strictly
        ],
    )
    def test_camel_to_sep(self, input_text, sep, expected):
        assert camel_to_sep(input_text, sep) == expected

    def test_to_lower(self):
        assert to_lower("ABC") == "abc"
        assert to_lower("Example_Word") == "example_word"


class TestPreprocessPipeline:
    def test_pipeline(self):
        p = Preprocess(start_token="<start>", end_token="<end>", sep="_")
        input_text = "ＥｘａｍｐｌｅＷｏｒｄ"  # Fullwidth chars
        expected = "<start>example_word<end>"
        result = p.pipeline(input_text)
        assert result == expected

    def test_pipeline_invalid_input(self):
        p = Preprocess(start_token="<start>", end_token="<end>", sep="_")
        input_text = "Hello世界"
        with pytest.raises(ValueError, match=r"non-halfwidth-English"):
            p.pipeline(input_text)
