import re

from mojimoji import zen_to_han


def pipeline(text: str) -> str:
    halfwidth_text = convert_fullwidth_to_halfwidth(text)
    halfwidth_text = check_halfwidth_english(halfwidth_text)
    halfwidth_text_lower = to_lower(halfwidth_text)
    return halfwidth_text_lower


def convert_fullwidth_to_halfwidth(text: str) -> str:
    return zen_to_han(text)


def check_halfwidth_english(text: str) -> str:
    """
    Verify that the input string consists only of halfwidth English letters (a-z, A-Z).

    Raises:
        ValueError: If the string contains any non-halfwidth-English character.
    """
    if re.search(r"[^a-zA-Z]", text):
        raise ValueError(f"Input contains non-halfwidth-English characters: {text}")
    return text


def to_lower(text: str) -> str:
    return text.lower()
