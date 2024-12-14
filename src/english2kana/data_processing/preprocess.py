import re

from mojimoji import zen_to_han


class Preprocess:
    def __init__(self, start_token: str, end_token: str, sep: str):
        self.start_token = start_token
        self.end_token = end_token
        self.sep = sep

    def pipeline(self, text: str) -> str:
        halfwidh_text = convert_fullwidth_to_halfwidth(text)
        halfwidh_text = check_halfwidth_english(halfwidh_text)
        halfwidh_text_with_sep = camel_to_sep(halfwidh_text, sep=self.sep)
        halfwidh_text_with_sep_lower = to_lower(halfwidh_text_with_sep)

        return self.start_token + halfwidh_text_with_sep_lower + self.end_token


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


def camel_to_sep(s: str, sep: str) -> str:
    """
    Convert a PascalCase or camelCase string into a separated form using `sep`.
    For example, "ExampleWord" -> "Example_Word" and "exampleWord" -> "example_Word".

    If the string does not match these patterns, returns it unchanged.
    """
    pattern = re.compile(r"^(?:[A-Z][a-z]+(?:[A-Z][a-z]+)*|[a-z]+(?:[A-Z][a-z]+)*)$")

    if pattern.match(s):
        return re.sub(r"(?<=[a-z])(?=[A-Z])", sep, s)
    else:
        return s


def to_lower(text: str) -> str:
    return text.lower()
