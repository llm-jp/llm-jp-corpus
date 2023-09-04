import math
import typing
import zlib
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import regex
from hojichar import Document
from hojichar.filters.document_filters import AcceptJapanese


def reformat_builder(text_field: str) -> Callable[..., dict[str, Any]]:
    def reformat(example: dict[str, Any]) -> dict[str, Any]:
        text = example[text_field]
        meta = example.get("meta", {})
        meta.update({k: v for k, v in example.items() if k not in {text_field, "meta"}})
        return {"text": text, "meta": meta}

    return reformat


def has_valid_domain(example: dict[str, Any]) -> bool:
    if example["meta"]["url"].startswith("https://ja.wikipedia.org/"):
        return False
    domain: typing.Optional[str] = urlparse(example["meta"]["url"]).hostname
    assert domain is not None
    tld: str = domain.split(".")[-1]
    return tld in {
        "jp",
        "com",
        "net",
        "org",
        "work",
        "info",
        "xyz",
        "biz",
        "work",
        "me",
        "tv",
        "site",
        "tokyo",
        "cc",
    }


def has_valid_extension(example: dict[str, Any]) -> bool:
    # https://github.com/togethercomputer/RedPajama-Data/blob/main/data_prep/github/github_run_filter.py
    valid_extensions: set[str] = {
        "asm",
        "bat",
        "cmd",
        "c",
        "h",
        "cs",
        "cpp",
        "hpp",
        "c++",
        "h++",
        "cc",
        "hh",
        "C",
        "H",
        "cmake",
        "css",
        "dockerfile",
        "f90",
        "f",
        "f03",
        "f08",
        "f77",
        "f95",
        "for",
        "fpp",
        "go",
        "hs",
        "html",
        "java",
        "js",
        "jl",
        "lua",
        "md",
        "markdown",
        "php",
        "php3",
        "php4",
        "php5",
        "phps",
        "phpt",
        "pl",
        "pm",
        "pod",
        "perl",
        "ps1",
        "psd1",
        "psm1",
        "py",
        "rb",
        "rs",
        "sql",
        "scala",
        "sh",
        "bash",
        "command",
        "zsh",
        "ts",
        "tsx",
        "tex",
        "vb",
        "Dockerfile",
        "Makefile",
        "xml",
        "rst",
        "m",
        "smali",
    }
    return example["meta"]["ext"] in valid_extensions


def has_valid_max_line_length(example: dict[str, Any]) -> bool:
    # https://github.com/togethercomputer/RedPajama-Data/blob/main/data_prep/github/github_run_filter.py
    return example["meta"]["max_line_length"] <= 1000


def has_valid_avg_line_length(example: dict[str, Any]) -> bool:
    # https://github.com/togethercomputer/RedPajama-Data/blob/main/data_prep/github/github_run_filter.py
    return example["meta"]["avg_line_length"] <= 100


def has_valid_alphanum_fraction(example: dict[str, Any]) -> bool:
    # https://github.com/togethercomputer/RedPajama-Data/blob/main/data_prep/github/github_run_filter.py
    return example["meta"]["alphanum_fraction"] >= 0.25


def has_good_compression_ratio(
    min_score: float = 0.3, max_score: float = 0.7, length_factor: float = 0.0
):
    """Checks if data compression (deflate) yields a desired size of data stream.

    NOTE(odashi, 2023-09-03):
    Ths judgment is based on an assumption that a "natual" sentence has an entropy
    within a certain range, and both "too simple" (low entropy) and "too complex" (high
    entropy) sentences don't reflect human's usual writing.
    This function calculates the data compression ratio (calculated by the Deflate
    algorithm) of the original stream, and compares if the resulting ratio is in-between
    the specified range.
    This criterion is somewhat sensitive against the length of the original stream (e.g.
    if the input is long, the resulting compression ratio tends to be small).
    This function also has a mechanism to consider the original length (adjusted by the
    `length_factor` parameter).

    Args:
        min_score: The lower bound of the compression ratio.
        max_score: The upper bound of the compression ratio.
        length_factor: Penalty factor of log(original_byte_length), usually set to
            something larger than 0. Using 0 falls back to a simple compression ratio.

    Returns:
        Judgment function, bound with `min` and `max`.

    Example:
        >>> judge = has_good_compression_ratio(0.1, 1.0, 0.0)
        >>> judge({"text": "LbdJA66Ufy4Pr6ffQEIo0DL60OL7kQl6y6ohAhqYKf3laCruuR"})
        False  # 1.16
        >>> judge({"text": "a"*200})
        False  # 0.06
        >>> judge({"text": "This is a usual sentence. This sentence should pass this judgment."})
        True  # 0.92
    """

    def judge(example):
        encoded = example["text"].encode("utf-8")
        compressed = zlib.compress(encoded, level=9)
        encoded_length = len(encoded)
        compressed_length = len(compressed)
        ratio = compressed_length / encoded_length
        length_penalty = (
            length_factor * math.log(encoded_length) if length_factor else 0.0
        )
        score = ratio + length_penalty
        return min_score <= score <= max_score

    return judge


accept_japanese_filter = AcceptJapanese()


def is_japanese(example: dict[str, Any]) -> bool:
    doc = accept_japanese_filter.apply(Document(example["text"]))
    return not doc.is_rejected


def is_not_empty(example: dict[str, Any]) -> bool:
    return example["text"].strip() != ""


nsfw_words: list[str] = []
with Path(__file__).parent.joinpath("nsfw_words/ja.txt").open() as f:
    for line in f:
        if not line.startswith("#"):
            nsfw_words.append(line.strip())


def is_ethical(example: dict[str, Any]) -> bool:
    nsfw_word_count: int = 0
    for word in nsfw_words:
        nsfw_word_count += example["text"].count(word)
        if nsfw_word_count >= 3:
            return False
    return True


def extract_japanese_text(example: dict[str, Any]) -> dict[str, Any]:
    ja_pat = regex.compile(r"[\p{Script=Hiragana}\p{Script=Katakana}ー]+")
    script_pat = regex.compile(
        r"[\u0000-\u007F\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]{100,}"
    )
    url_pat = regex.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")

    def regex_filter(sentence: str, pat) -> str:
        valid: str = ""
        index: int = 0
        for m in pat.finditer(sentence):
            valid += sentence[index : m.start()]
            index = m.end()
        valid += sentence[index:]
        return valid

    valid: str = ""
    for sentence in example["text"].split("\n"):
        if ja_pat.search(sentence):
            sentence = regex_filter(sentence, url_pat)
            sentence = regex_filter(sentence, script_pat)
            valid += sentence
    example["text"] = valid
    return example


def remove_wikipedia_footnote(example: dict[str, Any]) -> dict[str, Any]:
    footnote_sections: list[str] = [
        "脚注",
        "関連項目",
        "日本国内の関連項目",
        "出典",
        "出典・脚注",
        "参照",
        "外部リンク",
        "参考文献",
        "その他関連事項",
        "Footnotes",
        "See also",
        "Further reading",
        "Bibliography",
        "References",
        "Notes",
        "Citations",
        "Sources",
        "External links",
    ]
    footnote_pat = regex.compile(rf"\n({'|'.join(footnote_sections)})\s*\n")
    m = footnote_pat.search(example["text"])
    if m:
        example["text"] = example["text"][: m.start()]
    return example


def remove_empty_parenthesis(example: dict[str, Any]) -> dict[str, Any]:
    # Japanese
    example["text"] = regex.sub(r"（[\s,，、;；]*", "（", example["text"])
    example["text"] = regex.sub(r"[\s,，、;；]*）", "）", example["text"])
    example["text"] = regex.sub(r"（\s*）", "", example["text"])
    # English
    example["text"] = regex.sub(r"\([\s,;]*", "(", example["text"])
    example["text"] = regex.sub(r"[\s,;]*\)", ")", example["text"])
    example["text"] = regex.sub(r"\s?\(\s*\)", "", example["text"])
    return example
