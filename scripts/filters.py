import typing
from typing import Any, Callable
from urllib.parse import urlparse

import regex


def reformat_builder(text_field: str) -> Callable[..., dict[str, Any]]:
    def reformat(example: dict[str, Any]) -> dict[str, Any]:
        return {
            "text": example[text_field],
            "meta": {
                **{k: v for k, v in example.items() if k != text_field},
            },
        }

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


def has_empty_text(example: dict[str, Any]) -> bool:
    return example["text"].strip() != ""


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
    example["text"] = regex.sub(r"(（）|\s?\(\))", "", example["text"])
    example["text"] = regex.sub(r"、+）", "）", example["text"])
    example["text"] = regex.sub(r"（,\s+", "（", example["text"])
    example["text"] = regex.sub(r"\s?\([\s,]*\)", "", example["text"])
    return example
