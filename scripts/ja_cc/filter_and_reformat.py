import json
import logging
import typing
from urllib.parse import urlparse

import regex

logger = logging.getLogger(__name__)

DATASET_NAME = "mc4"
LANGUAGE = "ja"

PATTERN = regex.compile(r"[\p{Script=Hiragana}\p{Script=Katakana}ー]+")

SCRIPT_PATTERN = regex.compile(
    r"[\u0000-\u007F\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]{100,}"
)
URL_PATTERN = regex.compile(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+")

VALID_URLS = {
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


def regex_filter(sentence: str, pattern) -> tuple[str, str]:
    valid: str = ""
    invalid: str = ""
    matched = pattern.finditer(sentence)
    index: int = 0
    for m in matched:
        valid += sentence[index : m.start()]
        invalid += sentence[m.start() : m.end()]
        invalid += " "
        index = m.end()
    valid += sentence[index:]
    return valid, invalid


def extract_text(text: str) -> tuple[str, str]:
    valid: str = ""
    invalid: str = ""
    for sentence in text.split("\n"):
        if PATTERN.search(sentence):
            url_valid, url_invalid = regex_filter(sentence, URL_PATTERN)
            invalid += url_invalid
            script_valid, script_invalid = regex_filter(url_valid, SCRIPT_PATTERN)
            invalid += script_invalid
            valid += script_valid
        else:
            invalid += sentence
    return valid, invalid


def get_top_level_domain(url: str) -> str:
    domain: typing.Optional[str] = urlparse(url).hostname
    assert domain is not None
    tld: str = domain.split(".")[-1]
    return tld


def valid_url(url: str) -> bool:
    if url.startswith("https://ja.wikipedia.org/"):
        return False
    tld: str = get_top_level_domain(url)
    return tld in VALID_URLS


def filter_and_reformat(line: str) -> dict:
    row = json.loads(line)
    if valid_url(row["meta"]["url"]):
        valid, invalid = extract_text(row["text"])
    else:
        valid = ""
        invalid = row["text"]
    return {
        "text": valid,
        "meta": {
            "url": row["url"],
            "language": LANGUAGE,
            "timestamp": row["timestamp"],
            "source": DATASET_NAME,
            "invalid_text": invalid,
        },
    }
