import json
import logging
import pathlib
import typing
from urllib.parse import urlparse

import joblib
import regex
import tqdm

logger = logging.getLogger(__name__)


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


def _filter_and_reformat(line: str, language: str, source: str):
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
            "language": language,
            "timestamp": row["timestamp"],
            "source": source,
            "invalid_text": invalid,
        },
    }


def filter_and_reformat(data_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Reformatting {file_path.stem}.")
        source, language, _ = file_path.stem.split("_")
        with file_path.open("r") as fin:
            lines: list[str] = fin.readlines()
            rows: list[dict] = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(_filter_and_reformat)(line, language, source)
                for line in tqdm.tqdm(lines)
            )
        output_file_name = f"{file_path.stem}_filtered.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        logger.info(f"Writing the reformatted data to {output_file}.")
        with output_file.open("wt") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Finished reformatting {file_path.stem}.")
