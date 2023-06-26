import argparse
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


def sanitize(line: str):
    entry = json.loads(line)
    if valid_url(entry["meta"]["url"]):
        valid, invalid = extract_text(entry["text"])
    else:
        valid = ""
        invalid = entry["text"]
    return {"text": valid, "invalid_text": invalid, "meta": entry["meta"]}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the wikipedia data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    args = parser.parse_args()

    logger.info(f"Filtering the data in {args.data_dir}.")
    data_dir = pathlib.Path(args.data_dir)
    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Filtering {file_path}.")
        output_file_name = f"{file_path.stem}_filtered.jsonl"
        output_file = pathlib.Path(args.output_dir) / output_file_name
        if output_file.exists() and not args.overwrite:
            logger.info(f"{output_file} already exists. Skipping.")
            continue

        with file_path.open("r") as fin:
            lines = fin.readlines()
            rows = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(sanitize)(i) for i in tqdm.tqdm(lines)
            )

        with output_file.open("wt") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
