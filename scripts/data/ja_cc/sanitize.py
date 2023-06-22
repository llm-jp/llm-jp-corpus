import argparse
import csv
import joblib
import json
import regex
from urllib.parse import urlparse

PATTERN = regex.compile(r'[\p{Script=Hiragana}\p{Script=Katakana}ー]+')

SCRIPT_PATTERN = regex.compile(r'[\u0000-\u007F\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]{100,}')
URL_PATTERN = regex.compile(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+')

VALID_URLS = set(["jp", "com", "net", "org", "work", "info", "xyz", "biz", "work", "me", "tv", "site", "tokyo", "cc"])


def regex_filter(sentence, pattern):
    valid = ""
    invalid = ""
    matched = pattern.finditer(sentence)
    index = 0
    for m in matched:
        valid += sentence[index:m.start()]
        invalid += sentence[m.start():m.end()]
        invalid += " "
        index = m.end()
    valid += sentence[index:]
    return valid, invalid


def extract_text(text):
    valid = ""
    invalid = ""
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


def getTLD(url):
    domain = urlparse(url).hostname
    tld = domain.split(".")[-1]
    return tld


def valid_url(url):
    if url.startswith("https://ja.wikipedia.org/"):
        return False
    tld = getTLD(url)
    return tld in VALID_URLS


def sanitize(line):
    entry = json.loads(line)
    if valid_url(entry["url"]):
        valid, invalid = extract_text(entry["text"])
    else:
        valid = ""
        invalid = entry["text"]
    return {"text": valid, "timestamp": entry["timestamp"], "url": entry["url"], "invalid_text": invalid}


def main(args):
    with open(args.c4_json) as f:
        lines = f.readlines()
        data = joblib.Parallel(n_jobs=-1)(joblib.delayed(sanitize)(i) for i in lines)
    with open(args.output, "w") as wf:
        for d in data:
            js = json.dumps(d)
            wf.write(js)
            wf.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C4 sanitizer")
    parser.add_argument("c4_json")
    parser.add_argument("output")
    args = parser.parse_args()
    main(args)
