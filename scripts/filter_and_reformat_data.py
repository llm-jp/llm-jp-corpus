import logging
import pathlib
import time
import typing
from argparse import ArgumentParser
from typing import Any, Callable
from urllib.parse import urlparse

import regex
import tqdm
from datasets import Dataset, IterableDatasetDict, disable_caching, load_dataset
from datasets.splits import Split

logger = logging.getLogger(__name__)
disable_caching()

CHUNK_SIZE = 100_000


def get_data_files(search_dir: pathlib.Path, ext: str) -> dict[Split, pathlib.Path]:
    train_files = list(search_dir.glob(f"*train*.{ext}"))
    valid_files = list(search_dir.glob(f"*valid*.{ext}"))
    test_files = list(search_dir.glob(f"*test*.{ext}"))
    assert len(train_files) == 1, f"Found {len(train_files)} train files."
    assert len(valid_files) <= 1, f"Found {len(valid_files)} valid files."
    assert len(test_files) <= 1, f"Found {len(test_files)} test files."
    data_files = {Split.TRAIN: train_files[0]}
    if len(valid_files) == 1:
        data_files[Split.VALIDATION] = valid_files[0]
    if len(test_files) == 1:
        data_files[Split.TEST] = test_files[0]
    return data_files


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


def remove_non_japanese_text(example: dict[str, Any]) -> dict[str, Any]:
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


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
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

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("Loading the dataset")
    dataset: IterableDatasetDict = load_dataset(
        "json",
        data_files={k: str(v) for k, v in get_data_files(data_dir, "jsonl").items()},
        streaming=True,
    )

    map_fns: list[Callable[..., dict[str, Any]]] = []
    filter_fns: list[Callable[..., bool]] = []
    if args.DATASET_NAME == "ja_wiki":
        map_fns.append(reformat_builder("text"))
        filter_fns.append(has_empty_text)
    elif args.DATASET_NAME == "en_wiki":
        map_fns.append(reformat_builder("text"))
        filter_fns.append(has_empty_text)
    elif args.DATASET_NAME == "ja_cc":
        map_fns.append(reformat_builder("text"))
        map_fns.append(remove_non_japanese_text)
        filter_fns.append(has_valid_domain)
        filter_fns.append(has_empty_text)
    elif args.DATASET_NAME == "en_pile":
        map_fns.append(reformat_builder("text"))
        filter_fns.append(has_empty_text)
    elif args.DATASET_NAME == "code_stack":
        map_fns.append(reformat_builder("content"))
        filter_fns.append(has_empty_text)
    else:
        raise ValueError(f"Unknown dataset name: {args.DATASET_NAME}.")

    for map_fn in map_fns:
        dataset = dataset.map(map_fn, batched=False)
    dataset = dataset.map(
        remove_columns=list(
            set(list(dataset["train"].take(1))[0].keys()) - {"text", "meta"}
        )
    )
    for filter_fn in filter_fns:
        dataset = dataset.filter(filter_fn)

    logger.info(f"Writing the reformatted data to {output_dir}.")
    for split, ds in dataset.items():
        chunk_index = 0
        for batch in tqdm.tqdm(ds.iter(batch_size=CHUNK_SIZE)):
            output_file: pathlib.Path = output_dir.joinpath(
                f"{split}_{chunk_index}.parquet"
            )
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
                chunk_index += 1
                continue

            Dataset.from_dict(batch).to_parquet(output_file)
            chunk_index += 1

    end_time = time.time()
    logger.info(
        f"Finished processing the dataset. Elapsed time: {end_time - start_time} [sec]"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
