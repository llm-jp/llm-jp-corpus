import json
import logging
import pathlib
import random
from argparse import ArgumentParser
from collections.abc import Generator
from typing import Any, Union

from datasets import Dataset, disable_caching
from datasets.splits import Split

logger = logging.getLogger(__name__)
disable_caching()

random.seed(42)


def main() -> None:
    parser = ArgumentParser()
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
    parser.add_argument(
        "--train_token_size",
        type=int,
        default=-1,
        help="Train token size (negative values mean no limit).",
    )
    parser.add_argument(
        "--valid_token_size",
        type=int,
        default=50_000,
        help="Validation token size.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)

    logger.info(
        f"Extract data up to {args.train_token_size} tokens from {Split.TRAIN} split."
    )
    train_file = output_dir.joinpath("train.jsonl")
    valid_file = output_dir.joinpath("valid.jsonl")
    if train_file.exists() and not args.overwrite:
        logger.warning(
            f"{train_file} already exists. Specify --overwrite to overwrite."
        )
        return
    if valid_file.exists() and not args.overwrite:
        logger.warning(
            f"{valid_file} already exists. Specify --overwrite to overwrite."
        )
        return

    input_files = sorted(data_dir.glob(f"{Split.TRAIN}_*.parquet"))
    if not input_files:
        return

    with train_file.open(mode="wt") as fout:
        for example in extract_examples(args.train_token_size, input_files):
            fout.write(json.dumps(example, ensure_ascii=False) + "\n")


def extract_examples(
    token_size: int, input_files: list[pathlib.Path]
) -> Generator[dict[str, Any], None, None]:
    random.shuffle(input_files)
    remaining_token_size: Union[int, float] = (
        token_size if token_size > 0 else float("inf")
    )
    for input_file in input_files:
        dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        for example in dataset:
            yield example
            num_tokens = len(example["tokens"])
            remaining_token_size -= num_tokens
            if remaining_token_size <= 0:
                return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
