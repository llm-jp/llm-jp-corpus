import json
import logging
import pathlib
import random
from argparse import ArgumentParser
from collections.abc import Generator
from typing import Any

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
    parser.add_argument(
        "--interleave_steps",
        type=int,
        default=1_000,
        help="Interleave steps to sample validation data.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)

    logger.info(
        f"Extract data up to {args.train_token_size} tokens from {Split.TRAIN} split."
    )
    train_file = output_dir.joinpath("train.jsonl")
    valid_file = output_dir.joinpath("valid.jsonl")
    if (train_file.exists() or valid_file.exists()) and not args.overwrite:
        logger.warning("Already exists. Specify --overwrite to overwrite.")
        return

    input_files = sorted(data_dir.glob(f"{Split.TRAIN}_*.parquet"))
    if not input_files:
        return

    cur_train_token_size: int = 0
    cur_valid_token_size: int = 0
    with train_file.open("wt") as f_train, valid_file.open("wt") as f_valid:
        for i, example in enumerate(iterate_examples(input_files)):
            if (
                i % args.interleave_steps == 0
                and cur_valid_token_size < args.valid_token_size
            ):
                f_valid.write(json.dumps(example, ensure_ascii=False) + "\n")
                cur_valid_token_size += len(example["tokens"])
            else:
                f_train.write(json.dumps(example, ensure_ascii=False) + "\n")
                cur_train_token_size += len(example["tokens"])
            if (
                cur_train_token_size >= args.train_token_size
                and cur_valid_token_size >= args.valid_token_size
            ):
                break


def iterate_examples(
    input_files: list[pathlib.Path],
) -> Generator[dict[str, Any], None, None]:
    random.shuffle(input_files)
    for input_file in input_files:
        dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        yield from dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
