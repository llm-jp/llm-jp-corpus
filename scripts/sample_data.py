import logging
import pathlib
import random
from argparse import ArgumentParser
from typing import Union

from datasets import Dataset, DatasetDict, disable_caching
from datasets.splits import Split
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()

random.seed(42)

CHUNK_SIZE = 100_000
# This value should be as large as possible but no greater than the value in the respective corpus.
NUM_TOKENS_PER_EXAMPLE = 500


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
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
        type=str,
        default="-1",
        help="Train token size (negative values mean no limit).",
    )
    parser.add_argument(
        "--valid_token_size",
        type=str,
        default="10M",
        help="Validation token size.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output format.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_token_size: Union[int, float] = canonicalize_number(args.train_token_size)
    valid_token_size = canonicalize_number(args.valid_token_size)
    if train_token_size < 0:
        train_token_size = float("inf")
    assert valid_token_size > 0
    logger.info(f"Extract train data up to {train_token_size} tokens")
    logger.info(f"Extract valid data up to {valid_token_size} tokens")

    input_files = sorted(list_input_files(args.input_path))
    if not input_files:
        return
    random.shuffle(input_files)

    num_valid_examples_per_shard = int(
        valid_token_size / NUM_TOKENS_PER_EXAMPLE / len(input_files)
    )

    cur_train_token_size: int = 0
    cur_valid_token_size: int = 0
    output_file: pathlib.Path
    for input_file in input_files:
        dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        if cur_valid_token_size < valid_token_size:
            dataset_split: DatasetDict = dataset.train_test_split(
                test_size=num_valid_examples_per_shard, shuffle=True, seed=42
            )
            train_dataset = dataset_split[Split.TRAIN]
            valid_dataset = dataset_split[Split.TEST]
        else:
            # Use all the data for training.
            train_dataset = dataset
            valid_dataset = Dataset.from_dict({})
        if cur_train_token_size < train_token_size:
            output_file = output_dir / f"{input_file.stem}.{args.output_format}"
            save_dataset(
                train_dataset,
                output_file,
                args.overwrite,
                args.output_format,
            )
            cur_train_token_size += sum(train_dataset["num_tokens"])
        if cur_valid_token_size < valid_token_size:
            output_file = (
                output_dir
                / f"{input_file.stem.replace(str(Split.TRAIN), str(Split.VALIDATION))}.{args.output_format}"
            )
            save_dataset(
                valid_dataset,
                output_file,
                args.overwrite,
                args.output_format,
            )
            cur_valid_token_size += sum(valid_dataset["num_tokens"])

        if (
            cur_train_token_size >= train_token_size
            and cur_valid_token_size >= valid_token_size
        ):
            break

    logger.info(f"Finished extracting train data of {cur_train_token_size:,} tokens.")
    logger.info(f"Finished extracting valid data of {cur_valid_token_size:,} tokens.")


def canonicalize_number(number: str) -> int:
    if number.endswith("k") or number.endswith("K"):
        return int(number[:-1]) * 1_000
    elif number.endswith("M"):
        return int(number[:-1]) * 1_000_000
    elif number.endswith("B") or number.endswith("G"):
        return int(number[:-1]) * 1_000_000_000
    elif number.endswith("T"):
        return int(number[:-1]) * 1_000_000_000_000
    else:
        return int(number)


def save_dataset(
    dataset: Dataset, output_file: pathlib.Path, overwrite: bool, format: str
) -> None:
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    if format == "jsonl":
        dataset.to_json(output_file, force_ascii=False)
    else:
        assert format == "parquet"
        dataset.to_parquet(output_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
