import logging
import pathlib
import random
from argparse import ArgumentParser
from collections.abc import Iterator
from typing import Union

from datasets import Dataset, DatasetDict, concatenate_datasets, disable_caching
from datasets.splits import Split

logger = logging.getLogger(__name__)
disable_caching()

random.seed(42)

CHUNK_SIZE = 100_000


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
        default="1M",
        help="Validation token size.",
    )
    parser.add_argument(
        "--interleave_steps",
        type=str,
        default="1K",
        help="Interleave steps to sample validation data.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_token_size: Union[int, float] = canonicalize_number(args.train_token_size)
    valid_token_size = canonicalize_number(args.valid_token_size)
    interleave_steps = canonicalize_number(args.interleave_steps)
    if train_token_size < 0:
        train_token_size = float("inf")
    assert valid_token_size > 0
    assert interleave_steps > 0
    logger.info(f"Extract train data up to {train_token_size} tokens")
    logger.info(f"Extract valid data up to {valid_token_size} tokens")

    input_files = sorted(list_input_files(args.input_path))
    if not input_files:
        return
    random.shuffle(input_files)

    cur_train_token_size: int = 0
    cur_valid_token_size: int = 0
    buff_train_dataset: Dataset = Dataset.from_dict({})
    buff_valid_dataset: Dataset = Dataset.from_dict({})
    train_chunk_index: int = 0
    valid_chunk_index: int = 0
    output_file: pathlib.Path
    for input_file in input_files:
        dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        if cur_valid_token_size >= valid_token_size:
            # Use all the data for training.
            buff_train_dataset = concatenate_datasets([buff_train_dataset, dataset])
            cur_train_token_size += sum(dataset["num_tokens"])
        else:
            # Use 0.1% of the data for validation.
            dataset_split: DatasetDict = dataset.train_test_split(
                test_size=0.001, shuffle=True, seed=42
            )
            buff_train_dataset = concatenate_datasets(
                [buff_train_dataset, dataset_split["train"]]
            )
            buff_valid_dataset = concatenate_datasets(
                [buff_valid_dataset, dataset_split["test"]]
            )
            cur_train_token_size += sum(dataset_split["train"]["num_tokens"])
            cur_valid_token_size += sum(dataset_split["test"]["num_tokens"])

        # Save the data.
        if len(buff_train_dataset) >= CHUNK_SIZE:
            output_file = output_dir / f"{Split.TRAIN}_{train_chunk_index}.parquet"
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
            else:
                buff_train_dataset.select(range(0, CHUNK_SIZE)).to_parquet(output_file)
                train_chunk_index += 1
                buff_train_dataset = buff_train_dataset.select(
                    range(CHUNK_SIZE, len(buff_train_dataset))
                )

        if len(buff_valid_dataset) >= CHUNK_SIZE:
            output_file = output_dir / f"{Split.VALIDATION}_{valid_chunk_index}.parquet"
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
            else:
                buff_valid_dataset.select(range(0, CHUNK_SIZE)).to_parquet(output_file)
                valid_chunk_index += 1
                buff_valid_dataset = buff_valid_dataset.select(
                    range(CHUNK_SIZE, len(buff_valid_dataset))
                )

        if (
            cur_train_token_size >= train_token_size
            and cur_valid_token_size >= valid_token_size
        ):
            break

    if len(buff_train_dataset):
        output_file = output_dir / f"{Split.TRAIN}_{train_chunk_index}.parquet"
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to overwrite."
            )
        else:
            Dataset.from_dict(buff_train_dataset[:CHUNK_SIZE]).to_parquet(output_file)

    if len(buff_valid_dataset):
        output_file = output_dir / f"{Split.VALIDATION}_{valid_chunk_index}.parquet"
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to overwrite."
            )
        else:
            Dataset.from_dict(buff_valid_dataset[:CHUNK_SIZE]).to_parquet(output_file)
    logger.info(f"Finished extracting train data of {cur_train_token_size} tokens.")
    logger.info(f"Finished extracting valid data of {cur_valid_token_size} tokens.")


def canonicalize_number(number: str) -> int:
    if number.endswith("k") or number.endswith("K"):
        return int(number[:-1]) * 1000
    elif number.endswith("M"):
        return int(number[:-1]) * 1_000_000
    elif number.endswith("B") or number.endswith("G"):
        return int(number[:-1]) * 1_000_000_000
    elif number.endswith("T"):
        return int(number[:-1]) * 1_000_000_000_000
    else:
        return int(number)


def list_input_files(input_paths: list[str]) -> Iterator[pathlib.Path]:
    for path_str in input_paths:
        path = pathlib.Path(path_str)
        if path.exists() is False:
            logger.warning(f"{path} not found and skipped")
            continue
        yield from path.glob("*.parquet") if path.is_dir() else [path]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
