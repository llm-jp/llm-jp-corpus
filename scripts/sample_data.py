import logging
import pathlib
import random
from argparse import ArgumentParser
from collections.abc import Iterator
from typing import Any, Union

from datasets import Dataset, disable_caching
from datasets.splits import Split
from tqdm import tqdm

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

    cur_train_token_size: int = 0
    cur_valid_token_size: int = 0
    buff_train_examples = []
    buff_valid_examples = []
    train_chunk_index: int = 0
    valid_chunk_index: int = 0
    output_file: pathlib.Path
    for i, example in tqdm(enumerate(iterate_examples(input_files))):
        num_tokens = example.get("num_tokens", len(example["tokens"]))
        if i % interleave_steps == 0 and cur_valid_token_size < valid_token_size:
            buff_valid_examples.append(example)
            cur_valid_token_size += num_tokens
        elif cur_train_token_size < train_token_size:
            buff_train_examples.append(example)
            cur_train_token_size += num_tokens

        if len(buff_train_examples) >= CHUNK_SIZE:
            output_file = output_dir / f"{Split.TRAIN}_{train_chunk_index}.parquet"
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
            else:
                Dataset.from_list(buff_train_examples).to_parquet(output_file)
            train_chunk_index += 1
            buff_train_examples = []
        if len(buff_valid_examples) >= CHUNK_SIZE:
            output_file = output_dir / f"{Split.VALIDATION}_{valid_chunk_index}.parquet"
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
            else:
                Dataset.from_list(buff_valid_examples).to_parquet(output_file)
            valid_chunk_index += 1
            buff_valid_examples = []

        if (
            cur_train_token_size >= train_token_size
            and cur_valid_token_size >= valid_token_size
        ):
            break

    if buff_train_examples:
        output_file = output_dir / f"{Split.TRAIN}_{train_chunk_index}.parquet"
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to overwrite."
            )
        else:
            Dataset.from_list(buff_train_examples).to_parquet(output_file)
    if buff_valid_examples:
        output_file = output_dir / f"{Split.VALIDATION}_{valid_chunk_index}.parquet"
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to overwrite."
            )
        else:
            Dataset.from_list(buff_valid_examples).to_parquet(output_file)


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


def iterate_examples(input_files: list[pathlib.Path]) -> Iterator[dict[str, Any]]:
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
