import json
import logging
import pathlib
import random
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Optional

from datasets import Dataset, disable_caching
from datasets.splits import Split
from extract_ids import get_example_id
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()

random.seed(42)


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
        "--valid_id_file",
        type=str,
        default=None,
        help="Validation ID file.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Output format.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list_input_files(args.input_path))
    if not input_files:
        return
    random.shuffle(input_files)

    id_data: dict[str, Any] = json.loads(pathlib.Path(args.valid_id_file).read_text())
    validation_ids: set[str] = set(id_data["ids"])
    id_key = id_data["key"]

    with Pool(args.num_proc) as p:
        results = p.starmap(
            process_file,
            [
                (
                    input_file,
                    validation_ids,
                    id_key,
                    output_dir,
                    args.output_format,
                    args.overwrite,
                )
                for input_file in input_files
            ],
        )

    train_token_size = sum(result[0] for result in results)
    valid_token_size = sum(result[1] for result in results)
    train_example_size = sum(result[2] for result in results)
    valid_example_size = sum(result[3] for result in results)

    if train_token_size is not None:
        logger.info(
            f"Finished extracting train data of {train_token_size:,} tokens, {train_example_size:,} examples."
        )
    if valid_token_size is not None:
        logger.info(
            f"Finished extracting valid data of {valid_token_size:,} tokens, {valid_example_size:,} examples."
        )


def process_file(
    input_file: pathlib.Path,
    validation_ids: set[str],
    id_key: str,
    output_dir: pathlib.Path,
    output_format: str,
    overwrite: bool,
):
    dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
    train_examples = []
    valid_examples = []
    train_token_size: Optional[int] = 0
    valid_token_size: Optional[int] = 0
    train_example_size = valid_example_size = 0

    for example in dataset:
        if "num_tokens" not in example:
            valid_token_size = train_token_size = None
        id_ = get_example_id(example, id_key)
        if id_ in validation_ids:
            if valid_token_size is not None:
                valid_token_size += example["num_tokens"]
            valid_example_size += 1
            valid_examples.append(example)
        else:
            if train_token_size is not None:
                train_token_size += example["num_tokens"]
            train_example_size += 1
            train_examples.append(example)

    train_dataset = Dataset.from_list(train_examples)
    valid_dataset = Dataset.from_list(valid_examples)

    output_file: pathlib.Path = output_dir / f"{input_file.stem}.{output_format}"
    save_dataset(
        train_dataset,
        output_file,
        overwrite,
        output_format,
    )

    output_file = (
        output_dir
        / f"{input_file.stem.replace(str(Split.TRAIN), str(Split.VALIDATION))}.{output_format}"
    )
    save_dataset(
        valid_dataset,
        output_file,
        overwrite,
        output_format,
    )

    return train_token_size, valid_token_size, train_example_size, valid_example_size


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
