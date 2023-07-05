import json
import logging
import pathlib
from argparse import ArgumentParser
from collections.abc import Generator
from typing import Any, Union

from datasets import Dataset, disable_caching
from datasets.splits import Split

logger = logging.getLogger(__name__)
disable_caching()


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
        "--token_size",
        type=int,
        default=-1,
        help="Token size (negative values mean no limit).",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    for split in (Split.TRAIN, Split.VALIDATION, Split.TEST):
        logger.info(f"Extract data up to {args.token_size} tokens from {split} split.")
        output_file_name = f"{split}_sampled.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        if output_file.exists() and not args.overwrite:
            logger.warning(f"{output_file} already exists. Skip this file.")
            continue

        input_files = sorted(data_dir.glob(f"{split}_*.parquet"))
        if not input_files:
            continue

        examples = extract_examples(args.token_size, input_files)

        with output_file.open(mode="wt") as fout:
            for example in examples:
                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Finished extracting from {split} split.")


def extract_examples(
    token_size: int, input_files: list[pathlib.Path]
) -> Generator[dict[str, Any], None, None]:
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
