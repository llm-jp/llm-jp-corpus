"""Script to download the Stack data.

Note:
    This script is partly borrowed from the following repository,
    which is distributed under the Apache License 2.0.
        https://github.com/togethercomputer/RedPajama-Data
"""
import argparse
import logging
import pathlib

from datasets import load_dataset

logger = logging.getLogger(__name__)


def get_data(data_dir: pathlib.Path) -> None:
    stack_dataset = load_dataset("bigcode/the-stack")
    for split, dataset in stack_dataset.items():
        file_path: pathlib.Path = data_dir.joinpath(f"stack_{split}.jsonl")
        dataset.to_json(file_path, force_ascii=False)
    logger.info("Finished Downloading Stack.")


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
    args = parser.parse_args()

    get_data(data_dir=pathlib.Path(args.data_dir))
