"""Script to download Japanese Wikipedia data.

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

DATASET_NAME = "wikipedia"
LANGUAGE = "en"
DUMP_DATE = "20230320"


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

    data_dir = pathlib.Path(args.data_dir)
    wiki_dataset = load_dataset(
        DATASET_NAME,
        language=LANGUAGE,
        date=DUMP_DATE,
        beam_runner="DirectRunner",
    )
    for split, dataset in wiki_dataset.items():
        file_path: pathlib.Path = data_dir.joinpath(
            f"{DATASET_NAME}_{LANGUAGE}_{DUMP_DATE}_{split}.jsonl"
        )
        dataset.to_json(file_path, force_ascii=False)
        logger.info(
            f"Finished downloading the {split} split. "
            f"There are total {len(dataset['id'])} pages."
        )
