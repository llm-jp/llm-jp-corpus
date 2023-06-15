"""Script to download English Wikipedia data.

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

LANGUAGE = "en"
DUMP_DATE = "20230320"


def get_data(
    language: str, date: str, data_dir: pathlib.Path, num_proc: int = 1
) -> None:
    wiki_dataset = load_dataset(
        "wikipedia",
        language=language,
        date=date,
        beam_runner="DirectRunner",
        num_proc=num_proc,
    )
    for split, dataset in wiki_dataset.items():
        file_path: pathlib.Path = data_dir.joinpath(
            f"wiki_{language}_{date}_{split}.jsonl"
        )
        dataset.to_json(file_path)
        logger.info(
            f"Finished Downloading {language} {date}. "
            f"There are total {len(dataset['id'])} pages."
        )


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
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use for downloading.",
    )
    args = parser.parse_args()

    get_data(
        LANGUAGE,
        DUMP_DATE,
        data_dir=pathlib.Path(args.data_dir),
        num_proc=args.num_proc,
    )
