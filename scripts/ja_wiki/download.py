import logging
import pathlib

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "wikipedia"
LANGUAGE = "ja"
DUMP_DATE = "20230320"


def download(output_dir: pathlib.Path) -> None:
    wiki_dataset = load_dataset(
        DATASET_NAME,
        language=LANGUAGE,
        date=DUMP_DATE,
        beam_runner="DirectRunner",
    )
    for split, dataset in wiki_dataset.items():
        file_path: pathlib.Path = output_dir.joinpath(
            f"{DATASET_NAME}_{LANGUAGE}_{DUMP_DATE}_{split}.jsonl"
        )
        dataset.to_json(file_path, force_ascii=False)
        logger.info(
            f"Finished downloading the {split} split. "
            f"There are total {len(dataset['id'])} pages."
        )
