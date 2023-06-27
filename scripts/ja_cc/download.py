import logging
import pathlib

from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "mc4"
LANGUAGE = "ja"


def download(output_dir: pathlib.Path) -> None:
    cc_dataset = load_dataset(DATASET_NAME, languages=[LANGUAGE])
    for split, dataset in cc_dataset.items():
        file_path: pathlib.Path = output_dir.joinpath(
            f"{DATASET_NAME}_{LANGUAGE}_{split}.jsonl"
        )
        dataset.to_json(file_path, force_ascii=False)
        logger.info(
            f"Finished downloading the {split} split. "
            f"There are total {len(dataset['id'])} pages."
        )
