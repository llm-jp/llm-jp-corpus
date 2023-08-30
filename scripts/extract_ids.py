import hashlib
import json
import logging
import pathlib
from argparse import ArgumentParser
from typing import Any, Optional

from datasets import Dataset, disable_caching
from tqdm import tqdm
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()

DATASET_NAME_TO_KEY: dict[str, Optional[str]] = {
    "ja_wiki": "id",
    "en_wiki": "id",
    "ja_cc": None,
    "en_pile": None,
    "code_stack": "hexsha",
}


def get_example_id(example: dict[str, Any], id_key: Optional[str]) -> str:
    if id_key is None or id_key == "hash":
        return hashlib.sha256(example["text"].encode()).hexdigest()
    else:
        return example["meta"][id_key]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="jsonl",
        choices=["jsonl", "parquet"],
        help="Input format.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    input_files: list[pathlib.Path] = sorted(
        list_input_files(args.input_path, args.input_format)
    )
    key = DATASET_NAME_TO_KEY[args.DATASET_NAME]

    ids_: list[str] = []
    for input_file in tqdm(input_files):
        logger.info(f"Loading dataset from {input_file}.")
        if args.input_format == "jsonl":
            dataset = Dataset.from_json(str(input_file), keep_in_memory=True)
        else:
            assert args.input_format == "parquet"
            dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        logger.info(f"Extracting keys in {input_file.name}.")
        for example in dataset:
            ids_.append(get_example_id(example, key))
    pathlib.Path(args.output_file).write_text(
        json.dumps({"key": key or "hash", "ids": ids_}, indent=2)
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
