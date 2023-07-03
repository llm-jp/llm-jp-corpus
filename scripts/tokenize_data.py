import logging
import os
import pathlib
from argparse import ArgumentParser
from typing import Any

from datasets import DatasetDict, load_dataset
from datasets.splits import Split
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

tokenizer: PreTrainedTokenizer


def tokenize_function(examples) -> dict[str, Any]:
    encodings = tokenizer(
        examples["text"],
        truncation=False,
        return_attention_mask=False,
        is_split_into_words=False,
    )
    return {
        "input_ids": encodings["input_ids"],
        "tokens": [enc.tokens for enc in encodings.encodings],
    }


def get_data_files(search_dir: pathlib.Path) -> dict[Split, pathlib.Path]:
    train_files = list(search_dir.glob("*train*.jsonl"))
    valid_files = list(search_dir.glob("*valid*.jsonl"))
    test_files = list(search_dir.glob("*test*.jsonl"))
    assert len(train_files) == 1, f"Found {len(train_files)} train files."
    assert len(valid_files) <= 1, f"Found {len(valid_files)} valid files."
    assert len(test_files) <= 1, f"Found {len(test_files)} test files."
    data_files = {Split.TRAIN: train_files[0]}
    if len(valid_files) == 1:
        data_files[Split.VALIDATION] = valid_files[0]
    if len(test_files) == 1:
        data_files[Split.TEST] = test_files[0]
    return data_files


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
        "--model_name",
        type=str,
        default="cyberagent/open-calm-7b",  # TODO: Update the default model name.
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        logger.warning(f"{output_dir} already exists. Specify --overwrite to continue.")
        exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initialize the tokenizer.")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading the dataset")
    dataset: DatasetDict = load_dataset(
        "json", data_files={k: str(v) for k, v in get_data_files(data_dir).items()}
    )
    logger.info("Tokenizing the dataset.")
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=128,
        num_proc=os.cpu_count(),
    )
    logger.info("Finished tokenizing the dataset.")

    logger.info(f"Writing the tokenized data to {output_dir}.")
    for split, ds in dataset.items():
        output_file: pathlib.Path = output_dir.joinpath(f"{split}.parquet")
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to continue."
            )
            exit(1)
        ds.to_parquet(output_file)
        logger.info(f"Finished writing the tokenized {split} split to {output_file}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
