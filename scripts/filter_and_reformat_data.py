import logging
import pathlib
from argparse import ArgumentParser
from typing import Any, Callable

from code_stack.filter_and_reformat import filter_and_reformat as filter_code_stack
from datasets import Dataset, IterableDatasetDict, load_dataset
from datasets.splits import Split
from en_pile.filter_and_reformat import filter_and_reformat as filter_en_pile
from en_wiki.filter_and_reformat import filter_and_reformat as filter_en_wiki
from ja_cc.filter_and_reformat import filter_and_reformat as filter_ja_cc
from ja_wiki.filter_and_reformat import filter_and_reformat as filter_ja_wiki
from tqdm import tqdm

logger = logging.getLogger(__name__)

CHUNK_SIZE = 100_000


def get_data_files(search_dir: pathlib.Path, ext: str) -> dict[Split, pathlib.Path]:
    train_files = list(search_dir.glob(f"*train*.{ext}"))
    valid_files = list(search_dir.glob(f"*valid*.{ext}"))
    test_files = list(search_dir.glob(f"*test*.{ext}"))
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
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
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
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading the dataset")
    dataset: IterableDatasetDict = load_dataset(
        "json",
        data_files={k: str(v) for k, v in get_data_files(data_dir, "jsonl").items()},
        streaming=True,
    )

    filter_fn: Callable[..., dict[str, Any]]
    if args.DATASET_NAME == "ja_wiki":
        filter_fn = filter_ja_wiki
    elif args.DATASET_NAME == "en_wiki":
        filter_fn = filter_en_wiki
    elif args.DATASET_NAME == "ja_cc":
        filter_fn = filter_ja_cc
    elif args.DATASET_NAME == "en_pile":
        filter_fn = filter_en_pile
    elif args.DATASET_NAME == "code_stack":
        filter_fn = filter_code_stack
    else:
        raise ValueError(f"Unknown dataset name: {args.DATASET_NAME}.")

    dataset = dataset.map(
        filter_fn,
        remove_columns=list(
            set(list(dataset["train"].take(1))[0].keys()) - {"text", "meta"}
        ),
        batched=False,
    ).filter(
        lambda example: example["text"] != "",
    )

    logger.info(f"Writing the reformatted data to {output_dir}.")
    for split, ds in dataset.items():
        chunk_index = 0
        for batch in tqdm(ds.iter(batch_size=CHUNK_SIZE)):
            output_file: pathlib.Path = output_dir.joinpath(
                f"{split}_{chunk_index}.parquet"
            )
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
                chunk_index += 1
                continue

            Dataset.from_dict(batch).to_parquet(output_file)
            logger.info(
                f"Finished writing the tokenized {split} split to {output_file}."
            )
            chunk_index += 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
