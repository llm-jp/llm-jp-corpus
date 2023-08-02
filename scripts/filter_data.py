import logging
import pathlib
import time
from argparse import ArgumentParser
from typing import Any, Callable

import tqdm
from datasets import Dataset, IterableDatasetDict, disable_caching, load_dataset
from datasets.splits import Split
from filters import (
    extract_japanese_text,
    has_valid_alphanum_fraction,
    has_valid_avg_line_length,
    has_valid_domain,
    has_valid_extension,
    has_valid_max_line_length,
    is_non_empty,
    reformat_builder,
    remove_empty_parenthesis,
    remove_wikipedia_footnote,
)

logger = logging.getLogger(__name__)
disable_caching()

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

    start_time = time.time()

    logger.info("Loading the dataset")
    dataset: IterableDatasetDict = load_dataset(
        "json",
        data_files={k: str(v) for k, v in get_data_files(data_dir, "jsonl").items()},
        streaming=True,
    )

    reformat_fn: Callable[..., dict[str, Any]]
    map_fns: list[Callable[..., dict[str, Any]]] = []
    filter_fns: list[Callable[..., bool]] = []
    if args.DATASET_NAME == "ja_wiki":
        reformat_fn = reformat_builder("text")
        map_fns.append(remove_wikipedia_footnote)
        map_fns.append(remove_empty_parenthesis)
        filter_fns.append(is_non_empty)
    elif args.DATASET_NAME == "en_wiki":
        reformat_fn = reformat_builder("text")
        map_fns.append(remove_wikipedia_footnote)
        map_fns.append(remove_empty_parenthesis)
        filter_fns.append(is_non_empty)
    elif args.DATASET_NAME == "ja_cc":
        reformat_fn = reformat_builder("text")
        map_fns.append(extract_japanese_text)
        filter_fns.append(has_valid_domain)
        filter_fns.append(is_non_empty)
    elif args.DATASET_NAME == "en_pile":
        reformat_fn = reformat_builder("text")
        filter_fns.append(is_non_empty)
    elif args.DATASET_NAME == "code_stack":
        reformat_fn = reformat_builder("content")
        filter_fns.append(has_valid_extension)
        filter_fns.append(has_valid_max_line_length)
        filter_fns.append(has_valid_avg_line_length)
        filter_fns.append(has_valid_alphanum_fraction)
        filter_fns.append(is_non_empty)
    else:
        raise ValueError(f"Unknown dataset name: {args.DATASET_NAME}.")

    dataset = dataset.map(reformat_fn, batched=False)
    columns = list(dataset["train"].take(1))[0].keys()
    dataset = dataset.map(remove_columns=list(set(columns) - {"text", "meta"}))
    for filter_fn in filter_fns:
        dataset = dataset.filter(filter_fn)
    for map_fn in map_fns:
        dataset = dataset.map(map_fn, batched=False)
    dataset = dataset.filter(is_non_empty)

    logger.info(f"Writing the reformatted data to {output_dir}.")
    for split, ds in dataset.items():
        chunk_index = 0
        for batch in tqdm.tqdm(ds.iter(batch_size=CHUNK_SIZE)):
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
            chunk_index += 1

    end_time = time.time()
    logger.info(
        f"Finished processing the dataset. Elapsed time: {end_time - start_time} [sec]"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
