import logging
import pathlib
import time
import typing
from argparse import ArgumentParser
from typing import Any, Callable

import tqdm
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    disable_caching,
    load_dataset,
)
from datasets.splits import Split
from filters import (
    extract_japanese_text,
    has_good_average_sentence_length,
    has_good_compression_ratio,
    has_valid_alphanum_fraction,
    has_valid_avg_line_length,
    has_valid_domain,
    has_valid_extension,
    has_valid_max_line_length,
    is_japanese,
    is_not_ad_content,
    is_not_adult_content,
    is_not_discrimination_content,
    is_not_empty,
    is_not_violence_content,
    reformat_data,
    remove_empty_parenthesis,
    remove_wikipedia_footnote,
)

logger = logging.getLogger(__name__)
disable_caching()

CHUNK_SIZE = 10_000_000

def get_data_files(search_dir: pathlib.Path, ext: str) -> dict[Split, str]:
    data_files = {}
    if len(list(search_dir.glob(f"*train*.{ext}"))):
        data_files[Split.TRAIN] = str(search_dir.joinpath(f"*train*.{ext}"))
    if len(list(search_dir.glob(f"*valid*.{ext}"))):
        data_files[Split.VALIDATION] = str(search_dir.joinpath(f"*valid*.{ext}"))
    if len(list(search_dir.glob(f"*test*.{ext}"))):
        data_files[Split.TEST] = str(search_dir.joinpath(f"*test*.{ext}"))
    if len(data_files) == 0:
        raise ValueError(f"No data files found in {search_dir}.")
    return data_files


def reformat_and_filter_dataset(
    dataset: DatasetDict, dataset_name: str, strict: bool = False
) -> DatasetDict:
    reformat_fn: Callable[..., dict[str, Any]]
    map_fns: list[Callable[..., dict[str, Any]]] = []
    filter_fns: list[Callable[..., bool]] = []
    if dataset_name == "ja_wiki":
        reformat_fn = reformat_data("text")
        map_fns.append(remove_wikipedia_footnote())
        map_fns.append(remove_empty_parenthesis())
        filter_fns.append(is_not_empty())
    elif dataset_name == "en_wiki":
        reformat_fn = reformat_data("text")
        map_fns.append(remove_wikipedia_footnote())
        map_fns.append(remove_empty_parenthesis())
        filter_fns.append(is_not_empty())
    elif dataset_name == "ja_cc":
        reformat_fn = reformat_data("text")
        map_fns.append(extract_japanese_text())
        filter_fns.append(has_valid_domain())
        filter_fns.append(is_not_empty())
        filter_fns.append(is_japanese())
        filter_fns.append(is_not_ad_content())
        max_allowed_num: int = 2 if strict else 3
        filter_fns.append(is_not_adult_content(max_allowed_num))
        filter_fns.append(is_not_discrimination_content(max_allowed_num))
        filter_fns.append(is_not_violence_content(max_allowed_num))
        max_average_sentence_length: int = 80 if strict else 250
        filter_fns.append(has_good_average_sentence_length(max_average_sentence_length))
        min_score = 0.375 if strict else 0.30
        max_score = 0.70
        filter_fns.append(has_good_compression_ratio(min_score, max_score))
    elif dataset_name == "en_pile":
        reformat_fn = reformat_data("text")
        filter_fns.append(is_not_empty())
        filter_fns.append(lambda x: x["meta"]["pile_set_name"] != "Books3")
    elif dataset_name == "code_stack":
        reformat_fn = reformat_data("content")
        filter_fns.append(has_valid_extension())
        filter_fns.append(has_valid_max_line_length())
        filter_fns.append(has_valid_avg_line_length())
        filter_fns.append(has_valid_alphanum_fraction())
        filter_fns.append(is_not_empty())
    elif dataset_name == "en_slimpajama":
        reformat_fn = reformat_data("text")
        filter_fns.append(is_not_empty())
    elif dataset_name == "en_refinedweb":
        reformat_fn = reformat_data("content")
        filter_fns.append(is_not_empty())
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    dataset = dataset.map(reformat_fn, batched=False)
    train_dataset: typing.Union[Dataset, IterableDataset] = dataset["train"]
    if isinstance(train_dataset, Dataset):
        columns = list(train_dataset[0].keys())
    elif isinstance(train_dataset, IterableDataset):
        columns = list(train_dataset.take(1))[0].keys()
    else:
        raise ValueError
    dataset = dataset.map(remove_columns=list(set(columns) - {"text", "meta"}))
    for filter_fn in filter_fns:
        dataset = dataset.filter(filter_fn)
    for map_fn in map_fns:
        dataset = dataset.map(map_fn, batched=False)
    return dataset.filter(is_not_empty())


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=[
            # v1
            "ja_wiki",
            "en_wiki",
            "ja_cc",
            "en_pile",
            "code_stack",
            # v2
            "en_slimpajama",
            "en_refinedweb",
        ],
        help="Dataset name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to use strict filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="parquet",
        choices=["parquet", "json"],
        help="Whether to load the dataset from parquet or json.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="parquet",
        choices=["parquet", "json"],
        help="Whether to save the dataset to parquet or json.",
    )
    args = parser.parse_args()

    input_dir: pathlib.Path = pathlib.Path(args.input_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info("Loading the dataset")
    if args.input_format == "json":
        dataset: DatasetDict = load_dataset(
            "json",
            data_files=get_data_files(input_dir, "jsonl"),
            streaming=True,
        )
    elif args.input_format == "parquet":
        dataset: DatasetDict = load_dataset(
            "parquet",
            data_files=get_data_files(input_dir, "parquet"),
            streaming=True,
        )
    else:
        raise ValueError(f"Unknown input format: {args.input_format}.")

    dataset = reformat_and_filter_dataset(
        dataset, args.DATASET_NAME, strict=args.strict
    )

    ext: str
    if args.output_format == "json":
        ext = "jsonl"
    elif args.output_format == "parquet":
        ext = "parquet"
    else:
        raise ValueError(f"Unknown output format: {args.output_format}.")

    logger.info(f"Writing the reformatted data to {output_dir}.")
    for split, ds in dataset.items():
        chunk_index = 0
        for batch in tqdm.tqdm(ds.iter(batch_size=CHUNK_SIZE)):
            output_file: pathlib.Path = output_dir.joinpath(
                f"{split}_{chunk_index}.{ext}"
            )
            if output_file.exists() and not args.overwrite:
                logger.error(
                    f"{output_file} already exists. Specify --overwrite to overwrite."
                )
                chunk_index += 1
                continue

            if args.output_format == "json":
                Dataset.from_dict(batch).to_json(output_file, force_ascii=False)
            elif args.output_format == "parquet":
                Dataset.from_dict(batch).to_parquet(output_file)
            else:
                raise ValueError(f"Unknown output format: {args.output_format}.")
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
