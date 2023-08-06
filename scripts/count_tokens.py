import logging
import os
import pathlib
from argparse import ArgumentParser
from collections.abc import Iterator

from datasets import Dataset, disable_caching
from tqdm import tqdm

logger = logging.getLogger(__name__)
disable_caching()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    input_files = sorted(list_input_files(args.input_path))

    token_counts: dict[str, int] = {}
    for input_file in tqdm(input_files):
        logger.info(f"Loading dataset from {input_file}.")
        dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
        logger.info(f"Counting tokens in {input_file.stem}.")
        if "num_tokens" not in dataset.column_names:
            dataset = dataset.map(
                lambda example: {
                    "num_tokens": len(example["tokens"]),
                },
                batched=False,
                keep_in_memory=True,
                num_proc=os.cpu_count() if args.num_proc == -1 else args.num_proc,
            )
            dataset.to_parquet(input_file)
        token_count = sum(dataset["num_tokens"])
        logger.info(f"{input_file.stem} has {token_count:,} tokens.")
        token_counts[input_file.stem] = token_count
    logger.info(f"Total number of shards: {len(token_counts):,}.")
    logger.info(f"Total number of tokens: {sum(token_counts.values()):,}.")


def list_input_files(input_paths: list[str]) -> Iterator[pathlib.Path]:
    for path_str in input_paths:
        path = pathlib.Path(path_str)
        if path.exists() is False:
            logger.warning(f"{path} not found and skipped")
            continue
        yield from path.glob("*.parquet") if path.is_dir() else [path]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
