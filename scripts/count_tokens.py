import logging
import os
import pathlib
from argparse import ArgumentParser

from datasets import Dataset, disable_caching
from tqdm import tqdm

logger = logging.getLogger(__name__)
disable_caching()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the wikipedia data directory.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    token_counts: dict[str, int] = {}
    for file_path in tqdm(data_dir.glob("*.parquet")):
        dataset = Dataset.from_parquet(str(file_path))
        logger.info(f"Counting tokens in {file_path.stem}.")
        dataset.remove_columns(["text", "meta"])
        dataset = dataset.map(
            lambda example: {
                "num_tokens": len(example["tokens"]),
            },
            batched=False,
            num_proc=os.cpu_count(),
        )
        token_count = sum(dataset["num_tokens"])
        logger.info(f"{file_path.stem} has {token_count:,} tokens.")
        token_counts[file_path.stem] = token_count
    logger.info(f"Total number of shards: {len(token_counts):,}.")
    logger.info(f"Total number of tokens: {sum(token_counts.values()):,}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
