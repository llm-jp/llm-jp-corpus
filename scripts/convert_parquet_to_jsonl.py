import logging
import pathlib
from argparse import ArgumentParser

from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        "--overwrite",
        action="store_true",
        help="Whether to overwrite the output directory.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading the dataset")
    for input_file in tqdm(data_dir.glob("*.parquet")):
        logger.info(f"Loading {input_file}.")
        dataset: Dataset = Dataset.from_parquet(str(input_file))

        logger.info(f"Writing the dataset to {output_dir} in JSONL format.")
        output_file: pathlib.Path = output_dir.joinpath(f"{input_file.stem}.jsonl")
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to continue."
            )
            exit(1)
        dataset.to_json(output_file, force_ascii=False)
        logger.info(f"Finished exporting to {output_file}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
