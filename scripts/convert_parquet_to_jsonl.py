import logging
import pathlib
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, disable_caching
from tqdm import tqdm
from utils import list_input_files

logger = logging.getLogger(__name__)
disable_caching()


def process_file(
    input_file: pathlib.Path, output_dir: pathlib.Path, overwrite: bool
) -> None:
    logger.info(f"Writing the dataset to {output_dir} in JSONL format.")
    output_file: pathlib.Path = output_dir.joinpath(f"{input_file.stem}.jsonl")
    if output_file.exists() and not overwrite:
        logger.error(f"{output_file} already exists. Specify --overwrite to overwrite.")
        return
    dataset: Dataset = Dataset.from_parquet(str(input_file), keep_in_memory=True)
    dataset.to_json(output_file, force_ascii=False)
    logger.info(f"Finished exporting to {output_file}.")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        help="Path(s) to the input data directory or file.",
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
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes for parallel execution.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.num_proc) as executor:
        for input_file in tqdm(list_input_files(args.input_path)):
            executor.submit(process_file, input_file, output_dir, args.overwrite)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
