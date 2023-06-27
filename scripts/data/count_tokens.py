import json
import logging
import pathlib
from argparse import ArgumentParser

import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the wikipedia data directory.",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    for file_path in data_dir.glob("*.jsonl"):
        token_count = 0
        logger.info(f"Counting tokens in {file_path.stem}.")
        with file_path.open("r") as fin:
            for line in tqdm.tqdm(fin.readlines()):
                row: dict = json.loads(line)
                token_count += len(row["tokens"])
        logger.info(f"{file_path.stem} has {token_count} tokens.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
