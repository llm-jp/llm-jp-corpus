import json
import logging
import pathlib
from argparse import ArgumentParser

import tqdm

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

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

    logger.info(f"Reformatting the data in {args.data_dir}.")
    data_dir = pathlib.Path(args.data_dir)
    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Reformatting {file_path.stem}.")
        source, language, timestamp, _ = file_path.stem.split("_")
        output_file_name = f"{file_path.stem}_reformatted.jsonl"
        output_file = pathlib.Path(args.output_dir).joinpath(output_file_name)
        if output_file.exists() and not args.overwrite:
            logger.info(f"{output_file} already exists. Skipping.")
            continue
        with file_path.open("r") as fin:
            with output_file.open("wt") as fout:
                for line in tqdm.tqdm(fin):
                    row = json.loads(line)
                    reformatted_row = {
                        "text": row["text"],
                        "meta": {
                            "title": row["title"],
                            "url": row["url"],
                            "language": language,
                            "timestamp": timestamp,
                            "source": source,
                        },
                    }
                    fout.write(json.dumps(reformatted_row, ensure_ascii=False) + "\n")
                logger.info(f"Finished reformatting {file_path.stem}.")


if __name__ == "__main__":
    main()
