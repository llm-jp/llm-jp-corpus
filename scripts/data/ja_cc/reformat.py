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
    args = parser.parse_args()

    logger.info(f"Reformatting the data in {args.data_dir}.")
    data_dir = pathlib.Path(args.data_dir)
    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Reformatting {file_path.stem}.")
        output_file_name = f"{file_path.stem}_reformatted.jsonl"
        output_path = pathlib.Path(args.output_dir).joinpath(output_file_name)
        with file_path.open("r") as fin:
            with output_path.open("wt") as fout:
                for line in tqdm.tqdm(fin):
                    row = json.loads(line)
                    reformatted_row = {
                        "text": row["text"],
                        "meta": {
                            "url": row["url"],
                            "language": "ja",
                            "timestamp": row["timestamp"],
                            "source": "mc4",
                        },
                    }
                    fout.write(json.dumps(reformatted_row, ensure_ascii=False) + "\n")
                logger.info(f"Finished reformatting {file_path.stem}.")


if __name__ == "__main__":
    main()
