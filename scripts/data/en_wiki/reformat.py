import json
import logging
import pathlib
from argparse import ArgumentParser

import joblib
import tqdm

logger = logging.getLogger(__name__)


def reformat(line: str, language: str, timestamp: str, source: str) -> dict:
    row: dict = json.loads(line)
    return {
        "text": row["text"],
        "meta": {
            "title": row["title"],
            "url": row["url"],
            "language": language,
            "timestamp": timestamp,
            "source": source,
        },
    }


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
            lines: list[str] = fin.readlines()
            rows: list[dict] = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(reformat)(line, language, timestamp, source)
                for line in tqdm.tqdm(lines)
            )

        with output_file.open("wt") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Finished reformatting {file_path.stem}.")


if __name__ == "__main__":
    main()
