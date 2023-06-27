import json
import logging
import pathlib

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


def filter_and_reformat(file_path: pathlib.Path, output_file: pathlib.Path) -> None:
    logger.info(f"Reformatting {file_path.stem}.")
    source, language, timestamp, _ = file_path.stem.split("_")
    with file_path.open("r") as fin:
        lines: list[str] = fin.readlines()
        rows: list[dict] = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(reformat)(line, language, timestamp, source)
            for line in tqdm.tqdm(lines)
        )
    logger.info(f"Writing the reformatted data to {output_file}.")
    with output_file.open("wt") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(f"Finished reformatting {file_path.stem}.")
