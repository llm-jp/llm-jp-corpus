"""Script to download English C4 data.

Note:
    This script is partly borrowed from the following repository,
    which is distributed under the Apache License 2.0.
        https://github.com/togethercomputer/RedPajama-Data
"""
import argparse
import gzip
import json
import logging
import pathlib
from datetime import datetime

import joblib
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    return datetime.now().isoformat()


def process_record(record):
    return {
        "text": record["text"],
        "meta": {
            "timestamp": record["timestamp"],
            "url": record["url"],
            "language": "en",
            "source": "c4",
        },
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/c4/en")
    parser.add_argument("--output_dir", type=str, default="./data/c4/processed_en")
    args = parser.parse_args()

    num_cpus: int = joblib.cpu_count()
    logger.info(f"Using {num_cpus} processes")

    out_dir: pathlib.Path = pathlib.Path(args.output_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    records_files: list[pathlib.Path] = list(data_dir.glob("*.json.gz"))

    def process_file(fp):
        logger.info(f"Start processing {fp}...")
        out_fp = out_dir / fp.with_suffix("").name.replace("json", "jsonl")

        with gzip.open(fp, "r") as in_f:
            records = [json.loads(line) for line in in_f.readlines()]

        with open(out_fp, "w") as out_f:
            for record in records:
                record = process_record(record)
                if record is not None:
                    out_f.write(json.dumps(record) + "\n")

        logger.info(f"Done processing {fp}...")

    Parallel(n_jobs=num_cpus)(delayed(process_file)(fp) for fp in records_files)
