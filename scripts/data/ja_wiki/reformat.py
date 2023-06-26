import json
import logging
import pathlib
from argparse import ArgumentParser

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

    data_dir = pathlib.Path(args.data_dir)
    for file_path in data_dir.glob("*.jsonl"):
        output_file_name = f"{file_path.stem}_reformatted.jsonl"
        output_path = pathlib.Path(args.output_dir).joinpath(output_file_name)
        with file_path.open("r") as fin:
            with output_path.open("wt") as fout:
                _, language, timestamp, _ = file_path.stem.split("_")
                for line in fin:
                    row = json.loads(line)
                    reformatted_row = {
                        "text": row["text"],
                        "meta": {
                            "title": row["title"],
                            "url": row["url"],
                            "language": language,
                            "timestamp": timestamp,
                        },
                    }
                    fout.write(json.dumps(reformatted_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
