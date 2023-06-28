import json
import logging
import pathlib
from argparse import ArgumentParser

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
    parser.add_argument(
        "--token_size",
        type=int,
        default=-1,
        help="Token size (negative values mean no limit).",
    )
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    for file_path in data_dir.glob("*.jsonl"):
        output_file_name = f"{file_path.stem}_sampled.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        if output_file.exists() and not args.overwrite:
            logger.warning(f"{output_file} already exists. Skip this file.")
            continue

        if output_file.is_symlink():
            output_file.unlink()

        if args.token_size < 0:
            logger.info(f"Create a symbolic link to {file_path}.")
            # Create a symbolic link.
            output_file.symlink_to(file_path)
            continue

        token_count: int = 0
        rows: list[dict] = []
        logger.info(f"Extract data with {args.token_size} tokens.")
        with file_path.open("rt") as fin:
            for line in fin:
                row = json.loads(line)
                token_count += len(row["tokens"].split())
                if token_count > args.token_size:
                    break
                rows.append(row)
            else:
                logger.warning(
                    f"The total number of tokens {token_count:,} was less than {args.token_size:,}."
                )

        with output_file.open("wt") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Finished reformatting {file_path.stem}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
