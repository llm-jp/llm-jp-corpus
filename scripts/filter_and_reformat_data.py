import json
import logging
import pathlib
from argparse import ArgumentParser

import joblib
import tqdm
from code_stack.filter_and_reformat import filter_and_reformat as filter_code_stack
from en_pile.filter_and_reformat import filter_and_reformat as filter_en_pile
from en_wiki.filter_and_reformat import filter_and_reformat as filter_en_wiki
from ja_cc.filter_and_reformat import filter_and_reformat as filter_ja_cc
from ja_wiki.filter_and_reformat import filter_and_reformat as filter_ja_wiki

logger = logging.getLogger(__name__)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
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

    logger.info(f"Reformatting the data in {args.data_dir}.")
    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Reformatting {file_path.stem}.")

        output_file_name = f"{file_path.stem}_filtered.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        if output_file.exists() and not args.overwrite:
            logger.warning(f"{output_file} already exists. Skip this file.")
            continue

        if args.DATASET_NAME == "ja_wiki":
            filter_fn = filter_ja_wiki
        elif args.DATASET_NAME == "en_wiki":
            filter_fn = filter_en_wiki
        elif args.DATASET_NAME == "ja_cc":
            filter_fn = filter_ja_cc
        elif args.DATASET_NAME == "en_pile":
            filter_fn = filter_en_pile
        elif args.DATASET_NAME == "code_stack":
            filter_fn = filter_code_stack
        else:
            raise ValueError(f"Unknown dataset name: {args.DATASET_NAME}.")

        with file_path.open("r") as fin:
            lines: list[str] = fin.readlines()
            rows: list[dict] = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(filter_fn)(line) for line in tqdm.tqdm(lines)
            )
        logger.info(f"Writing the reformatted data to {output_file}.")
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
