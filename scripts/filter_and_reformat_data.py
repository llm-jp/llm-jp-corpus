import logging
import pathlib
from argparse import ArgumentParser

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
        output_file_name = f"{file_path.stem}_filtered.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        if output_file.exists() and not args.overwrite:
            logger.warning(f"{output_file} already exists. Skip this file.")
            continue

        if args.DATASET_NAME == "ja_wiki":
            filter_ja_wiki(file_path, output_file)
        elif args.DATASET_NAME == "en_wiki":
            filter_en_wiki(file_path, output_file)
        elif args.DATASET_NAME == "ja_cc":
            filter_ja_cc(file_path, output_file)
        elif args.DATASET_NAME == "en_pile":
            filter_en_pile(file_path, output_file)
        elif args.DATASET_NAME == "code_stack":
            filter_code_stack(file_path, output_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
