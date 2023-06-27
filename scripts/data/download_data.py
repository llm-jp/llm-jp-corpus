import argparse
import logging
import pathlib

from code_stack.download import download as download_code_stack
from en_pile.download import download as download_en_pile
from en_wiki.download import download as download_en_wiki
from ja_cc.download import download as download_ja_cc
from ja_wiki.download import download as download_ja_wiki

logger = logging.getLogger(__name__)

DATASET_NAME = "wikipedia"
LANGUAGE = "ja"
DUMP_DATE = "20230320"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the wikipedia data directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing files.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists.")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.DATASET_NAME == "ja_wiki":
        download_ja_wiki(output_dir)
    elif args.DATASET_NAME == "en_wiki":
        download_en_wiki(output_dir)
    elif args.DATASET_NAME == "ja_cc":
        download_ja_cc(output_dir)
    elif args.DATASET_NAME == "en_pile":
        download_en_pile(output_dir)
    elif args.DATASET_NAME == "code_stack":
        download_code_stack(output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
