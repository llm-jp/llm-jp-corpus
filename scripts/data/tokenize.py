import logging
import pathlib
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def tokenize(data_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    pass


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
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists.")
    output_dir.mkdir(parents=True)

    logger.info(f"Tokenize the data in {args.data_dir}.")
    tokenize(data_dir, output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
