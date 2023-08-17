import argparse
import json
import logging
import pathlib

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        help="Path to the output data directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing files.",
    )
    args = parser.parse_args()

    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already exists. Specify --overwrite to overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.DATASET_NAME == "ja_wiki":
        dataset = load_dataset(
            "wikipedia",
            language="ja",
            date="20230720",
            beam_runner="DirectRunner",
        )
    elif args.DATASET_NAME == "en_wiki":
        dataset = load_dataset(
            "wikipedia",
            language="en",
            date="20230720",
            beam_runner="DirectRunner",
        )
    elif args.DATASET_NAME == "ja_cc":
        dataset = load_dataset(
            "mc4",
            languages=["ja"],
            streaming=True,
        )
    elif args.DATASET_NAME == "en_pile":
        dataset = load_dataset(
            "EleutherAI/pile",
            streaming=True,
        )
    elif args.DATASET_NAME == "code_stack":
        dataset = load_dataset(
            "bigcode/the-stack",
            streaming=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.DATASET_NAME}")

    for split, ds in dataset.items():
        file_path: pathlib.Path = output_dir.joinpath(f"{split}.jsonl")
        num_examples: int = 0
        for example in tqdm(ds):
            num_examples += 1
            with file_path.open(mode="a") as f:
                json.dump(example, f, ensure_ascii=False)
                f.write("\n")
        logger.info(
            f"Finished downloading the {split} split. "
            f"There are total {num_examples} examples."
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
