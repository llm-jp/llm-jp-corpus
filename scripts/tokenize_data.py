import json
import logging
import pathlib
from argparse import ArgumentParser
from multiprocessing import Pool

import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

tokenizer: PreTrainedTokenizer


def tokenize(line: str) -> dict:
    row: dict = json.loads(line)
    text = row["text"]
    tokens = tokenizer.tokenize(text)
    row["tokens"] = tokens
    row["tokenizer_name"] = tokenizer.name_or_path
    return row


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
        "--model_name",
        type=str,
        default="cyberagent/open-calm-7b",  # TODO: Update the default model name.
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

    logger.info("Initialize the tokenizer.")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info(f"Tokenize the data in {args.data_dir}.")
    for file_path in data_dir.glob("*.jsonl"):
        output_file_name = f"{file_path.stem}_tokenized.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        if output_dir.exists() and not args.overwrite:
            logger.warning(f"{output_file} already exists. Skip this file.")
            continue
        logger.info(f"Tokenizing {file_path.stem}.")
        with file_path.open("r") as fin:
            lines: list[str] = fin.readlines()
            with Pool() as p:
                rows: list[dict] = []
                # Do not use imap_unordered because the order of the lines must
                # be preserved for reproducibility.
                for row in tqdm.tqdm(p.imap(tokenize, lines)):
                    rows.append(row)
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
