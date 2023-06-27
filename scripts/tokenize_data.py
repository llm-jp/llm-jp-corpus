import json
import logging
import pathlib
from argparse import ArgumentParser

import joblib
import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _tokenize(line: str, tokenizer: PreTrainedTokenizer) -> dict:
    row: dict = json.loads(line)
    text = row["text"]
    tokens = tokenizer.tokenize(text)
    row["tokens"] = tokens
    row["tokenizer_name"] = tokenizer.name_or_path
    return row


def tokenize(data_dir: pathlib.Path, output_dir: pathlib.Path, model_name: str) -> None:
    logger.info("Initialize the tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for file_path in data_dir.glob("*.jsonl"):
        logger.info(f"Tokenizing {file_path.stem}.")
        with file_path.open("r") as fin:
            lines: list[str] = fin.readlines()
            rows: list[dict] = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(_tokenize)(line, tokenizer) for line in tqdm.tqdm(lines)
            )
        output_file_name = f"{file_path.stem}_filtered.jsonl"
        output_file = output_dir.joinpath(output_file_name)
        logger.info(f"Writing the reformatted data to {output_file}.")
        with output_file.open("wt") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info(f"Finished reformatting {file_path.stem}.")


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
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} already exists.")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tokenize the data in {args.data_dir}.")
    tokenize(data_dir, output_dir, args.model_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
