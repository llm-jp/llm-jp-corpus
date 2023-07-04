import logging
import os
import pathlib
import time
from argparse import ArgumentParser
from typing import Any

import sentencepiece as spm
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

sentence_piece_processor: spm.SentencePieceProcessor


def tokenize_function(examples) -> dict[str, Any]:
    return {
        "tokens": [
            sentence_piece_processor.encode_as_pieces(text) for text in examples["text"]
        ],
    }


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
        "--sentencepiece_model",
        type=str,
        required=True,
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

    start_time = time.time()

    logger.info("Initialize the tokenizer.")
    global sentence_piece_processor
    sentence_piece_processor = spm.SentencePieceProcessor(args.sentencepiece_model)

    logger.info("Loading the dataset")
    for input_file in tqdm(data_dir.glob("*.parquet")):
        output_file: pathlib.Path = output_dir.joinpath(input_file.name)
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to overwrite."
            )
            continue

        logger.info(f"Loading {input_file}.")
        dataset: Dataset = Dataset.from_parquet(str(input_file))
        logger.info("Tokenizing the dataset.")
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=128,
            num_proc=os.cpu_count(),
        )
        logger.info("Finished tokenizing the dataset.")

        logger.info(f"Writing the tokenized data to {output_dir}.")
        dataset.to_parquet(output_file)
        logger.info(f"Finished writing the tokenized to {output_file}.")

    end_time = time.time()
    logger.info(
        f"Finished tokenizing the dataset. Elapsed time: {end_time - start_time} [sec]"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
