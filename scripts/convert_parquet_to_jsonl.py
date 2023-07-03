import io
import logging
import pathlib
from argparse import ArgumentParser

from datasets import Dataset, IterableDataset, IterableDatasetDict, load_dataset
from datasets.splits import Split
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_data_files(search_dir: pathlib.Path, ext: str) -> dict[Split, pathlib.Path]:
    train_files = list(search_dir.glob(f"*train*.{ext}"))
    valid_files = list(search_dir.glob(f"*valid*.{ext}"))
    test_files = list(search_dir.glob(f"*test*.{ext}"))
    assert len(train_files) == 1, f"Found {len(train_files)} train files."
    assert len(valid_files) <= 1, f"Found {len(valid_files)} valid files."
    assert len(test_files) <= 1, f"Found {len(test_files)} test files."
    data_files = {Split.TRAIN: train_files[0]}
    if len(valid_files) == 1:
        data_files[Split.VALIDATION] = valid_files[0]
    if len(test_files) == 1:
        data_files[Split.TEST] = test_files[0]
    return data_files


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
    args = parser.parse_args()

    data_dir: pathlib.Path = pathlib.Path(args.data_dir)
    output_dir: pathlib.Path = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading the dataset")
    dataset: IterableDatasetDict = load_dataset(
        "parquet",
        data_files={k: str(v) for k, v in get_data_files(data_dir, "parquet").items()},
        streaming=True,
    )

    logger.info(f"Writing the dataset to {output_dir} in JSONL format.")
    ds: IterableDataset
    for split, ds in dataset.items():
        output_file: pathlib.Path = output_dir.joinpath(f"{split}.jsonl")
        if output_file.exists() and not args.overwrite:
            logger.error(
                f"{output_file} already exists. Specify --overwrite to continue."
            )
            exit(1)
        with output_file.open(mode="wb") as f:
            for batch in tqdm(ds.iter(batch_size=100)):
                with io.BytesIO() as bf:
                    Dataset.from_dict(batch).to_json(bf, force_ascii=False)
                    f.write(bf.getvalue())

        logger.info(f"Finished exporting the {split} split.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
