import logging
from argparse import ArgumentParser
from enum import Enum

from datasets import Dataset, DatasetDict, disable_caching, load_dataset
from filter_data import reformat_and_filter_dataset

logger = logging.getLogger(__name__)
disable_caching()


class Label(Enum):
    """Label for instances."""

    ACCEPTABLE = "0"
    HARMFUL = "1"
    LOW_QUALITY = "2"


def get_stats(dataset: Dataset, filtered_dataset: Dataset) -> dict[str, float]:
    # TODO: Fine-grained evaluation
    def get_labels(ds: Dataset) -> list[str]:
        return [meta["label"] for meta in ds["meta"]]

    tp = (
        get_labels(dataset).count(Label.HARMFUL.value)
        - get_labels(filtered_dataset).count(Label.HARMFUL.value)
        + get_labels(dataset).count(Label.LOW_QUALITY.value)
        - get_labels(filtered_dataset).count(Label.LOW_QUALITY.value)
    )
    fp = get_labels(dataset).count(Label.ACCEPTABLE.value) - get_labels(
        filtered_dataset
    ).count(Label.ACCEPTABLE.value)
    fn = get_labels(filtered_dataset).count(Label.HARMFUL.value) + get_labels(
        filtered_dataset
    ).count(Label.LOW_QUALITY.value)
    pre = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0.0

    acceptable = get_labels(filtered_dataset).count(Label.ACCEPTABLE.value)
    harmful = get_labels(filtered_dataset).count(Label.HARMFUL.value)
    low_quality = get_labels(filtered_dataset).count(Label.LOW_QUALITY.value)
    return {
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "acceptable": acceptable,
        "harmful": harmful,
        "low_quality": low_quality,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "DATASET_NAME",
        type=str,
        choices=["ja_wiki", "en_wiki", "ja_cc", "en_pile", "code_stack"],
        help="Dataset name",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to the data directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to use strict filtering.",
    )
    args = parser.parse_args()

    logger.info("Loading the dataset")
    dataset: DatasetDict = load_dataset("json", data_files=args.input_path)
    filtered_dataset = reformat_and_filter_dataset(
        dataset, args.DATASET_NAME, strict=args.strict
    )

    assert "train" in dataset.keys() and "train" in filtered_dataset.keys()
    stats = get_stats(dataset["train"], filtered_dataset["train"])

    print(f"- Precision: {stats['precision']:.3f}")
    print(f"- Recall: {stats['recall']:.3f}")
    print(f"- F1: {stats['f1']:.3f}")
    size = len(filtered_dataset["train"])
    print(f"- Acceptable: {stats['acceptable']} ({stats['acceptable'] / size:.3f})")
    print(f"- Harmful: {stats['harmful']} ({stats['harmful'] / size:.3f})")
    print(f"- Low quality: {stats['low_quality']} ({stats['low_quality'] / size:.3f})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
