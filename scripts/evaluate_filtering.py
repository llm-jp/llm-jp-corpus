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


def get_scores(dataset: Dataset, filtered_dataset: Dataset) -> dict[str, float]:
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
    return {"precision": pre, "rec": rec, "f1": f1}


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
    args = parser.parse_args()

    logger.info("Loading the dataset")
    dataset: DatasetDict = load_dataset("json", data_files=args.input_path)
    filtered_dataset = reformat_and_filter_dataset(dataset, args.DATASET_NAME)

    assert "train" in dataset.keys() and "train" in filtered_dataset.keys()
    scores = get_scores(dataset["train"], filtered_dataset["train"])

    print(f"- Precision: {scores['precision']:.3f}")
    print(f"- Recall: {scores['recall']:.3f}")
    print(f"- F1: {scores['f1']:.3f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
