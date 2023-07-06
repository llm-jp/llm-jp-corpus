import logging
from typing import Any

logger = logging.getLogger(__name__)


DATASET_NAME = "EleutherAI/pile"


def filter_and_reformat(example) -> dict[str, Any]:
    return {
        "text": example["text"],
        "meta": {
            **{k: v for k, v in example.items() if k != "text"},
            "source": DATASET_NAME.split("/")[-1],
        },
    }
