import logging
from typing import Any

logger = logging.getLogger(__name__)


DATASET_NAME = "bigcode/the-stack"


def filter_and_reformat(example) -> dict[str, Any]:
    return {
        "text": example["content"],
        "meta": {
            **{k: v for k, v in example.items() if k != "content"},
            "source": DATASET_NAME.split("/")[-1],
        },
    }
