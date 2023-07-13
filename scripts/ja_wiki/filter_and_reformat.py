import logging
from typing import Any

logger = logging.getLogger(__name__)

DATASET_NAME = "wikipedia"
LANGUAGE = "ja"
DUMP_DATE = "20230320"


def filter_and_reformat(example) -> dict[str, Any]:
    return {
        "text": example["text"],
        "meta": {
            "title": example["title"],
            "url": example["url"],
            "language": LANGUAGE,
            "timestamp": DUMP_DATE,
            "source": DATASET_NAME,
        },
    }
