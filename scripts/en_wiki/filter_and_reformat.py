import logging
from typing import Any

logger = logging.getLogger(__name__)


DATASET_NAME = "wikipedia"
LANGUAGE = "en"
DUMP_DATE = "20230320"


def filter_and_reformat(examples) -> dict[str, Any]:
    return {
        "text": examples["text"],
        "title": examples["title"],
        "url": examples["url"],
        "language": [LANGUAGE] * len(examples["text"]),
        "timestamp": [DUMP_DATE] * len(examples["text"]),
        "source": [DATASET_NAME] * len(examples["text"]),
    }
