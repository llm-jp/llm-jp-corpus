import json
import logging

logger = logging.getLogger(__name__)

DATASET_NAME = "wikipedia"
LANGUAGE = "ja"
DUMP_DATE = "20230320"


def filter_and_reformat(line: str) -> dict:
    row: dict = json.loads(line)
    return {
        "text": row["text"],
        "meta": {
            "title": row["title"],
            "url": row["url"],
            "language": LANGUAGE,
            "timestamp": DUMP_DATE,
            "source": DATASET_NAME,
        },
    }
