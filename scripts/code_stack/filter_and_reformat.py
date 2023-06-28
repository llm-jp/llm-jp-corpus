import json
import logging

logger = logging.getLogger(__name__)


DATASET_NAME = "bigcode/the-stack"


def filter_and_reformat(line: str) -> dict:
    row: dict = json.loads(line)
    return {
        "text": row["content"],
        "meta": {
            **{k: v for k, v in row.items() if k != "content"},
            "source": DATASET_NAME.split("/")[-1],
        },
    }
