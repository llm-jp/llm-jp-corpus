import logging
import pathlib
from collections.abc import Iterator
from typing import Literal

logger = logging.getLogger(__name__)


def list_input_files(
    input_paths: list[str],
    input_format: Literal["parquet", "jsonl"] = "parquet",
) -> Iterator[pathlib.Path]:
    for path_str in input_paths:
        path = pathlib.Path(path_str)
        if path.exists() is False:
            logger.warning(f"{path} not found and skipped")
            continue
        yield from path.glob(f"*.{input_format}") if path.is_dir() else [path]
