import logging
import os
from logging import getLogger

import requests
from tqdm import tqdm

logger = getLogger(__name__)

DATA_DIR = os.getenv("DATA_DIR", "data/ja_wiki")
DUMP_DATE = "20230601"
URL = f"https://dumps.wikimedia.org/jawiki/{DUMP_DATE}/jawiki-{DUMP_DATE}-pages-articles.xml.bz2"


def download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_size: int = int(requests.head(url).headers["Content-Length"])
    pbar: tqdm = tqdm(total=file_size, unit="B", unit_scale=True)
    with open(path, "wb") as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    logger.info(f"Downloading {URL} to {DATA_DIR}")
    download(URL, os.path.join(DATA_DIR, "jawiki.xml.bz2"))
    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()
