from pathlib import Path

import msgpack

from wrdz import train_dictionary

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "datasets"
DICTS = ROOT / "src" / "wrdz" / "dicts"


def train_english_news_dict():
    with open(DATASETS / "en_US" / "train.txt.bkp") as f:
        news = f.read()

    cbook, dbook = train_dictionary(
        news,
        max_sub_len=4,
        dict_size=16384,
    )
    with open(DICTS / "en_US.dict", "wb") as f:
        msgpack.dump((cbook, dbook), f)


def train_urls_dict():
    with open(DATASETS / "train_urls.txt") as f:
        urls = f.read()

    cbook, dbook = train_dictionary(
        urls,
        max_sub_len=4,
        dict_size=16384,
    )
    with open(DICTS / "urls.dict", "wb") as f:
        msgpack.dump((cbook, dbook), f)


if __name__ == "__main__":
    # train_english_news_dict()
    train_urls_dict()
