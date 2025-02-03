from pathlib import Path

from wrdz.base import base_compress
from wrdz.base import base_decompress
from wrdz.train import train_dictionary

ROOT = Path(__file__).parent.parent
TRAIN_FILE = ROOT / "datasets" / "en_US" / "train.txt"


def test_train_dictionary():
    # train with the first 10k of the English training data, text with the next
    # 140 characters
    with open(TRAIN_FILE) as f:
        train = f.read(10000)
        text = f.read(140)

    cbook, dbook = train_dictionary(train, max_sub_len=4, dict_size=16384)

    compressed = base_compress(text, cbook)
    decompressed = base_decompress(compressed, dbook)
    assert decompressed == text
