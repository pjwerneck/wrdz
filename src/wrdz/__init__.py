from pathlib import Path

from .base import base_compress
from .base import base_decompress
from .train import load_dictionary

HERE = Path(__file__).parent
DICTS = HERE / "dicts"

ENGLISH_CBOOK, ENGLISH_DBOOK = load_dictionary(DICTS / "en_US.dict")
URLS_CBOOK, URLS_DBOOK = load_dictionary(DICTS / "urls.dict")


def compress(text: str) -> bytes:
    return base_compress(text, ENGLISH_CBOOK)


def decompress(data: bytes) -> str:
    return base_decompress(data, ENGLISH_DBOOK)


def compress_urls(text: str) -> bytes:
    return base_compress(text, URLS_CBOOK)


def decompress_urls(data: bytes) -> str:
    return base_decompress(data, URLS_DBOOK)


__all__ = [
    "compress",
    "decompress",
    "compress_urls",
    "decompress_urls",
]
