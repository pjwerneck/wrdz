from pathlib import Path

from .base import base_compress
from .base import base_decompress
from .train import load_dictionary

HERE = Path(__file__).parent
DICTS = HERE / "dicts"

ENGLISH_CBOOK, ENGLISH_DBOOK = load_dictionary(DICTS / "en_US.dict")
URLS_CBOOK, URLS_DBOOK = load_dictionary(DICTS / "urls.dict")


def compress(text: str) -> bytes:
    """Compress text using English-optimized dictionary.

    Args:
        text: Input text to compress

    Returns:
        Compressed bytes
    """
    return base_compress(text, ENGLISH_CBOOK)


def decompress(data: bytes) -> str:
    """Decompress data using English-optimized dictionary.

    Args:
        data: Compressed bytes to decompress

    Returns:
        Original text
    """
    return base_decompress(data, ENGLISH_DBOOK)


def compress_urls(text: str) -> bytes:
    """Compress URL text using URL-optimized dictionary.

    Args:
        text: Input URL to compress

    Returns:
        Compressed bytes
    """
    return base_compress(text, URLS_CBOOK)


def decompress_urls(data: bytes) -> str:
    """Decompress URL data using URL-optimized dictionary.

    Args:
        data: Compressed bytes to decompress

    Returns:
        Original URL text
    """
    return base_decompress(data, URLS_DBOOK)


__all__ = [
    "compress",
    "decompress",
    "compress_urls",
    "decompress_urls",
]
