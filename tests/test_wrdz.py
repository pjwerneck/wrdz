from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.provisional import urls as st_urls

from wrdz import compress
from wrdz import compress_urls
from wrdz import decompress
from wrdz import decompress_urls

HERE = Path(__file__).parent


@given(st.text(min_size=5, max_size=140, alphabet=st.characters(codec="ascii")))
def test_random_ascii_strings(text):
    compressed = compress(text)
    decompressed = decompress(compressed)
    assert decompressed == text


@given(st.text(min_size=5, max_size=140, alphabet=st.characters(codec="utf-8")))
def test_random_utf8_strings(text):
    compressed = compress(text)
    decompressed = decompress(compressed)
    assert decompressed == text


@given(st_urls())
def test_random_urls(text):
    compressed = compress_urls(text)
    decompressed = decompress_urls(compressed)
    assert decompressed == text


def test_english_text():
    with open(HERE / "english.txt") as f:
        text = f.read()
    for line in text.splitlines():
        compressed = compress(line)
        decompressed = decompress(compressed)
        assert decompressed == line


def test_urls_test_dataset():
    with open(HERE / "urls.txt") as f:
        text = f.read()
    for line in text.splitlines():
        compressed = compress_urls(line)
        decompressed = decompress_urls(compressed)
        assert decompressed == line
