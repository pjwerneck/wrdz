# wrdz
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/wrdz.svg)](https://pypi.org/project/wrdz/)
[![PyPI version](https://badge.fury.io/py/wrdz.svg)](https://badge.fury.io/py/wrdz)

Dictionary-based compression for short text strings and URLs.

## Key Features

- **Optimized for Short Text**: Designed for strings under 140 characters
- **URL Compression**: Special dictionary trained on URL patterns
- **UTF-8 Support**: Handles any valid UTF-8 text
- **Domain Adaptable**: Train custom dictionaries for your use case
- **Pure Python**: No external dependencies for runtime

## Quick Start

```bash
pip install wrdz
```

```python
from wrdz import compress, decompress

# English text compression
text = "The quick brown fox jumps over the lazy dog"
compressed = compress(text)
print(f"Original: {len(text)} bytes")
print(f"Compressed: {len(compressed)} bytes")
print(f"Ratio: {len(compressed)/len(text):.2%}")

# URL compression
from wrdz import compress_urls, decompress_urls

url = "https://github.com/pjwerneck/wrdz"
compressed = compress_urls(url)
print(f"Original: {len(url)} bytes")
print(f"Compressed: {len(compressed)} bytes")
print(f"Ratio: {len(compressed)/len(url):.2%}")
```


## Training Custom Dictionaries

```python
from wrdz.train import train_dictionary, save_dictionary

# Train on domain-specific text
cbook, dbook = train_dictionary(
    text="your training data",
    max_sub_len=4,    # Max sequence length
    dict_size=8192    # Dictionary entries
)

# Save for reuse
save_dictionary(cbook, dbook, "domain.dict")

# Use in compression
from wrdz.base import base_compress, base_decompress

compressed = base_compress("text", cbook)
original = base_decompress(compressed, dbook)
```


## Compression Benchmarks

The tables below show compression ratios for different dictionary sizes and
maximum sequence lengths. The `Δ%` column shows the improvement over the
baseline compression ratio.

### US English

The en_US dictionary is trained on a 1M lines subset of the
[cnn_dailymail](https://huggingface.co/datasets/ccdv/cnn_dailymail) dataset.

| Dict Size | Max Seq | Short wrdz | Short smaz | Short Δ%  | Long wrdz | Long smaz | Long Δ%  |
|----------:|---------|-----------:|-----------:|----------:|----------:|----------:|---------:|
|     16384 |       4 |     0.671  |     0.907  |     +26.0 |     0.521 |     0.621 |    +16.1 |
|      8192 |       4 |     0.704  |     0.907  |     +22.4 |     0.526 |     0.621 |    +15.4 |
|      4096 |       4 |     0.736  |     0.907  |     +18.9 |     0.540 |     0.621 |    +13.1 |
|      2048 |       4 |     0.799  |     0.907  |     +11.9 |     0.559 |     0.621 |    +10.1 |
|      1024 |       4 |     0.867  |     0.907  |      +4.4 |     0.591 |     0.621 |     +4.8 |
|       512 |       4 |     0.906  |     0.907  |      +0.1 |     0.627 |     0.621 |     -1.0 |
|       256 |       4 |     0.919  |     0.907  |      -1.3 |     0.651 |     0.621 |     -4.9 |

### URLs

The urls dictionary is trained on the
[ada-url](https://github.com/ada-url/url-dataset) dataset.

| Dict Size | Max Seq | wrdz Ratio | smaz Ratio | Improvement % |
|----------:|---------|------------|------------|---------------|
|      8192 |       4 |      0.552 |      0.830 |         +33.5 |
|     16384 |       4 |      0.552 |      0.830 |         +33.5 |
|      4096 |       4 |      0.562 |      0.830 |         +32.2 |
|      2048 |       4 |      0.587 |      0.830 |         +29.2 |
|      1024 |       4 |      0.611 |      0.830 |         +26.3 |
|       512 |       4 |      0.641 |      0.830 |         +22.7 |
|       256 |       4 |      0.666 |      0.830 |         +19.8 |

## Technical Details

1. **Dictionary Training**
   - Analyzes frequency of character sequences in training data
   - Selects sequences that maximize compression
   - Assigns variable-length binary codes based on frequency

2. **Compression Format**
   - 1-bit flag per token (dictionary/raw)
   - Variable-length dictionary codes (4-14 bits)
   - Raw UTF-8 encoding with 2-bit length prefix

## License

MIT License. See [LICENSE](LICENSE) file for details.
