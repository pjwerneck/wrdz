import concurrent.futures
import statistics
from pathlib import Path
from typing import NamedTuple

import smaz
from rich.console import Console
from rich.table import Table

from wrdz.base import base_compress
from wrdz.base import base_decompress
from wrdz.train import train_dictionary

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "datasets"
TEST_FILE = DATASETS / "urls" / "test.txt"
TRAIN_FILE = DATASETS / "urls" / "train.txt"


class Result(NamedTuple):
    name: str
    wrdz_ratio: float
    smaz_ratio: float
    dict_size: int
    max_sub_len: int

    @property
    def improvement(self) -> float:
        return (self.smaz_ratio - self.wrdz_ratio) / self.smaz_ratio * 100


def process_params(args) -> Result:
    train_text, test_lines, dict_size, max_sub_len = args

    # Train dictionary on training data
    cbook, dbook = train_dictionary(train_text, max_sub_len=max_sub_len, dict_size=dict_size)

    wrdz_ratios = []
    smaz_ratios = []

    for line in test_lines:
        if not line.strip():
            continue

        original_size = len(line.encode("utf-8"))
        wrdz_compressed = base_compress(line, cbook)
        smaz_compressed = smaz.compress(line)

        ratio_wrdz = len(wrdz_compressed) / original_size
        ratio_smaz = len(smaz_compressed) / original_size

        wrdz_ratios.append(ratio_wrdz)
        smaz_ratios.append(ratio_smaz)

        assert base_decompress(wrdz_compressed, dbook) == line

    return Result(
        name="benchmark_urls",
        wrdz_ratio=statistics.mean(wrdz_ratios),
        smaz_ratio=statistics.mean(smaz_ratios),
        dict_size=dict_size,
        max_sub_len=max_sub_len,
    )


def main():
    console = Console()

    # Read training and test files
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_text = f.read()

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_lines = [line.strip() for line in f if line.strip()]
        line_count = len(test_lines)

    # Expanded parameters to test
    dict_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    max_sub_lens = [3, 4]

    # Generate parameter combinations
    test_params = [
        (train_text, test_lines, dict_size, max_sub_len)
        for dict_size in dict_sizes
        for max_sub_len in max_sub_lens
    ]

    # Process parameters in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_params, test_params))

    # Sort results by compression improvement
    results.sort(key=lambda x: x.improvement, reverse=True)

    # Create pretty table
    table = Table(title=f"URL Compression Benchmark Results (Test Set: {line_count} URLs)")
    table.add_column("Dict Size", style="magenta", justify="right")
    table.add_column("Max Seq", style="magenta", justify="right")
    table.add_column("wrdz Ratio", justify="right")
    table.add_column("smaz Ratio", justify="right")
    table.add_column("Improvement %", justify="right")

    for r in results:
        table.add_row(
            str(r.dict_size),
            str(r.max_sub_len),
            f"{r.wrdz_ratio:.3f}",
            f"{r.smaz_ratio:.3f}",
            f"{r.improvement:+.1f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
