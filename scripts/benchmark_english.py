import concurrent.futures
import statistics
from pathlib import Path
from typing import List
from typing import NamedTuple

import smaz
from rich.console import Console
from rich.table import Table

from wrdz.base import base_compress
from wrdz.base import base_decompress
from wrdz.train import train_dictionary

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "datasets"
TEST_FILE = DATASETS / "en_US" / "test.txt"
TRAIN_FILE = DATASETS / "en_US" / "train.txt"

SHORT_STRING_THRESHOLD = 32


class Result(NamedTuple):
    name: str
    short_wrdz_ratio: float  # for texts < 10 chars
    short_smaz_ratio: float
    long_wrdz_ratio: float  # for texts >= 10 chars
    long_smaz_ratio: float
    dict_size: int
    max_sub_len: int

    @property
    def short_improvement(self) -> float:
        return (self.short_smaz_ratio - self.short_wrdz_ratio) / self.short_smaz_ratio * 100

    @property
    def long_improvement(self) -> float:
        return (self.long_smaz_ratio - self.long_wrdz_ratio) / self.long_smaz_ratio * 100


def process_params(args) -> Result:
    train_text, test_lines, dict_size, max_sub_len = args

    # Train dictionary on training data
    cbook, dbook = train_dictionary(train_text, max_sub_len=max_sub_len, dict_size=dict_size)

    # Split lines by length and initialize ratio lists
    short_wrdz_ratios = []
    short_smaz_ratios = []
    long_wrdz_ratios = []
    long_smaz_ratios = []

    for line in test_lines:
        if not line.strip():
            continue

        original_size = len(line.encode("utf-8"))
        wrdz_compressed = base_compress(line, cbook)
        smaz_compressed = smaz.compress(line)

        ratio_wrdz = len(wrdz_compressed) / original_size
        ratio_smaz = len(smaz_compressed) / original_size

        if len(line) < SHORT_STRING_THRESHOLD:
            short_wrdz_ratios.append(ratio_wrdz)
            short_smaz_ratios.append(ratio_smaz)
        else:
            long_wrdz_ratios.append(ratio_wrdz)
            long_smaz_ratios.append(ratio_smaz)

        assert base_decompress(wrdz_compressed, dbook) == line

    return Result(
        name="benchmark_test",
        short_wrdz_ratio=statistics.mean(short_wrdz_ratios) if short_wrdz_ratios else float("inf"),
        short_smaz_ratio=statistics.mean(short_smaz_ratios) if short_smaz_ratios else float("inf"),
        long_wrdz_ratio=statistics.mean(long_wrdz_ratios) if long_wrdz_ratios else float("inf"),
        long_smaz_ratio=statistics.mean(long_smaz_ratios) if long_smaz_ratios else float("inf"),
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
    results: List[Result] = []

    # Generate parameter combinations
    test_params = [
        (train_text, test_lines, dict_size, max_sub_len)
        for dict_size in dict_sizes
        for max_sub_len in max_sub_lens
    ]

    # Process parameters in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results.extend(executor.map(process_params, test_params))

    # Sort results by compression improvement
    results.sort(key=lambda x: x.long_improvement, reverse=True)

    # Create pretty table
    table = Table(title=f"Compression Benchmark Results (Test Set: {line_count} lines)")
    table.add_column("Dict Size", style="magenta", justify="right")
    table.add_column("Max Seq", style="magenta", justify="right")
    table.add_column("Short wrdz", justify="right")
    table.add_column("Short smaz", justify="right")
    table.add_column("Short Δ%", justify="right")
    table.add_column("Long wrdz", justify="right")
    table.add_column("Long smaz", justify="right")
    table.add_column("Long Δ%", justify="right")

    for r in results:
        table.add_row(
            str(r.dict_size),
            str(r.max_sub_len),
            f"{r.short_wrdz_ratio:.3f}",
            f"{r.short_smaz_ratio:.3f}",
            f"{r.short_improvement:+.1f}",
            f"{r.long_wrdz_ratio:.3f}",
            f"{r.long_smaz_ratio:.3f}",
            f"{r.long_improvement:+.1f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
