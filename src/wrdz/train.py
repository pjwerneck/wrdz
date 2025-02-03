import msgpack
from collections import defaultdict


MAX_SEQUENCE_LENGTH = 4  # Maximum UTF-8 sequence length due to 2-bit length field
LENGTH_BITS = 2  # Number of bits used for length encoding (1-4 bytes)

def load_dictionary(path):
    """Load a dictionary from a file.

    Args:
        path: Path to dictionary file
    
    Returns:
        Tuple of codebook and decodebook
    """
    with open(path, "rb") as f:
        cbook, dbook = msgpack.load(f, use_list=False, strict_map_key=False)
    return cbook, dbook


def save_dictionary(cbook, dbook, path):
    """Save a dictionary to a file.

    Args:
        cbook: Codebook
        dbook: Decodebook
    
    Returns:
        None

    """
    with open(path, "wb") as f:
        msgpack.dump((cbook, dbook), f)


def train_dictionary(text, max_sub_len=3, dict_size=512):
    """Train a dictionary for compression.

    Args:
        text: Input text to analyze
        max_sub_len: Maximum subsequence length (must not exceed 4)
        dict_size: Maximum dictionary size

    Raises:
        ValueError: If max_sub_len exceeds format limitation of 4
    """
    if max_sub_len > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"max_sub_len cannot exceed {MAX_SEQUENCE_LENGTH} due to format limitations"
        )

    freq = defaultdict(int)
    n = len(text)

    # Count sequences first
    for length in range(1, max_sub_len + 1):
        for i in range(n - length + 1):
            seq = text[i : i + length]
            freq[seq] += 1

    # Score formula compares actual encoded sizes
    scores = {}
    bits_needed = (dict_size - 1).bit_length()
    escape_bits = 1

    for seq, f in freq.items():
        if f > 1:
            # Size when using dictionary: escape bit (0) + dictionary bits
            compressed_bits = f * (escape_bits + bits_needed)

            # Size when using raw encoding: escape bit (1) + 2 bits length + 8 bits per byte
            raw_bits = f * (escape_bits + 2 + len(seq.encode("utf-8")) * 8)

            # Only include if we actually save space
            if compressed_bits < raw_bits:
                scores[seq] = raw_bits - compressed_bits

    # Select top sequences
    selected = sorted([(s, scores[s]) for s in scores], key=lambda x: (-x[1], len(x[0])))[
        :dict_size
    ]

    if not selected:
        return {}, {}

    # Use minimum bits needed
    bits_needed = max(4, (len(selected) - 1).bit_length())

    # Assign sequential codes
    cbook = {s: (i, bits_needed) for i, (s, _) in enumerate(selected)}
    dbook = {(code, bits_needed): s for s, (code, bits_needed) in cbook.items()}
    return cbook, dbook
