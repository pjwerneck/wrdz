from pathlib import Path

from .bit_io import BitReader
from .bit_io import BitWriter
from .train import LENGTH_BITS
from .train import MAX_SEQUENCE_LENGTH
from .train import load_dictionary

HERE = Path(__file__).parent
DICTS = HERE / "dicts"

ENGLISH_CBOOK, ENGLISH_DBOOK = load_dictionary(DICTS / "en_US.dict")
URLS_CBOOK, URLS_DBOOK = load_dictionary(DICTS / "urls.dict")


def base_compress(text, codebook, escape_code=(0b1, 1)):
    writer = BitWriter()
    i = 0
    escape_value, escape_bits = escape_code

    while i < len(text):
        matched = False
        best_length = 0
        best_code = None
        best_bits = None

        # Limit search to maximum sequence length
        for length in range(min(MAX_SEQUENCE_LENGTH, len(text) - i), 0, -1):
            substr = text[i : i + length]
            if substr in codebook:
                code, bits = codebook[substr]
                matched = True
                best_length = length
                best_code = code
                best_bits = bits
                break

        if matched:
            writer.write_bits(escape_value ^ 1, escape_bits)
            writer.write_bits(best_code, best_bits)
            i += best_length
        else:
            char_bytes = text[i].encode("utf-8")
            writer.write_bits(escape_value, escape_bits)
            writer.write_bits(len(char_bytes) - 1, LENGTH_BITS)  # 2 bits for size
            for b in char_bytes:
                writer.write_bits(b, 8)
            i += 1

    return writer.finalize()


def base_decompress(data, decode_book, escape_code=(0b1, 1)):
    reader = BitReader(data)
    result = []
    escape_value, escape_bits = escape_code
    bits_needed = next(iter(decode_book))[1] if decode_book else 4

    while reader.has_more_bits():
        flag = reader.read_bits(escape_bits)
        if flag is None:
            break

        if flag == escape_value:  # Raw bytes
            # Read variable length size (2 bits can handle up to 3 bytes)
            size = reader.read_bits(LENGTH_BITS)
            if size is None or size >= MAX_SEQUENCE_LENGTH:
                break

            # Read UTF-8 bytes
            char_bytes = bytearray()
            for _ in range(size + 1):
                byte = reader.read_bits(8)
                if byte is None:
                    break
                char_bytes.append(byte)

            if len(char_bytes) == size + 1:
                try:
                    result.append(char_bytes.decode("utf-8"))
                except UnicodeDecodeError:
                    break
            else:
                break
        else:  # Dictionary entry
            code = reader.read_bits(bits_needed)
            if code is None:
                break
            if (code, bits_needed) in decode_book:
                result.append(decode_book[(code, bits_needed)])

    return "".join(result)
