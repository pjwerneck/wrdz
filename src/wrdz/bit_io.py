class BitWriter:
    __slots__ = ("buffer", "bit_count", "bytes")

    def __init__(self):
        self.buffer = 0
        self.bit_count = 0
        self.bytes = bytearray()

    def write_bits(self, value, num_bits):
        self.buffer = (self.buffer << num_bits) | value
        self.bit_count += num_bits
        while self.bit_count >= 8:
            self.bytes.append((self.buffer >> (self.bit_count - 8)) & 0xFF)
            self.bit_count -= 8
            self.buffer &= (1 << self.bit_count) - 1

    def finalize(self):
        if self.bit_count > 0:
            self.buffer <<= 8 - self.bit_count
            self.bytes.append(self.buffer)
        return bytes(self.bytes)


class BitReader:
    __slots__ = ("data", "byte_pos", "bit_pos")

    def __init__(self, data):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0

    def has_more_bits(self):
        return (self.byte_pos < len(self.data) - 1) or (
            self.byte_pos == len(self.data) - 1 and self.bit_pos < 8
        )

    def read_bit(self):
        if not self.has_more_bits():
            return None
        byte = self.data[self.byte_pos]
        bit = (byte >> (7 - self.bit_pos)) & 1
        self.bit_pos += 1
        if self.bit_pos >= 8:
            self.bit_pos = 0
            self.byte_pos += 1
        return bit

    def read_bits(self, num_bits):
        value = 0
        bits_read = 0

        for _ in range(num_bits):
            bit = self.read_bit()
            if bit is None:
                return None  # Return None if we can't read all requested bits
            value = (value << 1) | bit
            bits_read += 1

        if bits_read < num_bits:
            return None  # Return None if we got a partial read

        return value
