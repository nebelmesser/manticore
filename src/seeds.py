from __future__ import annotations

import hashlib
import math
import re

from src.prompt import PROMPT_ENTROPY_BITS, estimate_seed_entropy


TORCH_MIN_SEED = -(2**63)
TORCH_MAX_SEED = 2**64 - 1
BATCH_SEED_BITS = 512


def parse_seed(value: str) -> int | str:
    """Keep ordinary decimal seeds numeric and accept everything else as text."""

    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)", value):
        return int(value)
    return value


def seed_payload(seed: int | str) -> bytes:
    kind = b"int:" if isinstance(seed, int) else b"text:"
    encoding = "ascii" if isinstance(seed, int) else "utf-8"
    return kind + str(seed).encode(encoding)


def seed_entropy_material(seed: int | str) -> tuple[int, int]:
    """Return a deterministic bit reservoir capped by estimated input entropy."""

    bit_count = max(0, math.floor(estimate_seed_entropy(seed)))
    if bit_count == 0:
        return 0, 0
    if isinstance(seed, int):
        return abs(seed), bit_count
    byte_count = (bit_count + 7) // 8
    value = int.from_bytes(
        hashlib.shake_256(b"manticore-input\0" + seed_payload(seed)).digest(byte_count),
        byteorder="big",
    )
    padding = byte_count * 8 - bit_count
    return value >> padding, bit_count


def source_entropy_chunk(seed: int | str, card_index: int) -> tuple[int, int]:
    """Take up to one prompt's worth of unconsumed source-seed entropy."""

    material, total_bits = seed_entropy_material(seed)
    start = card_index * PROMPT_ENTROPY_BITS
    if start >= total_bits:
        return 0, 0
    chunk_bits = min(PROMPT_ENTROPY_BITS, total_bits - start)
    shift = total_bits - start - chunk_bits
    mask = (1 << chunk_bits) - 1
    return (material >> shift) & mask, chunk_bits


def generated_seed_bits(seed: int | str, offset: int, bit_count: int) -> int:
    if bit_count <= 0:
        return 0
    byte_count = (bit_count + 7) // 8
    value = int.from_bytes(
        hashlib.shake_256(
            b"manticore-batch-fill\0"
            + seed_payload(seed)
            + b"\0item:"
            + str(offset).encode("ascii")
        ).digest(byte_count),
        byteorder="big",
    )
    return value >> (byte_count * 8 - bit_count)


def sequence_seed(seed: int | str, offset: int) -> int | str:
    """Carry surplus source entropy forward, then fill to 512 bits."""

    if offset == 0:
        return seed
    if offset < 0:
        raise ValueError("seed sequence offset cannot be negative")
    carried, carried_bits = source_entropy_chunk(seed, offset)
    fill_bits = BATCH_SEED_BITS - carried_bits
    return (carried << fill_bits) | generated_seed_bits(seed, offset, fill_bits)


def to_sd_seed(seed: int | str) -> int:
    """Map an integer or text seed into the range accepted by PyTorch."""

    if isinstance(seed, int) and TORCH_MIN_SEED <= seed <= TORCH_MAX_SEED:
        return seed
    encoding = "ascii" if isinstance(seed, int) else "utf-8"
    digest = hashlib.blake2b(
        str(seed).encode(encoding),
        digest_size=8,
        person=b"manticore-sd",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
