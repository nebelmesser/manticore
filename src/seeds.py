from __future__ import annotations

import hashlib
import math
import re

from src.prompt import PROMPT_ENTROPY_BITS, estimate_seed_entropy


TORCH_MIN_SEED = -(2**63)
TORCH_MAX_SEED = 2**64 - 1
SD_SEED_BITS = 64


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


def take_bits(seed: int | str, start: int, width: int) -> tuple[int, int]:
    """Read up to `width` unconsumed source bits, most significant bit first."""

    if start < 0 or width < 0:
        raise ValueError("bit slice cannot be negative")
    material, total_bits = seed_entropy_material(seed)
    if width == 0 or start >= total_bits:
        return 0, 0
    bit_count = min(width, total_bits - start)
    shift = total_bits - start - bit_count
    mask = (1 << bit_count) - 1
    return (material >> shift) & mask, bit_count


def generated_seed_bits(seed: int | str, offset: int, bit_count: int, role: bytes) -> int:
    if bit_count <= 0:
        return 0
    byte_count = (bit_count + 7) // 8
    value = int.from_bytes(
        hashlib.shake_256(
            b"manticore-batch-fill\0"
            + seed_payload(seed)
            + b"\0"
            + role
            + b"\0item:"
            + str(offset).encode("ascii")
        ).digest(byte_count),
        byteorder="big",
    )
    return value >> (byte_count * 8 - bit_count)


def _place_high(carried: int, carried_bits: int, width: int, fill: int) -> int:
    if carried_bits == 0:
        return fill
    return (carried << (width - carried_bits)) | fill


def card_parts(seed: int | str, index: int) -> tuple[int, int]:
    """Split source entropy into this card's prompt bits, then its image seed.

    Each card takes 272 prompt bits and then 64 seed bits. Real bits occupy the
    high part of each value. Whatever is missing is deterministic fill, so later
    cards never reuse bits already poured into an earlier prompt or seed.
    """

    if index < 0:
        raise ValueError("card index cannot be negative")
    stride = PROMPT_ENTROPY_BITS + SD_SEED_BITS
    prompt_at = index * stride
    prompt_chunk, prompt_bits = take_bits(seed, prompt_at, PROMPT_ENTROPY_BITS)
    seed_chunk, seed_bits = take_bits(seed, prompt_at + PROMPT_ENTROPY_BITS, SD_SEED_BITS)
    prompt_seed = _place_high(
        prompt_chunk,
        prompt_bits,
        PROMPT_ENTROPY_BITS,
        generated_seed_bits(seed, index, PROMPT_ENTROPY_BITS - prompt_bits, b"prompt"),
    )
    sd_seed = _place_high(
        seed_chunk,
        seed_bits,
        SD_SEED_BITS,
        generated_seed_bits(seed, index, SD_SEED_BITS - seed_bits, b"sd"),
    )
    return prompt_seed, sd_seed


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
