from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from src.paths import LEXICON


INDEX_BITS = 17
FRAGMENT_COUNT = 2**INDEX_BITS
FRAGMENTS_PER_PROMPT = 16
PROMPT_ENTROPY_BITS = INDEX_BITS * FRAGMENTS_PER_PROMPT


class LexiconError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedPrompt:
    seed: int | str
    seed_entropy_bits: float
    prompt: str


@lru_cache(maxsize=1)
def load_fragments(path: Path = LEXICON) -> tuple[str, ...]:
    if not path.is_file():
        raise LexiconError(f"Prompt corpus not found: {path}")

    fragments = tuple(line.rstrip("\n") for line in path.open(encoding="utf-8"))
    if len(fragments) != FRAGMENT_COUNT:
        raise LexiconError(
            f"Expected {FRAGMENT_COUNT} fragments in {path}, found {len(fragments)}"
        )
    if any(not fragment for fragment in fragments):
        raise LexiconError(f"Corpus contains an empty fragment: {path}")
    if len(set(fragments)) != FRAGMENT_COUNT:
        raise LexiconError(f"Corpus fragments are not unique: {path}")
    return fragments


def seed_digest(seed: int | str) -> bytes:
    if isinstance(seed, int):
        payload = b"int:" + str(seed).encode("ascii")
    else:
        payload = b"text:" + seed.encode("utf-8")
    return hashlib.blake2b(
        payload,
        digest_size=PROMPT_ENTROPY_BITS // 8,
        person=b"manticore-prompt",
    ).digest()


def estimate_seed_entropy(seed: int | str) -> float:
    """Estimate input entropy without crediting bits added by hashing."""

    if isinstance(seed, int):
        return float(abs(seed).bit_length())
    if not seed:
        return 0.0
    length = len(seed)
    counts = Counter(seed)
    return sum(-count * math.log2(count / length) for count in counts.values())


def prompt_indices(seed: int | str) -> tuple[int, ...]:
    """Turn the seed digest into sixteen unsigned 17-bit corpus indexes."""

    digest_value = int.from_bytes(seed_digest(seed), byteorder="big", signed=False)
    mask = FRAGMENT_COUNT - 1
    return tuple(
        (digest_value >> shift) & mask
        for shift in range(PROMPT_ENTROPY_BITS - INDEX_BITS, -1, -INDEX_BITS)
    )


def generate_prompt(
    seed: int | str,
    fragments: Sequence[str] | None = None,
) -> GeneratedPrompt:
    """Select sixteen corpus fragments using all 272 bits of the seed digest."""

    pool = load_fragments() if fragments is None else fragments
    if len(pool) != FRAGMENT_COUNT:
        raise LexiconError(f"Expected {FRAGMENT_COUNT} fragments, found {len(pool)}")
    prompt = ", ".join(pool[index] for index in prompt_indices(seed))
    return GeneratedPrompt(
        seed=seed,
        seed_entropy_bits=round(estimate_seed_entropy(seed), 1),
        prompt=prompt,
    )
