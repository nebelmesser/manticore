from __future__ import annotations

import math

import pytest

from src.prompt import (
    FRAGMENT_COUNT,
    FRAGMENTS_PER_PROMPT,
    INDEX_BITS,
    PROMPT_ENTROPY_BITS,
    estimate_seed_entropy,
    generate_prompt,
    prompt_indices,
)
from src.seeds import (
    BATCH_SEED_BITS,
    TORCH_MAX_SEED,
    parse_seed,
    sequence_seed,
    source_entropy_chunk,
    to_sd_seed,
)


class TestFragments:
    def __len__(self) -> int:
        return FRAGMENT_COUNT

    def __getitem__(self, index: int) -> str:
        return f"fragment-{index}"


TEST_FRAGMENTS = TestFragments()


@pytest.fixture(autouse=True)
def use_test_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.prompt.load_fragments", lambda: TEST_FRAGMENTS)


def test_same_seed_produces_same_prompt() -> None:
    assert generate_prompt(42) == generate_prompt(42)


def test_different_seeds_produce_different_prompts() -> None:
    assert generate_prompt(42).prompt != generate_prompt(43).prompt


def test_text_seed_is_reproducible() -> None:
    seed = "the tower remembers the rain"
    assert generate_prompt(seed) == generate_prompt(seed)
    assert generate_prompt(seed).prompt != generate_prompt("another omen").prompt


def test_seed_entropy_estimate_does_not_credit_hash_output() -> None:
    assert estimate_seed_entropy(42) == 6.0
    assert estimate_seed_entropy(0) == 0.0
    assert estimate_seed_entropy("") == 0.0
    assert estimate_seed_entropy("aaaa") == 0.0
    assert math.isclose(estimate_seed_entropy("abcd"), 8.0)


def test_prompt_consumes_all_entropy_bits() -> None:
    indexes = prompt_indices("all entropy bits")

    assert INDEX_BITS == 17
    assert PROMPT_ENTROPY_BITS == 272
    assert len(indexes) == FRAGMENTS_PER_PROMPT == 16
    assert all(0 <= index < FRAGMENT_COUNT for index in indexes)
    assert generate_prompt("all entropy bits").prompt.split(", ") == [
        TEST_FRAGMENTS[index] for index in indexes
    ]


def test_decimal_seed_remains_numeric() -> None:
    assert parse_seed("42") == 42
    assert parse_seed("the tower") == "the tower"
    assert parse_seed("0042") == "0042"


def test_batch_seed_sequence_is_reproducible_and_wide() -> None:
    seeds = [sequence_seed("omen", offset) for offset in range(3)]

    assert seeds[0] == "omen"
    assert seeds == [sequence_seed("omen", offset) for offset in range(3)]
    assert len(set(seeds)) == 3
    assert all(isinstance(seed, int) for seed in seeds[1:])
    assert all(0 <= seed < 2**BATCH_SEED_BITS for seed in seeds[1:])
    assert any(seed.bit_length() > 256 for seed in seeds[1:])


def test_surplus_entropy_flows_into_following_cards() -> None:
    seed = (1 << 600) - 1

    second_chunk, second_bits = source_entropy_chunk(seed, 1)
    third_chunk, third_bits = source_entropy_chunk(seed, 2)
    second_seed = sequence_seed(seed, 1)
    third_seed = sequence_seed(seed, 2)

    assert second_bits == PROMPT_ENTROPY_BITS
    assert second_chunk == 2**PROMPT_ENTROPY_BITS - 1
    assert second_seed >> (BATCH_SEED_BITS - second_bits) == second_chunk
    assert third_bits == 56
    assert third_chunk == 2**56 - 1
    assert third_seed >> (BATCH_SEED_BITS - third_bits) == third_chunk


def test_large_seed_is_deterministically_mapped_for_sd() -> None:
    seed = 12323243545098734098234092834029384067
    mapped = to_sd_seed(seed)

    assert mapped == to_sd_seed(seed)
    assert 0 <= mapped <= TORCH_MAX_SEED
    assert mapped != seed


def test_text_seed_is_deterministically_mapped_for_sd() -> None:
    seed = "башня помнит дождь"
    mapped = to_sd_seed(seed)

    assert mapped == to_sd_seed(seed)
    assert 0 <= mapped <= TORCH_MAX_SEED
