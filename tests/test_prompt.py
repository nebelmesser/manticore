from __future__ import annotations

import math

import pytest

from src.prompt import (
    FRAGMENT_COUNT,
    FRAGMENTS_PER_PROMPT,
    INDEX_BITS,
    PROMPT_ENTROPY_BITS,
    LexiconError,
    choose_negative,
    estimate_seed_entropy,
    generate_prompt,
    prompt_indices,
)
from src.seeds import (
    SD_SEED_BITS,
    TORCH_MAX_SEED,
    card_parts,
    parse_seed,
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


def test_single_negative_line_is_always_chosen(tmp_path) -> None:
    path = tmp_path / "negative.txt"
    path.write_text("blurry watermark\n", encoding="utf-8")

    assert choose_negative(path) == "blurry watermark"
    assert choose_negative(path) == "blurry watermark"


def test_several_negative_lines_choose_one_at_random(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "negative.txt"
    path.write_text("blurry\nwatermark\ntext\n", encoding="utf-8")
    monkeypatch.setattr("src.prompt.random.choice", lambda lines: lines[1])

    assert choose_negative(path) == "watermark"


def test_empty_negative_corpus_is_refused(tmp_path) -> None:
    path = tmp_path / "negative.txt"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(LexiconError):
        choose_negative(path)


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


def test_card_parts_are_reproducible() -> None:
    first = [card_parts("omen", offset) for offset in range(3)]

    assert first == [card_parts("omen", offset) for offset in range(3)]
    assert len(set(first)) == 3
    assert all(0 <= sd_seed < 2**SD_SEED_BITS for _prompt_seed, sd_seed in first)


def test_surplus_flows_from_prompt_into_seed_then_onward() -> None:
    prompt_only = (1 << PROMPT_ENTROPY_BITS) - 1
    image_seed = 0xABC
    first_prompt, first_sd = card_parts(prompt_only, 0)
    wider = (prompt_only << SD_SEED_BITS) | image_seed
    poured_prompt, poured_sd = card_parts(wider, 0)
    next_prompt_bits = (1 << PROMPT_ENTROPY_BITS) - 1
    with_next_prompt = (wider << PROMPT_ENTROPY_BITS) | next_prompt_bits
    second_prompt, second_sd = card_parts(with_next_prompt, 1)
    third_prompt, _third_sd = card_parts(with_next_prompt, 2)

    assert first_prompt == prompt_only
    assert poured_prompt == prompt_only
    assert poured_sd == image_seed
    assert poured_sd != to_sd_seed(poured_prompt)
    assert second_prompt == next_prompt_bits
    assert second_sd != poured_sd
    assert third_prompt != second_prompt
    assert generate_prompt(poured_prompt).prompt == generate_prompt(first_prompt).prompt


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
