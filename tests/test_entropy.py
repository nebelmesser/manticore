from __future__ import annotations

import math
import re

import pytest

from src.entropy import (
    ENTROPY_PROMPT,
    MIN_ENTROPY_BITS,
    NotEnoughEntropy,
    enter_entropy,
    format_input_entropy,
    require_entropy,
)
from src.prompt import estimate_seed_entropy


def high_entropy_text() -> str:
    return "".join(chr(32 + index) for index in range(64))


def test_input_entropy_line_is_one_line() -> None:
    assert format_input_entropy(256) == "entropy: 256.0 bits\n"


def test_short_values_are_rejected() -> None:
    with pytest.raises(NotEnoughEntropy) as error:
        require_entropy(42)
    assert error.value.bits == 6.0

    with pytest.raises(NotEnoughEntropy):
        require_entropy("abcd")


def test_256_bit_integer_is_accepted() -> None:
    assert require_entropy(1 << 255) == 256.0
    with pytest.raises(NotEnoughEntropy) as error:
        require_entropy((1 << 255) - 1)
    assert error.value.bits == 255.0


def chars(source: str):
    cursor = {"index": 0}

    def read1() -> str:
        index = cursor["index"]
        if index >= len(source):
            return ""
        cursor["index"] = index + 1
        return source[index]

    return read1


def test_entry_returns_on_enter() -> None:
    source = "short\rleftover"
    written: list[str] = []
    text = enter_entropy(chars(source), written.append)

    assert text == "short"
    assert estimate_seed_entropy(text) < MIN_ENTROPY_BITS
    assert written[0] == ENTROPY_PROMPT
    assert written[-1] == "\n"
    assert "/ 256" not in "".join(written)


def test_entry_keeps_text_past_256_bits_until_enter() -> None:
    source = high_entropy_text() + "\n"
    assert estimate_seed_entropy(source[:-1]) >= MIN_ENTROPY_BITS
    text = enter_entropy(chars(source), lambda _data: None)

    assert text == source[:-1]


def test_entry_shows_growing_bit_count() -> None:
    source = "abc\n"
    written: list[str] = []
    enter_entropy(chars(source), written.append)
    statuses = [line for line in written if line.startswith("\r")]
    bits = [float(re.search(r"\d+\.\d+", line).group(0)) for line in statuses]

    assert bits[0] == 0.0
    assert bits[-1] == pytest.approx(estimate_seed_entropy("abc"), abs=0.05)
    assert bits == sorted(bits)


def test_backspace_removes_entropy() -> None:
    text = enter_entropy(chars("abcd\x7f\n"), lambda _data: None)

    assert text == "abc"
    assert math.isclose(estimate_seed_entropy(text), estimate_seed_entropy("abc"))


def test_paste_before_enter_is_kept() -> None:
    source = high_entropy_text() + "\r"
    text = enter_entropy(chars(source), lambda _data: None)

    assert text == source[:-1]
