from __future__ import annotations

import math
import re

import pytest

from src.entropy import (
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


def test_entry_starts_at_256_bits_and_stops() -> None:
    source = high_entropy_text()
    assert estimate_seed_entropy(source) >= MIN_ENTROPY_BITS
    cursor = {"index": 0}

    def read1() -> str:
        index = cursor["index"]
        cursor["index"] = index + 1
        return source[index]

    written: list[str] = []
    text = enter_entropy(read1, written.append, lambda _timeout: False)

    assert estimate_seed_entropy(text) >= MIN_ENTROPY_BITS
    assert text == source[: len(text)]
    assert cursor["index"] == len(text)
    assert cursor["index"] < len(source)
    assert written[0] == "Enter entropy\n"
    assert written[-1] == "\n"
    assert f"{MIN_ENTROPY_BITS} bits" in "".join(written)


def test_entry_shows_growing_bit_count() -> None:
    source = high_entropy_text()
    cursor = {"index": 0}

    def read1() -> str:
        index = cursor["index"]
        cursor["index"] = index + 1
        return source[index]

    written: list[str] = []
    enter_entropy(read1, written.append, lambda _timeout: False)
    statuses = [line for line in written if line.startswith("\r")]
    bits = [float(re.search(r"\d+\.\d+", line).group(0)) for line in statuses]

    assert bits[0] == 0.0
    assert bits[-1] >= MIN_ENTROPY_BITS
    assert bits == sorted(bits)


def test_backspace_removes_entropy() -> None:
    source = "abcd\x7f"
    cursor = {"index": 0}

    def read1() -> str:
        if cursor["index"] >= len(source):
            return ""
        char = source[cursor["index"]]
        cursor["index"] += 1
        return char

    with pytest.raises(NotEnoughEntropy) as error:
        enter_entropy(read1, lambda _data: None, lambda _timeout: False)

    assert math.isclose(error.value.bits, estimate_seed_entropy("abc"))


def test_buffered_paste_is_kept() -> None:
    source = high_entropy_text()
    crossing = 1
    while estimate_seed_entropy(source[:crossing]) < MIN_ENTROPY_BITS:
        crossing += 1
    cursor = {"index": 0}
    waits = {"count": 0}

    def read1() -> str:
        char = source[cursor["index"]]
        cursor["index"] += 1
        return char

    def wait(timeout: float) -> bool:
        if timeout == 0:
            return cursor["index"] < len(source)
        waits["count"] += 1
        return cursor["index"] < len(source)

    text = enter_entropy(read1, lambda _data: None, wait)

    assert text == source
    assert waits["count"] >= 1
