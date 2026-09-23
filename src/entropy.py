from __future__ import annotations

import sys
import termios
import tty
from typing import Callable, TextIO

from src.prompt import estimate_seed_entropy


MIN_ENTROPY_BITS = 256
ENTROPY_PROMPT = "Type any entropy and press Enter\n"


class NotEnoughEntropy(Exception):
    def __init__(self, bits: float) -> None:
        self.bits = bits
        super().__init__(bits)


def not_enough_entropy_message(bits: float) -> str:
    return f"Need at least {MIN_ENTROPY_BITS} bits of entropy, got {bits:.1f}."


def entropy_status(bits: float) -> str:
    return f"\r\033[K{bits:.1f} bits"


def format_input_entropy(bits: float) -> str:
    return f"entropy: {bits:.1f} bits\n"


def require_entropy(seed: int | str) -> float:
    bits = estimate_seed_entropy(seed)
    if bits < MIN_ENTROPY_BITS:
        raise NotEnoughEntropy(bits)
    return bits


def _apply_char(text: str, char: str) -> str:
    if char in ("\x7f", "\b"):
        return text[:-1]
    if char == "\x00":
        return text
    return text + char


def enter_entropy(
    read1: Callable[[], str],
    write: Callable[[str], None],
) -> str:
    """Read characters until Enter. Any amount of entropy is accepted."""

    write(ENTROPY_PROMPT)
    text = ""
    while True:
        write(entropy_status(estimate_seed_entropy(text)))
        char = read1()
        if char in ("", "\x04", "\r", "\n"):
            write("\r\n")
            return text
        if char == "\x03":
            raise KeyboardInterrupt
        text = _apply_char(text, char)


def enter_entropy_tty(stdin: TextIO = sys.stdin, stderr: TextIO = sys.stderr) -> str:
    fd = stdin.fileno()
    previous = termios.tcgetattr(fd)

    def read1() -> str:
        return stdin.read(1)

    def write(data: str) -> None:
        stderr.write(data)
        stderr.flush()

    try:
        tty.setraw(fd)
        return enter_entropy(read1, write)
    except KeyboardInterrupt:
        stderr.write("\r\n")
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
