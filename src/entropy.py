from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Callable, TextIO

from src.prompt import estimate_seed_entropy


MIN_ENTROPY_BITS = 256
ENTROPY_PROMPT = "Enter entropy\n"


class NotEnoughEntropy(Exception):
    def __init__(self, bits: float) -> None:
        self.bits = bits
        super().__init__(bits)


def not_enough_entropy_message(bits: float) -> str:
    return f"Need at least {MIN_ENTROPY_BITS} bits of entropy, got {bits:.1f}."


def entropy_status(bits: float) -> str:
    return f"\r\033[K{bits:.1f} / {MIN_ENTROPY_BITS} bits"


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


def _absorb(text: str, read1: Callable[[], str], wait: Callable[[float], bool], timeout: float) -> tuple[str, bool]:
    if not wait(timeout):
        return text, False
    changed = False
    while True:
        char = read1()
        if char in ("", "\x04"):
            break
        if char == "\x03":
            raise KeyboardInterrupt
        updated = _apply_char(text, char)
        changed = changed or updated != text
        text = updated
        if not wait(0):
            break
    return text, changed


def enter_entropy(
    read1: Callable[[], str],
    write: Callable[[str], None],
    wait: Callable[[float], bool],
    *,
    minimum: int = MIN_ENTROPY_BITS,
) -> str:
    """Read characters until estimated entropy reaches `minimum` bits."""

    write(ENTROPY_PROMPT)
    text = ""
    while True:
        bits = estimate_seed_entropy(text)
        write(entropy_status(bits))
        if bits >= minimum:
            text, absorbed = _absorb(text, read1, wait, 0.05)
            if absorbed:
                continue
            write("\n")
            return text
        char = read1()
        if char in ("", "\x04"):
            raise NotEnoughEntropy(bits)
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

    def wait(timeout: float) -> bool:
        return bool(select.select([fd], [], [], timeout)[0])

    try:
        tty.setraw(fd)
        return enter_entropy(read1, write, wait)
    except KeyboardInterrupt:
        stderr.write("\n")
        raise
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)
