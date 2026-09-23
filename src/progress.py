from __future__ import annotations

import sys
from typing import TextIO


CHAR_COLUMNS = 16
CHAR_ROWS = 8
PAIR_WIDTH = 2
GRID_COLUMNS = CHAR_COLUMNS // PAIR_WIDTH
GRID_ROWS = CHAR_ROWS
CELL_COUNT = GRID_COLUMNS * GRID_ROWS

_EMPTY = "░"
_FULL = "█"


def _hilbert_path(order: int = 4) -> tuple[tuple[int, int], ...]:
    """Visit every cell of a 2^order square along a Hilbert curve."""

    side = 1 << order
    points: list[tuple[int, int]] = []

    def walk(x: int, y: int, xi: int, xj: int, yi: int, yj: int, size: int) -> None:
        if size <= 1:
            points.append((x + (xi + yi) // 2, y + (xj + yj) // 2))
            return
        half = size // 2
        walk(x, y, yi // 2, yj // 2, xi // 2, xj // 2, half)
        walk(x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2, half)
        walk(x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2, yj // 2, half)
        walk(x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2, half)

    walk(0, 0, side, 0, 0, side, side)
    return tuple(points)


HILBERT_PATH = _hilbert_path(3)


def _scaled(done: int, total: int, units: int) -> int:
    if total <= 0 or done >= total:
        return units
    fraction = min(1.0, max(0.0, done / total))
    return round(units * fraction)


def render_bar(done: int, total: int) -> str:
    """Draw a 16×8 Hilbert field. Each curve step fills two characters."""

    filled = _scaled(done, total, CELL_COUNT)
    cells = [[False] * GRID_COLUMNS for _ in range(GRID_ROWS)]
    for index, (x, y) in enumerate(HILBERT_PATH):
        if index >= filled:
            break
        cells[y][x] = True

    lines = []
    for row in range(CHAR_ROWS):
        line = "".join(
            (_FULL if cells[row][column] else _EMPTY) * PAIR_WIDTH
            for column in range(GRID_COLUMNS)
        )
        lines.append(line)
    return "\n".join(lines)


class ProgressBar:
    """One in-place Hilbert field. It prints no other text."""

    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self.total = total
        self.done = 0
        self.stream = stream if stream is not None else sys.stderr
        self._closed = False
        self._drawn = False
        self.draw()

    def set_done(self, done: int) -> None:
        self.done = min(self.total, max(0, done))
        self.draw()

    def draw(self) -> None:
        if self._drawn:
            self.stream.write(f"\033[{CHAR_ROWS - 1}A")
        frame = render_bar(self.done, self.total)
        self.stream.write("\r" + frame.replace("\n", "\n\r"))
        self.stream.flush()
        self._drawn = True

    def finish(self) -> None:
        self.done = self.total
        self.draw()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self.stream.write("\n")
        self.stream.flush()
        self._closed = True
