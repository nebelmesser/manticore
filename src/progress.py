from __future__ import annotations

import sys
from typing import TextIO


PAIR_WIDTH = 2

_EMPTY = "░"
_FULL = "█"


def _hilbert_path(order: int) -> tuple[tuple[int, int], ...]:
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

    walk(0, 0, 0, side, side, 0, side)
    return tuple(points)


def _scaled(done: int, total: int, units: int) -> int:
    if total <= 0 or done >= total:
        return units
    fraction = min(1.0, max(0.0, done / total))
    return round(units * fraction)


class Field:
    """A Hilbert field. Each cell is two characters wide and one row tall."""

    def __init__(self, columns: int, rows: int) -> None:
        if rows < 1 or rows & (rows - 1) or columns != rows * PAIR_WIDTH:
            raise ValueError("field must be a Hilbert square drawn two characters per cell")
        self.columns = columns
        self.rows = rows
        self.path = _hilbert_path(rows.bit_length() - 1)

    def scaled(self, factor: int) -> Field:
        return Field(self.columns * factor, self.rows * factor)

    def render(self, done: int, total: int) -> str:
        cells_across = self.columns // PAIR_WIDTH
        filled = _scaled(done, total, len(self.path))
        cells = [[False] * cells_across for _ in range(self.rows)]
        for index, (x, y) in enumerate(self.path):
            if index >= filled:
                break
            cells[y][x] = True

        lines = []
        for row in range(self.rows):
            line = "".join(
                (_FULL if cells[row][column] else _EMPTY) * PAIR_WIDTH
                for column in range(cells_across)
            )
            lines.append(line)
        return "\n".join(lines)


NARROW = Field(16, 8)
VIDEO = NARROW.scaled(2)


def render_bar(done: int, total: int, field: Field = NARROW) -> str:
    """Draw a Hilbert field. Each curve step fills two characters."""

    return field.render(done, total)


class ProgressBar:
    """One in-place Hilbert field. It prints no other text."""

    def __init__(self, total: int, stream: TextIO | None = None, field: Field = NARROW) -> None:
        self.total = total
        self.done = 0
        self.field = field
        self.stream = stream if stream is not None else sys.stderr
        self._closed = False
        self._drawn = False
        self.draw()

    def set_done(self, done: int) -> None:
        self.done = min(self.total, max(0, done))
        self.draw()

    def draw(self) -> None:
        if self._drawn:
            self.stream.write(f"\033[{self.field.rows - 1}A")
        frame = self.field.render(self.done, self.total)
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
