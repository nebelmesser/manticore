from __future__ import annotations

import sys
from typing import TextIO


def render_bar(done: int, total: int, width: int = 32) -> str:
    if total <= 0:
        fraction = 1.0
    else:
        fraction = min(1.0, max(0.0, done / total))
    filled = width if total > 0 and done >= total else round(width * fraction)
    percent = 100 if total > 0 and done >= total else round(100 * fraction)
    return f"\r[{'█' * filled}{'░' * (width - filled)}] {percent:3d}%"


class ProgressBar:
    """One in-place bar. It prints no other text."""

    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self.total = total
        self.done = 0
        self.stream = stream if stream is not None else sys.stderr
        self._closed = False
        self.draw()

    def set_done(self, done: int) -> None:
        self.done = min(self.total, max(0, done))
        self.draw()

    def draw(self) -> None:
        self.stream.write(render_bar(self.done, self.total))
        self.stream.flush()

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
