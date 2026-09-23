from __future__ import annotations

import pytest

from src.cards import BORDER, BOTTOM_BORDER, CORNER_RADIUS, TRIAD, card_filename, frame_card


def test_three_cards_use_triad_names() -> None:
    assert [card_filename(index, 3) for index in (1, 2, 3)] == [
        "thesis.png",
        "antithesis.png",
        "synthesis.png",
    ]


def test_other_counts_stay_numbered() -> None:
    assert card_filename(1, 1) == "card_1.png"
    assert card_filename(2, 4) == "card_2.png"


def test_frame_adds_black_border_and_centered_white_caption() -> None:
    PIL = pytest.importorskip("PIL")
    image = PIL.Image.new("RGB", (40, 20), "red")

    framed = frame_card(image, "THESIS")

    assert framed.size == (40 + BORDER * 2, 20 + BORDER + BOTTOM_BORDER)
    assert framed.getpixel((0, 0))[3] == 0
    assert framed.getpixel((0, CORNER_RADIUS))[:3] == (0, 0, 0)
    assert framed.getpixel((0, CORNER_RADIUS))[3] == 255
    assert framed.getpixel((BORDER, BORDER))[:3] == (255, 0, 0)
    assert framed.getpixel((BORDER - 1, BORDER))[:3] == (0, 0, 0)
    white = [
        (x, y)
        for y in range(framed.height - BOTTOM_BORDER, framed.height)
        for x in range(framed.width)
        if framed.getpixel((x, y))[:3] == (255, 255, 255)
    ]
    assert white
    assert all(y >= framed.height - BOTTOM_BORDER for _x, y in white)
    center = sum(x for x, _y in white) / len(white)
    assert abs(center - framed.width / 2) < 8
    assert "THESIS" in TRIAD
