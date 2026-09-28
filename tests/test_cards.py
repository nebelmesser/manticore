from __future__ import annotations

import pytest

from src.cards import (
    BORDER,
    BOTTOM_BORDER,
    CORNER_RADIUS,
    TRIAD,
    card_filename,
    choose_triad,
    frame_card,
    join_cards,
    load_triads,
)
from src.prompt import LexiconError


def test_three_cards_use_triad_names() -> None:
    assert [card_filename(index, 3, TRIAD) for index in (1, 2, 3)] == [
        "thesis.png",
        "antithesis.png",
        "synthesis.png",
    ]


def test_chosen_triad_names_the_files() -> None:
    captions = ("CAMEL", "LION", "CHILD")
    assert [card_filename(index, 3, captions) for index in (1, 2, 3)] == [
        "camel.png",
        "lion.png",
        "child.png",
    ]


def test_triad_captions_are_uppercase(tmp_path) -> None:
    path = tmp_path / "triads.txt"
    path.write_text("Camel Lion Child\n\nBirth Death Transformation\n", encoding="utf-8")

    assert load_triads(path) == (
        ("CAMEL", "LION", "CHILD"),
        ("BIRTH", "DEATH", "TRANSFORMATION"),
    )


def test_shipped_triads_are_three_uppercase_words() -> None:
    triads = load_triads()

    assert len(triads) > 1
    assert triads[0] == TRIAD
    assert all(len(triad) == len(TRIAD) for triad in triads)
    assert all(word == word.upper() for triad in triads for word in triad)


def test_choose_triad_is_stable(tmp_path) -> None:
    path = tmp_path / "triads.txt"
    path.write_text("Camel Lion Child\nFaith Doubt Understanding\n", encoding="utf-8")

    chosen = choose_triad(1 << 255, 3, path)

    assert chosen == choose_triad(1 << 255, 3, path)
    assert chosen in {("CAMEL", "LION", "CHILD"), ("FAITH", "DOUBT", "UNDERSTANDING")}


def test_bad_triad_line_is_refused(tmp_path) -> None:
    path = tmp_path / "triads.txt"
    path.write_text("Only Two\n", encoding="utf-8")

    with pytest.raises(LexiconError):
        load_triads(path)


def test_empty_triads_are_refused(tmp_path) -> None:
    path = tmp_path / "triads.txt"
    path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(LexiconError):
        load_triads(path)


def test_other_counts_stay_numbered() -> None:
    assert card_filename(1, 1) == "card_1.png"
    assert card_filename(2, 4) == "card_2.png"


def test_frame_adds_black_border_and_centered_white_caption() -> None:
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (40, 20), "red")

    framed = frame_card(image, "THESIS")

    assert framed.size == (40 + BORDER * 2, 20 + BORDER + BOTTOM_BORDER)
    assert framed.getpixel((0, 0)) == (0, 0, 0, 255)
    assert framed.getpixel((BORDER, BORDER))[:3] == (0, 0, 0)
    assert framed.getpixel((BORDER + 1, BORDER + 1))[0] not in (0, 255)
    assert framed.getpixel((BORDER + CORNER_RADIUS, BORDER + CORNER_RADIUS))[:3] == (255, 0, 0)
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


def test_three_cards_join_left_to_right_on_black() -> None:
    Image = pytest.importorskip("PIL.Image")
    colors = ((255, 0, 0), (0, 128, 0), (0, 0, 255))
    cards = [frame_card(Image.new("RGB", (40, 20), color), caption) for color, caption in zip(colors, TRIAD)]

    sheet = join_cards(cards)

    width = cards[0].width
    assert sheet.size == (width * 3 + BORDER * 2, cards[0].height + BORDER * 2)
    assert sheet.getpixel((0, 0))[:3] == (0, 0, 0)
    assert sheet.getpixel((BORDER - 1, BORDER))[:3] == (0, 0, 0)
    for index, color in enumerate(colors):
        origin = BORDER + index * width
        assert sheet.getpixel((origin, BORDER))[:3] == (0, 0, 0)
        assert sheet.getpixel((origin + width - 1, BORDER))[:3] == (0, 0, 0)
        assert sheet.getpixel((origin + BORDER + CORNER_RADIUS, BORDER + BORDER + CORNER_RADIUS))[:3] == color
