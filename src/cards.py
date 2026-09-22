from __future__ import annotations

from src.paths import FONT_PATH


TRIAD = ("THESIS", "ANTITHESIS", "SYNTHESIS")
BORDER = 20
BOTTOM_BORDER = 40
CAPTION_SIZE = 28


def card_filename(index: int, count: int) -> str:
    if count == len(TRIAD):
        return f"{TRIAD[index - 1].lower()}.png"
    return f"card_{index}.png"


def frame_card(image, caption: str):
    """Add a black border and a centered white caption in the bottom band."""

    from PIL import Image, ImageDraw, ImageFont

    framed = Image.new(
        "RGB",
        (image.width + BORDER * 2, image.height + BORDER + BOTTOM_BORDER),
        "black",
    )
    framed.paste(image, (BORDER, BORDER))
    draw = ImageDraw.Draw(framed)
    font = ImageFont.truetype(str(FONT_PATH), size=CAPTION_SIZE)
    draw.text(
        (framed.width / 2, image.height + BORDER + BOTTOM_BORDER / 2),
        caption,
        fill="white",
        font=font,
        anchor="mm",
    )
    return framed
