from __future__ import annotations

from src.paths import FONT_PATH


TRIAD = ("THESIS", "ANTITHESIS", "SYNTHESIS")
DIVINATION_NAME = "divination.png"
BORDER = 20
BOTTOM_BORDER = 40
CAPTION_SIZE = 28
CORNER_RADIUS = 5


def card_filename(index: int, count: int) -> str:
    if count == len(TRIAD):
        return f"{TRIAD[index - 1].lower()}.png"
    return f"card_{index}.png"


def _round_image(image, radius: int):
    """Clip an image to a rounded rectangle with a supersampled edge."""

    from PIL import Image, ImageChops, ImageDraw

    image = image.convert("RGBA")
    scale = 8
    big = (image.width * scale, image.height * scale)
    mask = Image.new("L", big, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, big[0] - 1, big[1] - 1),
        radius=radius * scale,
        fill=255,
    )
    mask = mask.resize(image.size, Image.Resampling.BOX)
    red, green, blue, alpha = image.split()
    image.putalpha(ImageChops.multiply(alpha, mask))
    return image


def frame_card(image, caption: str):
    """Add a black border and a centered white caption in the bottom band."""

    from PIL import Image, ImageDraw, ImageFont

    picture = _round_image(image, CORNER_RADIUS)
    picture = Image.alpha_composite(Image.new("RGBA", picture.size, (0, 0, 0, 255)), picture)
    size = (picture.width + BORDER * 2, picture.height + BORDER + BOTTOM_BORDER)
    framed = Image.new("RGBA", size, (0, 0, 0, 255))
    framed.paste(picture, (BORDER, BORDER))
    draw = ImageDraw.Draw(framed)
    font = ImageFont.truetype(str(FONT_PATH), size=CAPTION_SIZE)
    draw.text(
        (framed.width / 2, image.height + BORDER + BOTTOM_BORDER / 2),
        caption,
        fill=(255, 255, 255, 255),
        font=font,
        anchor="mm",
    )
    return framed


def join_cards(cards):
    """Place framed cards side by side, with one more black border around them."""

    from PIL import Image

    width = sum(card.width for card in cards)
    height = max(card.height for card in cards)
    sheet = Image.new("RGBA", (width + BORDER * 2, height + BORDER * 2), (0, 0, 0, 255))
    offset = BORDER
    for card in cards:
        sheet.paste(card, (offset, BORDER))
        offset += card.width
    return sheet
