from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from src.cards import BORDER, TRIAD
from src.render import denoise_passes, step_sheets
from src.video import ping_pong, reset_directory, video_destination, write_ping_pong_video


def test_video_reruns_every_step_count() -> None:
    assert denoise_passes(25, video=False) == 25
    assert denoise_passes(25, video=True) == 25 * 26 // 2 - 1
    assert denoise_passes(2, video=True) == 2
    assert denoise_passes(1, video=True) == 1


def test_ping_pong_returns_to_the_start() -> None:
    assert ping_pong([]) == []
    assert [path.name for path in ping_pong([Path("a.png")])] == ["a.png"]
    assert [path.name for path in ping_pong([Path("a.png"), Path("b.png"), Path("c.png")])] == [
        "a.png",
        "b.png",
        "c.png",
        "b.png",
        "a.png",
    ]


def test_reset_directory_replaces_a_file_or_folder(tmp_path) -> None:
    folder = tmp_path / "vertesserax-frames"
    folder.mkdir()
    (folder / "old.png").write_bytes(b"old")
    reset_directory(folder)
    assert folder.is_dir()
    assert list(folder.iterdir()) == []

    occupied = tmp_path / "reading-frames"
    occupied.write_text("not a directory", encoding="utf-8")
    reset_directory(occupied)
    assert occupied.is_dir()
    assert list(occupied.iterdir()) == []


def test_video_sits_beside_the_image_or_inside_the_folder(tmp_path) -> None:
    image = tmp_path / "reading.png"
    folder = tmp_path / "reading"
    assert video_destination([(None, image)], separate=False) == tmp_path / "reading.mp4"
    assert video_destination([(None, folder / "thesis.png")], separate=True) == folder / "divination.mp4"


def _panel_pixel(sheet, panel: int) -> tuple[int, int, int]:
    panel_width = 40 + BORDER * 2
    origin = BORDER + panel * panel_width + BORDER + 8
    return sheet.getpixel((origin, BORDER + BORDER + 8))[:3]


def test_step_sheets_move_every_panel_together(tmp_path) -> None:
    Image = pytest.importorskip("PIL.Image")
    colors = (("red", "blue"), ("green", "yellow"), ("white",))
    card_steps = []
    for index, pair in enumerate(colors):
        folder = tmp_path / TRIAD[index].lower()
        folder.mkdir()
        paths = []
        for step, color in enumerate(pair):
            path = folder / f"{step:05d}.png"
            Image.new("RGB", (40, 20), color).save(path)
            paths.append(path)
        card_steps.append(paths)

    sheets = step_sheets(card_steps, tmp_path / "sheets", TRIAD)

    assert [path.name for path in sheets] == ["00000.png", "00001.png"]
    first = Image.open(sheets[0])
    second = Image.open(sheets[1])
    assert _panel_pixel(first, 0) == (255, 0, 0)
    assert _panel_pixel(first, 1)[1] > _panel_pixel(first, 1)[0]
    assert _panel_pixel(first, 2) == (255, 255, 255)
    assert _panel_pixel(second, 0)[2] > _panel_pixel(second, 0)[0]
    assert _panel_pixel(second, 1)[0] > 200 and _panel_pixel(second, 1)[1] > 200
    assert _panel_pixel(second, 2) == (255, 255, 255)


def test_ping_pong_video_plays_forward_and_back(tmp_path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg is required")
    Image = pytest.importorskip("PIL.Image")
    frames = []
    for index, color in enumerate(("red", "green")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (16, 16), color).save(path)
        frames.append(path)

    dest = tmp_path / "reading.mp4"
    write_ping_pong_video(frames, dest, fps=10)

    assert dest.is_file()
    assert not (tmp_path / "reading-sequence").exists()
    counted = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert counted.stdout.strip() == "3"
