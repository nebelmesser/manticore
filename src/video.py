from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def ping_pong(frames: Sequence[Path]) -> list[Path]:
    """Play the frames forward, then back to the first without repeating the last."""

    ordered = list(frames)
    if len(ordered) < 2:
        return ordered
    return ordered + ordered[-2::-1]


def video_destination(jobs: Sequence[tuple[object, Path]], separate: bool) -> Path:
    first = jobs[0][1]
    if separate:
        return first.parent / "divination.mp4"
    return first.with_suffix(".mp4")


def frames_directory(video: Path) -> Path:
    return video.with_name(f"{video.stem}-frames")


def reset_directory(path: Path) -> Path:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.mkdir(parents=True)
    return path


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("scry: --video needs ffmpeg", file=sys.stderr)
        raise SystemExit(1)
    return ffmpeg


def write_ping_pong_video(frames: Sequence[Path], dest: Path, fps: int) -> None:
    ffmpeg = require_ffmpeg()
    ordered = ping_pong(frames)
    sequence = dest.with_name(f"{dest.stem}-sequence")
    sequence.mkdir(parents=True)
    try:
        for index, frame in enumerate(ordered):
            (sequence / f"{index:05d}.png").symlink_to(Path(frame).resolve())
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(sequence / "%05d.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(sequence, ignore_errors=True)
