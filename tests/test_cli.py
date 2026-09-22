from __future__ import annotations

import io
import sys
from datetime import datetime

import pytest

from src.cli import DEFAULT_NEGATIVE, build_parser, main, reserve_output_dir
from src.entropy import MIN_ENTROPY_BITS
from src.model import REQUIRED_FILES, model_is_ready
from src.progress import render_bar
from src.prompt import FRAGMENT_COUNT


class TestFragments:
    def __len__(self) -> int:
        return FRAGMENT_COUNT

    def __getitem__(self, index: int) -> str:
        return f"fragment-{index}"


def high_entropy_text() -> str:
    return "".join(chr(32 + index) for index in range(64))


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch, tmp_path):
    calls: list[dict] = []

    def render(jobs, *, width, height, steps, negative_prompt):
        calls.append(
            {
                "paths": [path for _card, path in jobs],
                "prompts": [card.prompt for card, _path in jobs],
                "width": width,
                "height": height,
                "steps": steps,
                "negative_prompt": negative_prompt,
            }
        )

    monkeypatch.setattr("src.prompt.load_fragments", lambda: TestFragments())
    monkeypatch.setattr("src.cli.render_cards", render)
    monkeypatch.setattr("src.cli.reserve_output_dir", lambda: tmp_path)
    return calls


def test_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.seed is None
    assert args.count == 3
    assert args.width == 512
    assert args.height == 1024
    assert args.steps == 25
    assert args.negative == DEFAULT_NEGATIVE
    assert args.negative == "text letters label title panels comics captions subtitle"


def test_short_seed_is_refused(fake_render, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["42"])

    captured = capsys.readouterr()
    assert error.value.code == 2
    assert fake_render == []
    assert "256" in captured.err
    assert captured.out == ""


def test_run_uses_defaults_and_card_names(fake_render, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    main([str(1 << 255)])

    assert len(fake_render) == 1
    call = fake_render[0]
    assert [path.name for path in call["paths"]] == ["thesis.png", "antithesis.png", "synthesis.png"]
    assert len(set(call["prompts"])) == 3
    assert call["width"] == 512
    assert call["height"] == 1024
    assert call["steps"] == 25
    assert call["negative_prompt"] == DEFAULT_NEGATIVE
    captured = capsys.readouterr()
    assert captured.out.strip() == str(tmp_path)
    assert captured.err == ""


def test_flags_override_defaults(fake_render) -> None:
    main(
        [
            high_entropy_text(),
            "--count",
            "1",
            "--width",
            "768",
            "--height",
            "768",
            "--steps",
            "10",
            "--negative",
            "blurry",
        ]
    )

    call = fake_render[0]
    assert [path.name for path in call["paths"]] == ["card_1.png"]
    assert call["width"] == 768
    assert call["height"] == 768
    assert call["steps"] == 10
    assert call["negative_prompt"] == "blurry"


def test_missing_seed_reads_tty(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("src.cli.enter_entropy_tty", lambda _stdin, _stderr: high_entropy_text())

    main([])

    assert len(fake_render[0]["paths"]) == 3


def test_piped_entropy_is_accepted(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(high_entropy_text()))

    main([])

    assert len(fake_render) == 1


def test_piped_short_entropy_is_refused(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("short"))

    with pytest.raises(SystemExit) as error:
        main([])

    assert error.value.code == 2
    assert fake_render == []


def test_output_directory_name(tmp_path) -> None:
    moment = datetime(2026, 9, 22, 16, 50, 7)
    first = reserve_output_dir(moment, tmp_path)
    second = reserve_output_dir(moment, tmp_path)

    assert first.name == "2026-09-22-16-50-07"
    assert second.name == "2026-09-22-16-50-07-2"
    assert first.is_dir() and second.is_dir()


def test_dimensions_must_be_multiples_of_eight() -> None:
    with pytest.raises(SystemExit):
        main([str(1 << 255), "--width", "513"])


def test_progress_bar_is_only_a_bar() -> None:
    assert render_bar(0, 75) == "\r[" + "░" * 32 + "]   0%"
    finished = render_bar(75, 75)
    assert finished.endswith("100%")
    assert "█" * 32 in finished
    assert "prompt" not in finished
    assert "card" not in finished


def test_model_ready_requires_every_file(tmp_path) -> None:
    assert model_is_ready(tmp_path) is False
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert model_is_ready(tmp_path) is True


def test_minimum_entropy_constant() -> None:
    assert MIN_ENTROPY_BITS == 256
