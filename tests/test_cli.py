from __future__ import annotations

import io
import sys
from datetime import datetime

import pytest

from src.cli import build_parser, main, reserve_output_dir, reserve_output_file
from src.entropy import MIN_ENTROPY_BITS
from src.model import REQUIRED_FILES, model_is_ready
from src.progress import render_bar
from src.prompt import FRAGMENT_COUNT, estimate_seed_entropy


class TestFragments:
    def __len__(self) -> int:
        return FRAGMENT_COUNT

    def __getitem__(self, index: int) -> str:
        return f"fragment-{index}"


def high_entropy_text() -> str:
    return "".join(chr(32 + index) for index in range(64))


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch, tmp_path):
    class Calls(list):
        out: str | None = None

    calls = Calls()

    def render(jobs, *, width, height, steps, negative_prompt, entropy_bits, prepare, show_entropy, separate):
        calls.append(
            {
                "paths": [path for _card, path in jobs],
                "prompts": [card.prompt for card, _path in jobs],
                "seeds": [card.sd_seed for card, _path in jobs],
                "width": width,
                "height": height,
                "steps": steps,
                "negative_prompt": negative_prompt,
                "entropy_bits": entropy_bits,
                "show_entropy": show_entropy,
                "separate": separate,
            }
        )

    monkeypatch.setattr("src.prompt.load_fragments", lambda: TestFragments())
    monkeypatch.setattr(
        "src.cli.choose_negative",
        lambda: "text letters label title panels comics captions subtitle",
    )
    monkeypatch.setattr("src.cli.render_cards", render)

    class DummyLoad:
        def result(self) -> None:
            return None

    monkeypatch.setattr("src.render.start_pipeline", lambda **_kwargs: DummyLoad())
    monkeypatch.setattr("src.model.model_is_ready", lambda _path=None: True)
    calls.out = None

    def reserve(*_args, **kwargs):
        calls.out = kwargs.get("name")
        return tmp_path

    def reserve_file(*_args, **kwargs):
        calls.out = kwargs.get("name")
        return tmp_path / "sheet.png"

    monkeypatch.setattr("src.cli.reserve_output_dir", reserve)
    monkeypatch.setattr("src.cli.reserve_output_file", reserve_file)
    return calls


def test_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.seed is None
    assert args.count is None
    assert args.width == 512
    assert args.height == 1024
    assert args.steps == 25
    assert args.out is None
    assert not hasattr(args, "negative")


def test_cold_start_finishes_before_entropy(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    order = []
    monkeypatch.setattr("src.model.model_is_ready", lambda _path=None: False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        "src.cli.enter_entropy_tty",
        lambda *_args: order.append("entropy") or high_entropy_text(),
    )

    class Ready:
        def result(self) -> None:
            return None

    monkeypatch.setattr("src.render.load_cold_pipeline", lambda **_kwargs: order.append("cold") or Ready())
    monkeypatch.setattr(
        "src.render.start_pipeline",
        lambda **_kwargs: order.append("warm") or Ready(),
    )

    main([])

    assert order == ["cold", "entropy"]


def test_short_seed_is_refused_before_a_cold_download(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.model.model_is_ready", lambda _path=None: False)

    def fail_download(**_kwargs):
        raise AssertionError("download started")

    monkeypatch.setattr("src.render.load_cold_pipeline", fail_download)

    with pytest.raises(SystemExit) as error:
        main(["42"])

    assert error.value.code == 2


def test_interrupt_exits_quietly(fake_render, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def interrupted(argument, stdin=None, stderr=None):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.cli.read_seed", interrupted)

    with pytest.raises(SystemExit) as error:
        main([str(1 << 255)])

    captured = capsys.readouterr()
    assert error.value.code == 130
    assert captured.out == ""
    assert captured.err == ""
    assert fake_render == []


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
    assert call["separate"] is False
    assert len(call["paths"]) == 3
    assert len({path.name for path in call["paths"]}) == 1
    assert call["paths"][0].name == "sheet.png"
    assert len(set(call["prompts"])) == 3
    assert call["width"] == 512
    assert call["height"] == 1024
    assert call["steps"] == 25
    assert call["negative_prompt"] == "text letters label title panels comics captions subtitle"
    assert call["entropy_bits"] == 256.0
    assert call["show_entropy"] is True
    assert len(set(call["seeds"])) == 3
    captured = capsys.readouterr()
    assert captured.out.strip() == str(tmp_path / "sheet.png")
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
            "--out",
            "reading",
        ]
    )

    call = fake_render[0]
    assert call["separate"] is True
    assert [path.name for path in call["paths"]] == ["card_1.png"]
    assert call["width"] == 768
    assert call["height"] == 768
    assert call["steps"] == 10
    assert call["negative_prompt"] == "text letters label title panels comics captions subtitle"
    assert fake_render.out == "reading"


def test_out_must_be_a_folder_name() -> None:
    with pytest.raises(SystemExit):
        main([str(1 << 255), "--out", "nested/name"])


def test_negative_flag_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--negative", "blurry"])


def test_missing_seed_reads_tty(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("src.cli.enter_entropy_tty", lambda _stdin, _stderr: high_entropy_text())

    main([])

    assert len(fake_render[0]["paths"]) == 3
    assert fake_render[0]["show_entropy"] is False


def test_tty_accepts_short_entropy(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("src.cli.enter_entropy_tty", lambda _stdin, _stderr: "short")

    main([])

    assert fake_render[0]["entropy_bits"] == pytest.approx(estimate_seed_entropy("short"))
    assert fake_render[0]["show_entropy"] is False


def test_piped_entropy_is_accepted(fake_render, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO(high_entropy_text()))

    main([])

    assert len(fake_render) == 1
    assert fake_render[0]["show_entropy"] is True


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

    assert first.name == "2026-09-22-165007"
    assert second.name == "2026-09-22-165007-2"
    named = reserve_output_dir(name="reading", root=tmp_path)
    again = reserve_output_dir(name="reading", root=tmp_path)
    assert named.name == "reading"
    assert again.name == "reading-2"
    assert first.is_dir() and named.is_dir() and again.is_dir()


def test_explicit_count_keeps_separate_cards(fake_render, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    main([str(1 << 255), "--count", "3", "--out", "reading"])

    call = fake_render[0]
    assert call["separate"] is True
    assert [path.name for path in call["paths"]] == ["thesis.png", "antithesis.png", "synthesis.png"]
    assert fake_render.out == "reading"
    assert capsys.readouterr().out.strip() == str(tmp_path)


def test_output_file_name(tmp_path) -> None:
    moment = datetime(2026, 9, 22, 16, 50, 7)
    first = reserve_output_file(moment, tmp_path)
    first.touch()
    second = reserve_output_file(moment, tmp_path)

    assert first.name == "2026-09-22-165007.png"
    assert second.name == "2026-09-22-165007-2.png"
    named = reserve_output_file(name="reading", root=tmp_path)
    named.touch()
    again = reserve_output_file(name="reading", root=tmp_path)
    assert named.name == "reading.png"
    assert again.name == "reading-2.png"
    assert first.is_file()


def test_dimensions_must_be_multiples_of_eight() -> None:
    with pytest.raises(SystemExit):
        main([str(1 << 255), "--width", "513"])


def test_progress_bar_is_only_a_bar() -> None:
    empty = render_bar(0, 75).split("\n")
    assert len(empty) == 8
    assert all(line == "░" * 16 for line in empty)

    finished = render_bar(75, 75)
    assert finished.split("\n") == ["█" * 16] * 8
    assert "%" not in finished
    assert "prompt" not in finished
    assert "card" not in finished


def test_progress_bar_fills_pairs_from_the_top_left() -> None:
    lines = render_bar(1, 64).split("\n")
    assert lines[0] == "██" + "░" * 14
    assert all(line == "░" * 16 for line in lines[1:])

    two = render_bar(2, 64).split("\n")
    assert two[0] == "██" + "░" * 14
    assert two[1] == "██" + "░" * 14
    assert "▀" not in render_bar(3, 64)
    assert "▄" not in render_bar(3, 64)


def test_model_ready_requires_every_file(tmp_path) -> None:
    assert model_is_ready(tmp_path) is False
    for name in REQUIRED_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    assert model_is_ready(tmp_path) is True


def test_minimum_entropy_constant() -> None:
    assert MIN_ENTROPY_BITS == 256
