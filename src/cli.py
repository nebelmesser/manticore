from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

from src.cards import card_filename
from src.entropy import (
    NotEnoughEntropy,
    enter_entropy_tty,
    not_enough_entropy_message,
    require_entropy,
)
from src.paths import OUTPUTS_DIR
from src.prompt import GeneratedPrompt, LexiconError, generate_prompt
from src.seeds import parse_seed, sequence_seed


DEFAULT_NEGATIVE = "text letters label title panels comics captions subtitle"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scry",
        description="Techno-divination: entropy becomes Stable Diffusion 2.1 cards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "seed",
        nargs="?",
        help="integer or text of at least 256 bits; omit it to type entropy",
    )
    parser.add_argument("--count", type=int, default=3, help="number of cards to generate")
    parser.add_argument("--width", type=int, default=512, help="width, a multiple of 8")
    parser.add_argument("--height", type=int, default=1024, help="height, a multiple of 8")
    parser.add_argument("--steps", type=int, default=25, help="denoising steps")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE, help="negative prompt")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    for name in ("width", "height"):
        value = getattr(args, name)
        if value <= 0 or value % 8:
            parser.error(f"--{name} must be a positive multiple of 8")


def reserve_output_dir(moment: datetime | None = None, root: Path = OUTPUTS_DIR) -> Path:
    stamp = (moment or datetime.now()).strftime("%Y-%m-%d-%H-%M-%S")
    candidate = root / stamp
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def read_seed(
    argument: str | None,
    stdin: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int | str:
    if stdin is None:
        stdin = sys.stdin
    if stderr is None:
        stderr = sys.stderr
    try:
        if argument is not None:
            seed = parse_seed(argument)
        elif stdin.isatty():
            seed = enter_entropy_tty(stdin, stderr)
        else:
            seed = stdin.read()
        require_entropy(seed)
        return seed
    except NotEnoughEntropy as error:
        print(not_enough_entropy_message(error.bits), file=stderr)
        raise SystemExit(2) from error


def cards_for(seed: int | str, count: int) -> list[GeneratedPrompt]:
    return [generate_prompt(sequence_seed(seed, offset)) for offset in range(count)]


def render_cards(
    jobs: Sequence[tuple[GeneratedPrompt, Path]],
    *,
    width: int,
    height: int,
    steps: int,
    negative_prompt: str,
) -> None:
    from src.render import render_cards as render

    render(
        jobs,
        width=width,
        height=height,
        steps=steps,
        negative_prompt=negative_prompt,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    seed = read_seed(args.seed)
    try:
        cards = cards_for(seed, args.count)
    except LexiconError as error:
        print(f"scry: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    output_dir = reserve_output_dir()
    jobs = [
        (card, output_dir / card_filename(index, args.count))
        for index, card in enumerate(cards, start=1)
    ]
    render_cards(
        jobs,
        width=args.width,
        height=args.height,
        steps=args.steps,
        negative_prompt=args.negative,
    )
    print(output_dir)
