from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
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
from src.prompt import (
    GeneratedPrompt,
    LexiconError,
    choose_negative,
    estimate_seed_entropy,
    generate_prompt,
)
from src.seeds import card_parts, parse_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scry",
        description="Techno-divination: entropy becomes Stable Diffusion 2.1 cards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "seed",
        nargs="?",
        help="integer or text of at least 256 bits; omit it to type any entropy and press Enter",
    )
    parser.add_argument("--count", type=int, default=3, help="number of cards to generate")
    parser.add_argument("--width", type=int, default=512, help="width, a multiple of 8")
    parser.add_argument("--height", type=int, default=1024, help="height, a multiple of 8")
    parser.add_argument("--steps", type=int, default=25, help="denoising steps")
    parser.add_argument(
        "--out",
        help="output folder name under outputs/; a timestamp is used when omitted",
    )
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
    if args.out is not None and (args.out in {"", ".", ".."} or "/" in args.out or "\\" in args.out):
        parser.error("--out must be a folder name")


def reserve_output_dir(
    moment: datetime | None = None,
    root: Path = OUTPUTS_DIR,
    name: str | None = None,
) -> Path:
    stamp = name if name is not None else (moment or datetime.now()).strftime("%Y-%m-%d-%H%M%S")
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
            return enter_entropy_tty(stdin, stderr)
        else:
            seed = stdin.read()
        require_entropy(seed)
        return seed
    except NotEnoughEntropy as error:
        print(not_enough_entropy_message(error.bits), file=stderr)
        raise SystemExit(2) from error


def cards_for(seed: int | str, count: int) -> list[GeneratedPrompt]:
    cards = []
    for offset in range(count):
        prompt_seed, sd_seed = card_parts(seed, offset)
        cards.append(replace(generate_prompt(prompt_seed), sd_seed=sd_seed))
    return cards


def render_cards(
    jobs: Sequence[tuple[GeneratedPrompt, Path]],
    *,
    width: int,
    height: int,
    steps: int,
    negative_prompt: str,
    entropy_bits: float,
    prepare=None,
    show_entropy: bool = True,
) -> None:
    from src.render import render_cards as render

    render(
        jobs,
        width=width,
        height=height,
        steps=steps,
        negative_prompt=negative_prompt,
        entropy_bits=entropy_bits,
        prepare=prepare,
        show_entropy=show_entropy,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    interactive = args.seed is None and sys.stdin.isatty()
    if interactive:
        os.environ["TQDM_DISABLE"] = "1"
    from src.render import start_pipeline

    loader = start_pipeline(quiet_download=interactive)
    seed = read_seed(args.seed)
    try:
        cards = cards_for(seed, args.count)
        negative_prompt = choose_negative()
    except LexiconError as error:
        print(f"scry: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    output_dir = reserve_output_dir(name=args.out)
    jobs = [
        (card, output_dir / card_filename(index, args.count))
        for index, card in enumerate(cards, start=1)
    ]
    render_cards(
        jobs,
        width=args.width,
        height=args.height,
        steps=args.steps,
        negative_prompt=negative_prompt,
        entropy_bits=estimate_seed_entropy(seed),
        prepare=loader.result,
        show_entropy=not interactive,
    )
    print(output_dir)
