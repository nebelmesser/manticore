# scry

Techno-divination with Stable Diffusion 2.1. Entropy becomes a prompt and a seed.

```bash
./scry
./scry "enough entropy to pass the gate"
```

Without an argument, scry asks you to enter entropy and shows the accumulated bits.
Generation starts at 256 bits. A shorter seed is refused.

The default is three cards, 512×1024, 25 steps. Generation shows the input
entropy above one progress bar for all images. Files land in `outputs/YYYY-MM-DD-HHMMSS/`, or in
`outputs/<name>/` when `--out` is set. Three cards are
saved as `thesis.png`, `antithesis.png`, and `synthesis.png`. Each one has a
black border — 20px around the picture and 40px along the bottom — and a
centered white Vidaloka caption. Any other count is saved as `card_1.png`,
`card_2.png`, and so on.

```bash
./scry "…" --count 3 --width 512 --height 1024 --steps 25 --out reading
```

The negative prompt is a line from `negative.txt`. One line is used as written.
Several lines contribute one of them at random.

The first run creates `.venv`, installs dependencies, and downloads SD 2.1
if the weights are not already present, showing download progress.
