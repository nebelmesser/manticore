from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import warnings
from pathlib import Path
from typing import Callable, Sequence, TextIO

from src.cards import DIVINATION_NAME, TRIAD, frame_card, join_cards
from src.entropy import format_input_entropy
from src.model import SD21_REPO, ensure_model
from src.paths import MODEL_DIR
from src.progress import NARROW, VIDEO, ProgressBar
from src.prompt import GeneratedPrompt


GUIDANCE_SCALE = 7.5


def _quiet_libraries() -> None:
    os.environ["DIFFUSERS_VERBOSITY"] = "error"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TQDM_DISABLE"] = "1"
    warnings.filterwarnings("ignore")
    for name in ("diffusers", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from diffusers.utils.logging import disable_progress_bar as disable_diffusers_progress_bar
    from huggingface_hub.utils import disable_progress_bars
    from transformers.utils.logging import disable_progress_bar as disable_transformers_progress_bar
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    disable_diffusers_progress_bar()
    disable_transformers_progress_bar()
    transformers_logging.set_verbosity_error()


class PipelineLoad:
    """Load Stable Diffusion while entropy is being collected."""

    def __init__(self, model_dir: Path, quiet_download: bool) -> None:
        self.model_dir = model_dir
        self.quiet_download = quiet_download
        self.pipe = None
        self.device = "cpu"
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> PipelineLoad:
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            if self.quiet_download:
                _quiet_libraries()
                ensure_model(self.model_dir)
            else:
                ensure_model(self.model_dir)
                _quiet_libraries()
            self.pipe, self.device = _load_pipeline(self.model_dir)
        except BaseException as exc:
            self.error = exc

    def result(self) -> tuple:
        self._thread.join()
        if self.error is not None:
            raise self.error
        return self.pipe, self.device


def start_pipeline(model_dir: Path = MODEL_DIR, quiet_download: bool = False) -> PipelineLoad:
    return PipelineLoad(model_dir, quiet_download).start()


class ReadyPipeline:
    def __init__(self, pipe, device: str) -> None:
        self.pipe = pipe
        self.device = device

    def result(self) -> tuple:
        return self.pipe, self.device


def load_cold_pipeline(model_dir: Path = MODEL_DIR, stream: TextIO | None = None) -> ReadyPipeline:
    """Download with the library bars, then load weights. Hilbert stays for generation."""

    output = stream if stream is not None else sys.stderr
    hidden = os.environ.pop("TQDM_DISABLE", None)
    try:
        output.write(f"Downloading Stable Diffusion 2.1 from {SD21_REPO}\ninto {model_dir}\n")
        output.flush()
        ensure_model(model_dir)
        pipe, device = _load_pipeline(model_dir, announce=output)
    finally:
        if hidden is not None:
            os.environ["TQDM_DISABLE"] = hidden
    _quiet_libraries()
    return ReadyPipeline(pipe, device)


def _accelerator(torch) -> tuple[str, object]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _empty_cache(torch, device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def _load_pipeline(model_dir: Path, announce: TextIO | None = None):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device, dtype = _accelerator(torch)
    if announce is not None:
        announce.write(f"Loading Stable Diffusion 2.1 on {device}\n")
        announce.flush()
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir),
        variant="fp16",
        use_safetensors=True,
        local_files_only=True,
        dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe, device


def _card_record(
    card: GeneratedPrompt,
    *,
    index: int,
    sd_seed: int,
    steps: int,
    width: int,
    height: int,
    negative_prompt: str,
    caption: str | None,
) -> dict:
    return {
        "prompt": card.prompt,
        "negative_prompt": negative_prompt,
        "source_seed": card.seed,
        "seed": sd_seed,
        "steps": steps,
        "guidance": GUIDANCE_SCALE,
        "width": width,
        "height": height,
        "card": index,
        "caption": caption,
        "model": "stable-diffusion-2-1",
    }


def _write_png(image, path: Path, payload) -> None:
    from PIL.PngImagePlugin import PngInfo

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = PngInfo()
    metadata.add_text("sd_parameters", json.dumps(payload, ensure_ascii=False))
    image.save(path, pnginfo=metadata)


def _save_card(
    image,
    path: Path,
    card: GeneratedPrompt,
    *,
    index: int,
    sd_seed: int,
    steps: int,
    width: int,
    height: int,
    negative_prompt: str,
):
    caption = path.stem.upper()
    if caption in TRIAD:
        image = frame_card(image, caption)
    else:
        caption = None
    _write_png(
        image,
        path,
        _card_record(
            card,
            index=index,
            sd_seed=sd_seed,
            steps=steps,
            width=width,
            height=height,
            negative_prompt=negative_prompt,
            caption=caption,
        ),
    )
    return image


def denoise_passes(steps: int, *, video: bool) -> int:
    """UNet steps for one card. Video frames are full runs of 2..steps."""

    if not video or steps < 2:
        return steps
    return steps * (steps + 1) // 2 - 1


def _show_panel(image, caption: str | None):
    if caption is None:
        return image
    return frame_card(image, caption)


def _captions(count: int) -> tuple[str | None, ...]:
    if count == len(TRIAD):
        return TRIAD
    return tuple(None for _ in range(count))


def step_sheets(card_steps: Sequence[Sequence[Path]], dest: Path, captions: Sequence[str | None]) -> list[Path]:
    """Join one finished step-count of every card into one triptych frame."""

    from PIL import Image

    dest.mkdir(parents=True, exist_ok=True)
    length = max(len(steps) for steps in card_steps)
    sheets = []
    for step in range(length):
        panels = []
        for index, steps in enumerate(card_steps):
            image = Image.open(steps[min(step, len(steps) - 1)])
            caption = captions[index] if index < len(captions) else None
            panels.append(_show_panel(image, caption))
        sheet = panels[0] if len(panels) == 1 else join_cards(panels)
        path = dest / f"{step:05d}.png"
        sheet.convert("RGB").save(path)
        sheets.append(path)
    return sheets


def render_cards(
    jobs: Sequence[tuple[GeneratedPrompt, Path]],
    *,
    width: int,
    height: int,
    steps: int,
    negative_prompt: str,
    entropy_bits: float,
    model_dir: Path = MODEL_DIR,
    progress: TextIO | None = None,
    prepare: Callable[[], tuple] | None = None,
    show_entropy: bool = True,
    separate: bool = True,
    video_fps: int | None = None,
    forward_only: bool = False,
) -> None:
    import torch

    from src.video import frames_directory, require_ffmpeg, reset_directory, video_destination, write_ping_pong_video

    stream = progress if progress is not None else sys.stderr
    frames_dir: Path | None = None
    if video_fps is not None:
        require_ffmpeg()
        frames_dir = reset_directory(frames_directory(video_destination(jobs, separate)))
    if show_entropy:
        stream.write("\r" + format_input_entropy(entropy_bits))
        stream.flush()
    passes = denoise_passes(steps, video=video_fps is not None)
    bar = ProgressBar(len(jobs) * passes, stream, VIDEO if video_fps is not None else NARROW)
    pipe = None
    device = "cpu"
    try:
        if prepare is None:
            ensure_model(model_dir)
            _quiet_libraries()
            pipe, device = _load_pipeline(model_dir)
        else:
            pipe, device = prepare()
        saved = []
        records = []
        card_steps: list[list[Path]] = []
        captions = _captions(len(jobs))
        progress_done = 0

        def generate(card: GeneratedPrompt, step_count: int):
            nonlocal progress_done
            done_base = progress_done

            def on_step(_pipe, step, _timestep, callback_kwargs, done_base=done_base):
                bar.set_done(done_base + step + 1)
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(card.sd_seed)
            with torch.inference_mode():
                image = pipe(
                    prompt=card.prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=step_count,
                    guidance_scale=GUIDANCE_SCALE,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
            progress_done += step_count
            return image

        full_frames: list[Path] = []
        raw_images = []
        for image_index, (card, _path) in enumerate(jobs):
            image = generate(card, steps)
            raw_images.append(image)
            if frames_dir is not None and steps >= 2:
                panel_name = captions[image_index].lower() if captions[image_index] else f"{image_index:02d}"
                panel_dir = frames_dir / panel_name
                panel_dir.mkdir()
                frame_path = panel_dir / f"{steps - 2:05d}.png"
                image.convert("RGB").save(frame_path)
                full_frames.append(frame_path)
        for image_index, (card, path) in enumerate(jobs):
            image = raw_images[image_index]
            if separate:
                image = _save_card(
                    image,
                    path,
                    card,
                    index=image_index + 1,
                    sd_seed=card.sd_seed,
                    steps=steps,
                    width=width,
                    height=height,
                    negative_prompt=negative_prompt,
                )
            elif len(jobs) == len(TRIAD):
                caption = TRIAD[image_index]
                image = frame_card(image, caption)
                records.append(
                    _card_record(
                        card,
                        index=image_index + 1,
                        sd_seed=card.sd_seed,
                        steps=steps,
                        width=width,
                        height=height,
                        negative_prompt=negative_prompt,
                        caption=caption,
                    )
                )
            saved.append(image)
        if separate and len(saved) == len(TRIAD):
            join_cards(saved).save(jobs[0][1].parent / DIVINATION_NAME)
        elif not separate:
            _write_png(join_cards(saved), jobs[0][1], {"cards": records})
        if full_frames:
            for image_index, (card, _path) in enumerate(jobs):
                panel_dir = full_frames[image_index].parent
                card_frames = []
                for index, step_count in enumerate(range(2, steps)):
                    preview = generate(card, step_count)
                    frame_path = panel_dir / f"{index:05d}.png"
                    preview.convert("RGB").save(frame_path)
                    card_frames.append(frame_path)
                card_frames.append(full_frames[image_index])
                card_steps.append(card_frames)
            bar.set_done(progress_done)
        if card_steps and video_fps is not None:
            sheets_dir = video_destination(jobs, separate).with_name(
                f"{video_destination(jobs, separate).stem}-sheets"
            )
            sheets = step_sheets(card_steps, sheets_dir, captions)
            try:
                write_ping_pong_video(
                    sheets,
                    video_destination(jobs, separate),
                    video_fps,
                    forward_only=forward_only,
                )
            finally:
                shutil.rmtree(sheets_dir, ignore_errors=True)
        bar.finish()
    finally:
        bar.close()
        if pipe is not None:
            _empty_cache(torch, device)
