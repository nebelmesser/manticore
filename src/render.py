from __future__ import annotations

import json
import logging
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Callable, Sequence, TextIO

from src.cards import DIVINATION_NAME, TRIAD, frame_card, join_cards
from src.entropy import format_input_entropy
from src.model import ensure_model
from src.paths import MODEL_DIR
from src.progress import ProgressBar
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


def _load_pipeline(model_dir: Path):
    import torch
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
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
) -> None:
    import torch

    stream = progress if progress is not None else sys.stderr
    if show_entropy:
        stream.write("\r" + format_input_entropy(entropy_bits))
        stream.flush()
    bar = ProgressBar(len(jobs) * steps, stream)
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
        for image_index, (card, path) in enumerate(jobs):
            sd_seed = card.sd_seed
            generator = torch.Generator(device="cpu").manual_seed(sd_seed)

            def on_step(_pipe, step, _timestep, callback_kwargs, image_index=image_index):
                bar.set_done(image_index * steps + step + 1)
                return callback_kwargs

            with torch.inference_mode():
                image = pipe(
                    prompt=card.prompt,
                    negative_prompt=negative_prompt or None,
                    num_inference_steps=steps,
                    guidance_scale=GUIDANCE_SCALE,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=on_step,
                    callback_on_step_end_tensor_inputs=[],
                ).images[0]
            if separate:
                image = _save_card(
                    image,
                    path,
                    card,
                    index=image_index + 1,
                    sd_seed=sd_seed,
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
                        sd_seed=sd_seed,
                        steps=steps,
                        width=width,
                        height=height,
                        negative_prompt=negative_prompt,
                        caption=caption,
                    )
                )
            saved.append(image)
            bar.set_done((image_index + 1) * steps)
        if separate and len(saved) == len(TRIAD):
            join_cards(saved).save(jobs[0][1].parent / DIVINATION_NAME)
        elif not separate:
            _write_png(join_cards(saved), jobs[0][1], {"cards": records})
        bar.finish()
    finally:
        bar.close()
        if pipe is not None and device == "mps":
            torch.mps.empty_cache()
