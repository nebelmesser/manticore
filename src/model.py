from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.paths import MODEL_DIR


SD21_REPO = "sd2-community/stable-diffusion-2-1"
SD21_PATTERNS = [
    "model_index.json",
    "feature_extractor/*",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
REQUIRED_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def model_is_ready(path: Path = MODEL_DIR) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in REQUIRED_FILES)


class SnapshotBar:
    """Silent tqdm stand-in. Byte reconstruction updates the shared callback."""

    def __init__(self, *args, total=None, initial=0, desc=None, unit=None, **kwargs) -> None:
        self.desc = desc or ""
        self.unit = unit
        self.n = initial or 0
        self.format_dict: dict = {}
        self._total = 0
        self._callback: Callable[[int, int], None] | None = kwargs.get("callback")
        self._track = self.unit == "B" and str(self.desc).startswith("Reconstruct")
        self.total = total or 0

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value) -> None:
        self._total = int(value or 0)
        self._report()

    def update(self, n=1) -> None:
        self.n += int(n or 0)
        self._report()

    def refresh(self) -> None:
        self._report()

    def set_description(self, desc=None, refresh=True) -> None:
        self.desc = desc or ""

    def set_description_str(self, desc=None, refresh=True) -> None:
        self.desc = desc or ""

    def set_postfix_str(self, postfix="", refresh=True) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> SnapshotBar:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _report(self) -> None:
        if self._track and self._callback is not None and self._total > 0:
            self._callback(self.n, self._total)


def download_model(
    path: Path = MODEL_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    class _Bar(SnapshotBar):
        def __init__(self, *args, **kwargs) -> None:
            kwargs["callback"] = progress
            super().__init__(*args, **kwargs)

    snapshot_download(
        repo_id=SD21_REPO,
        local_dir=path,
        allow_patterns=SD21_PATTERNS,
        tqdm_class=_Bar,
    )


def ensure_model(
    path: Path = MODEL_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    if model_is_ready(path):
        return path
    path.mkdir(parents=True, exist_ok=True)
    download_model(path, progress=progress)
    if not model_is_ready(path):
        raise SystemExit(f"SD 2.1 model download is incomplete: {path}")
    return path
