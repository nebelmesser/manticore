from __future__ import annotations

from pathlib import Path

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


def download_model(path: Path = MODEL_DIR) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=SD21_REPO,
        local_dir=path,
        allow_patterns=SD21_PATTERNS,
    )


def ensure_model(path: Path = MODEL_DIR) -> Path:
    if model_is_ready(path):
        return path
    path.mkdir(parents=True, exist_ok=True)
    download_model(path)
    if not model_is_ready(path):
        raise SystemExit(f"SD 2.1 model download is incomplete: {path}")
    return path
