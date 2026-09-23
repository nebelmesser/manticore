from __future__ import annotations

from src.render import _accelerator


class _Torch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, cuda: bool, mps: bool) -> None:
        self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: cuda)})()
        self.backends = type("Backends", (), {})()
        self.backends.mps = type("Mps", (), {"is_available": staticmethod(lambda: mps)})()


def test_cuda_is_preferred_over_mps() -> None:
    assert _accelerator(_Torch(True, True)) == ("cuda", "float16")
    assert _accelerator(_Torch(False, True)) == ("mps", "float16")
    assert _accelerator(_Torch(False, False)) == ("cpu", "float32")


def test_cold_start_prints_download_details(monkeypatch) -> None:
    import io
    from pathlib import Path

    from src.render import load_cold_pipeline

    notes = io.StringIO()
    def load(path, announce=None):
        if announce is not None:
            announce.write("Loading Stable Diffusion 2.1 on cuda\n")
        return None, "cuda"

    monkeypatch.setattr("src.render.ensure_model", lambda path: path)
    monkeypatch.setattr("src.render._load_pipeline", load)
    monkeypatch.setattr("src.render._quiet_libraries", lambda: None)

    ready = load_cold_pipeline(Path("models/sd21"), notes)
    text = notes.getvalue()

    assert "sd2-community/stable-diffusion-2-1" in text
    assert "models/sd21" in text
    assert "Loading Stable Diffusion 2.1 on cuda" in text
    assert ready.device == "cuda"
