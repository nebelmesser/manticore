from __future__ import annotations

import io

from src.model import SnapshotBar
from src.progress import CELL_COUNT
from src.render import WEIGHT_LOADS, ColdLoad, _WeightBar, _accelerator


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


def test_snapshot_bar_reports_reconstructed_bytes_only() -> None:
    seen = []
    report = lambda done, total: seen.append((done, total))
    bar = SnapshotBar(desc="Reconstructing files", unit="B", total=100, callback=report)
    transfer = SnapshotBar(desc="Downloading bytes", unit="B", total=100, callback=report)

    bar.update(40)
    transfer.update(10)

    assert seen == [(0, 100), (40, 100)]


def test_cold_bar_moves_from_download_into_weight_loading() -> None:
    cold = ColdLoad(__import__("pathlib").Path("."), io.StringIO())
    cold._bar = type("Bar", (), {"set_done": lambda self, done: setattr(cold, "shown", done)})()

    cold._on_download(1, 2)
    after_download = cold.shown
    assert 0 < after_download < CELL_COUNT

    list(_WeightBar(range(4), desc="Loading weights", on_step=cold._on_weight))
    assert cold.shown > after_download
    assert len(cold._bars) == 1

    cold.note_device()
    assert cold.shown == CELL_COUNT
    assert WEIGHT_LOADS == 3
