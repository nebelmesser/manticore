from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
POSITIVE = DATA_DIR / "positive.txt"
NEGATIVE = DATA_DIR / "negative.txt"
MODEL_DIR = ROOT / "models" / "sd21"
OUTPUTS_DIR = ROOT / "outputs"
FONT_PATH = ROOT / "assets" / "Vidaloka-Regular.ttf"
