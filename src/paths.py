from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LEXICON = DATA_DIR / "fragments.txt"
MODEL_DIR = ROOT / "models" / "sd21"
OUTPUTS_DIR = ROOT / "outputs"
FONT_PATH = ROOT / "fonts" / "Vidaloka-Regular.ttf"
