from pathlib import Path

# Project directories, all built from the repo root so nothing is hard-coded.
# On the LinuxONE VM data/raw, data/processed and models/llm are symlinks.
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CONFIGS_DIR = ROOT_DIR / "configs"
