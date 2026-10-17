import json
import os
from pathlib import Path

_here = Path(__file__).parent
_root = _here.parent

# every script reads its settings from these files
settings = json.loads((_here / "llm" / "settings.json").read_text())
thresholds = json.loads((_here / "agent" / "thresholds.json").read_text())

# env var lets each machine point at its own model folder, otherwise the path
# in settings.json is treated as relative to the repo root
_model_dir = Path(os.environ.get("TRITON_MODEL_DIR", settings["model_dir"]))
_model_dir = _model_dir.expanduser()
if not _model_dir.is_absolute():
    _model_dir = _root / _model_dir
settings["model_dir"] = str(_model_dir)
