import json
from pathlib import Path

_here = Path(__file__).parent

# every script reads its settings from these files
settings = json.loads((_here / "llm" / "settings.json").read_text())
thresholds = json.loads((_here / "agent" / "thresholds.json").read_text())
