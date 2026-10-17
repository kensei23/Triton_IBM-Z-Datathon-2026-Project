import json
from pathlib import Path

# every script reads its settings from the settings json
_path = Path(__file__).parent / "llm" / "settings.json"
settings = json.loads(_path.read_text())
