import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from triton.config import PROCESSED

DATE_COLUMNS = {"timestamp", "start", "end"}


class RawStore:
    """Saves downloads byte for byte and records where each file came from."""

    def __init__(self, root, force=False):
        self.root = Path(root)
        self.force = force
        self.manifest_path = self.root / "manifest.json"
        self.manifest = (
            json.loads(self.manifest_path.read_text())
            if self.manifest_path.exists()
            else {}
        )

    def has(self, relpath):
        return not self.force and (self.root / relpath).exists()

    def save(self, relpath, content, url, params=None):
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        self.manifest[str(Path(relpath).as_posix())] = {
            "url": url,
            "params": params or {},
            "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "bytes": len(content),
        }
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1))
        return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_csv(df, folder, name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(folder) / f"{name}.csv", index=False)


def load(name, folder=PROCESSED):
    """Reads a pipeline CSV and restores date columns as UTC timestamps."""
    df = pd.read_csv(Path(folder) / f"{name}.csv")
    for col in DATE_COLUMNS.intersection(df.columns):
        df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
    return df
