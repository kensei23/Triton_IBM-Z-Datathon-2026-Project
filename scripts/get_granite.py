"""Download IBM Granite 4.0 350M (instruct) for the alert writer.

Usage: uv run python scripts/get_granite.py [--dir models/llm]

IBM publishes little-endian GGUF files. IBM Z (s390x) is big-endian, so on a
big-endian machine the file is byte-swapped with the `gguf` package's
gguf-convert-endian tool (run with `uv tool run`, so not a project dependency)
and saved under the big-endian name from configs/llm/alerts.json.
The model file is never committed (see .gitignore).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import requests

from triton.alerts import alert_settings
from triton.constants import MODELS_DIR


def download(url, path):
    tmp = path.with_suffix(".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%} of {total >> 20} MB", end="")
    print()
    tmp.rename(path)


def main(folder):
    cfg = alert_settings()
    folder.mkdir(parents=True, exist_ok=True)
    little = folder / cfg["chat_model"]
    big = folder / cfg["chat_model_big_endian"]
    target = big if sys.byteorder == "big" else little
    if target.exists():
        print(f"already there: {target}")
        return

    if not little.exists():
        print(f"downloading {cfg['chat_model_url']}")
        download(cfg["chat_model_url"], little)

    if sys.byteorder == "big":
        print("big-endian machine (IBM Z): converting byte order, takes a minute")
        subprocess.run(
            [
                "uv",
                "tool",
                "run",
                "--from",
                "gguf",
                "gguf-convert-endian",
                str(little),
                "big",
            ],
            input="YES\n",
            text=True,
            check=True,
        )
        little.rename(big)
    print(f"ready: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=MODELS_DIR / "llm")
    main(parser.parse_args().dir)
