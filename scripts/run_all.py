"""Run the whole pipeline: fetch data, build features, train.

Usage: uv run python scripts/run_all.py [--top-n 40] [--skip-fetch]

Each step is cached, so a rerun after a successful fetch takes about a minute.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run(name, *args):
    print(f"\n=== {name} ===", flush=True)
    start = time.time()
    result = subprocess.run([sys.executable, str(SCRIPTS / name), *args])
    if result.returncode != 0:
        sys.exit(f"{name} failed (exit {result.returncode}), stopping")
    print(f"--- {name} done in {time.time() - start:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", default="40", help="overflows to model")
    parser.add_argument(
        "--skip-fetch", action="store_true", help="use the data already in data/raw"
    )
    args = parser.parse_args()

    if not args.skip_fetch:
        run("fetch_data.py", "--top-n", args.top_n)
    run("build_features.py")
    run("train.py")


if __name__ == "__main__":
    main()
